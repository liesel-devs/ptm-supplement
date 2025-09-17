"""
Installation::

    pip install https://github.com/liesel-devs/liesel-bctm.git
"""

from pathlib import Path
import pandas as pd
import jax.numpy as jnp
import time
from datetime import datetime
import argparse

import liesel.goose as gs
import jax
import liesel_bctm as bctm
import liesel_ptm as ptm

from liesel_ptm.waic import waic as waic_fn

parser = argparse.ArgumentParser()
parser.add_argument("--jobdir", type=str, default="application-db/jobs/003-bctm")
parser.add_argument("--jobrow", type=int, default=0)
parser.add_argument("--testing", type=bool, default=True)
args, _ = parser.parse_known_args()


MODEL = "bctm"


jobdir = Path(args.jobdir)
jobrow = args.jobrow

params = pd.read_csv(jobdir / "params.csv").iloc[jobrow, :].to_dict()

THINNING = params["thinning"]
WARMUP = params["warmup"]
POSTERIOR = params["posterior"]

if args.testing:
    THINNING = 1
    WARMUP = 200
    POSTERIOR = 20

finished = Path(jobdir) / "finished"
finished.mkdir(parents=True, exist_ok=True)
finfile = finished / str(jobrow)

if finfile.exists():
    raise RuntimeError("Run is already finished")

# Define paths
data_path = Path(jobdir) / ".." / ".." / "data"
if args.testing:
    out_path = Path(jobdir) / "out-test"
else:
    out_path = Path(jobdir) / "out"
out_path.mkdir(parents=True, exist_ok=True)

data = pd.read_csv(data_path / "dbbmi.csv")

bmi_mean = data["bmi"].mean()
bmi_sd = data["bmi"].std()
age_mean = data["age"].mean()
age_sd = data["age"].std()

data["bmi"] = (data["bmi"] - bmi_mean) / bmi_sd
data["age"] = (data["age"] - age_mean) / age_sd

train = data[data["fold"] != params["fold"]]
test = data[data["fold"] == params["fold"]]

ALL_DATA = params["fold"] == -1
if ALL_DATA:
    train = data
    test = data.iloc[0:2, :]  # effectively no test data


if args.testing:
    train = train.iloc[:200, :]

# ..............................................................................
# ---- Model ----
# ..............................................................................

ymin = data["bmi"].min()
ymax = data["bmi"].max()
xmin = data["age"].min()
xmax = data["age"].max()
ctmb = (
    bctm.CTMBuilder(train)
    .add_intercept()
    .add_trafo_teprod_full(
        "bmi",
        "age",
        (9, 9),
        a=1.0,
        b=0.001,
        positive_tranformation=jnp.exp,
        name="te_bmi_age",
        knot_boundaries=((ymin, ymax), (xmin, xmax)),
    )
    .add_response("bmi")
)

ctm_model = ctmb.build_model()

# ..............................................................................
# ---- Sampling ----
# ..............................................................................

eb = gs.EngineBuilder(params["fold"], num_chains=4)

eb.set_model(gs.LieselInterface(ctm_model))
eb.set_initial_values(ctm_model.state)

nuts_params = []
for group in ctm_model.groups().values():
    if group.sampled_params:  # type: ignore
        nuts_params += group.sampled_params  # type: ignore

nuts = gs.NUTSKernel(nuts_params, da_target_accept=0.9, mm_diag=False, max_treedepth=8)
eb.add_kernel(nuts)

for group in ctm_model.groups().values():
    for kernel in group.gibbs_kernels():  # type: ignore
        eb.add_kernel(kernel)

    if hasattr(group, "mcmc_kernels"):
        for kernel in group.mcmc_kernels:  # type: ignore
            eb.add_kernel(kernel)

eb.positions_included += ["z"]

fast_warmup = 0.5
fast_warmup_duration = fast_warmup * WARMUP
init_duration = int(fast_warmup_duration / 2)
term_duration = init_duration
slow_warmup_duration = WARMUP - init_duration - term_duration
warmup = slow_warmup_duration + init_duration + term_duration

epochs = gs.stan_epochs(
    warmup_duration=warmup,
    posterior_duration=POSTERIOR,
    thinning_posterior=1,
    thinning_warmup=1,
    init_duration=init_duration,
    term_duration=term_duration,
)
eb.set_epochs(epochs)

engine = eb.build()
tic = time.time()
engine.sample_all_epochs()
toc = time.time()

results = engine.get_results()
samples = results.get_posterior_samples()

summary = gs.Summary(results, deselected=["z"])
# ..............................................................................
# ---- Log score on test data ----
# ..............................................................................
newdata = {"te_bmi_age": (test["bmi"].to_numpy(), test["age"].to_numpy())}
ctmp = bctm.ConditionalPredictions(samples, ctmb, **newdata)
log_prob_samples = ctmp.log_prob()
nsamples = log_prob_samples.shape[0] * log_prob_samples.shape[1]
lppd_sum = jax.scipy.special.logsumexp(log_prob_samples, axis=(0, 1))
lppd_i = lppd_sum - jnp.log(nsamples)
log_score = -lppd_i.sum()

# ..............................................................................
# ---- WAIC ----
# ..............................................................................
ctmp_train = bctm.ConditionalPredictions(samples, ctmb)
waic = waic_fn(ctmp_train.log_prob())

waic = float(waic["waic_deviance"].iloc[0])

# ..............................................................................
# ---- CRPS ----
# ..............................................................................

N_SUBSAMPLES_CRPS = min(POSTERIOR * 4, 1000)
N_NEWSAMPLES_PER_POSTERIOR_SAMPLE_CRPS = 1

subsamples = ptm.util.subsample_tree(
    jax.random.key(params["fold"]), samples, num_samples=N_SUBSAMPLES_CRPS
)

ntest = test.shape[0]
ygrid = jnp.linspace(ymin, ymax, 500)
smooths_list = []
for i in range(ntest):
    smooths = {"te_bmi_age": (ygrid, jnp.asarray(test["age"].iloc[i]))}
    smooths_list.append(smooths)

probs = jnp.linspace(0.005, 0.995, 25)
ypred_q = bctm.summary.trafo_cquantiles(
    probs, samples=subsamples, smooths_list=smooths_list, ygrid=ygrid, builder=ctmb
)
ypred_q = jnp.moveaxis(ypred_q, 0, -1)
crps = bctm.summary.crps(test["bmi"].to_numpy(), ypred_q, probs)

# ..............................................................................
# ---- Summary of distribution analysis ----
# ..............................................................................

dist_summary = pd.DataFrame(
    {"waic": waic, "log_score": log_score, "crps": crps},
    index=[0],  # type: ignore
)

# ..............................................................................
# ---- Quantile Score ----
# ..............................................................................

probs = jnp.array([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
ypred_q = bctm.summary.trafo_cquantiles(
    probs, samples=subsamples, smooths_list=smooths_list, ygrid=ygrid, builder=ctmb
)
ypred_q = jnp.moveaxis(ypred_q, 0, -1)

quantile_score_df = bctm.summary.quantile_score_df(
    test["bmi"].to_numpy(), ypred_q, probs
)

# ..............................................................................
# ---- Quantile curves ----
# ..............................................................................
if ALL_DATA:
    age_grid = jnp.linspace(data["age"].min(), data["age"].max(), 150)  # type: ignore
    ngrid = age_grid.shape[0]
    ygrid = jnp.linspace(ymin, ymax, 500)
    smooths_list = []
    for i in range(ngrid):
        smooths = {"te_bmi_age": (ygrid, age_grid[i])}
        smooths_list.append(smooths)
    probs = jnp.array([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
    ypred_q = bctm.summary.trafo_cquantiles(
        probs, samples=subsamples, smooths_list=smooths_list, ygrid=ygrid, builder=ctmb
    )

    pred_quantiles = ypred_q.mean(axis=(1, 2))

    quantiles = pd.DataFrame(
        pred_quantiles.T, columns=pd.Index([str(p) for p in probs.squeeze()])
    )
    quantiles["age_std"] = age_grid
    quantiles = quantiles.reset_index(names="n")

    quantiles = quantiles.melt(
        id_vars=["n", "age_std"], value_name="bmi_quantile", var_name="prob_level"
    )


# ..............................................................................
# ---- Save run information ----
# ..............................................................................
job = Path(jobdir).name
tid = datetime.now().strftime("%Y%m%d-%H%M%S")

dist_summary["fit_seconds"] = toc - tic

summaries = {
    "dist": dist_summary,
    "quantile_score_df": quantile_score_df,
}

if ALL_DATA:
    summaries["quantile_curves"] = quantiles

for name, df in summaries.items():
    df["model"] = MODEL
    df["job"] = job
    df["run"] = tid
    df["fold"] = params["fold"]


# ..............................................................................
# ---- Write results to disk ----
# ..............................................................................

for name, df in summaries.items():
    identifier = f"{MODEL}-{name}-fold{params['fold']:02d}-row{jobrow}.csv"
    directory = out_path / name
    directory.mkdir(parents=True, exist_ok=True)
    df.to_csv(directory / identifier, index=False)

finfile.touch()
