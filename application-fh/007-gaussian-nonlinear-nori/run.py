from pathlib import Path
import pandas as pd
import jax.numpy as jnp
import time
from datetime import datetime
import argparse

import liesel.goose as gs
import liesel_ptm as ptm
import jax
from liesel_ptm import ps, term, lin

jax.config.update("jax_enable_x64", True)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--jobdir", type=str, default="application-fh/007-gaussian-nonlinear-nori"
)
parser.add_argument("--jobrow", type=int, default=0)
parser.add_argument("--testing", type=bool, default=True)
args, _ = parser.parse_known_args()


MODEL = "gaussian-nonlinear-nori"


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
out_path = Path(jobdir) / "out"
out_path.mkdir(parents=True, exist_ok=True)

data = pd.read_csv(data_path / "framingham.csv")

cholst_mean = data["cholst"].mean()
cholst_sd = data["cholst"].std()

data["age_at_start"] = data["age"]
data["age"] = data["age_at_start"] + data["year"]

age_mean = data["age"].mean()
age_sd = data["age"].std()

data["cholst"] = (data["cholst"] - cholst_mean) / cholst_sd
data["age"] = (data["age"] - age_mean) / age_sd
data["sex"] = data["sex"] - 0.5

train = data[data["fold"] != params["fold"]]
test = data[data["fold"] == params["fold"]]

ALL_DATA = params["fold"] == -1
if ALL_DATA:
    train = data
    test = data.iloc[0:2, :]  # effectively no test data


# ..............................................................................
# ---- Model ----
# ..............................................................................

mod = ptm.LocScalePTM.new_gaussian(
    response=train["cholst"].to_numpy(),
    to_float32=False,
)

ps_age = ps(train["age"], nbases=20, xname="age")  # P-Spline basis
mod.loc += term.f_ig(ps_age, ig_concentration=1.0, ig_scale=0.001, fname="s")
mod.scale += term.f_ig(ps_age, ig_concentration=1.0, ig_scale=0.001, fname="g")

lin_sex = lin(train["sex"], xname="sex")  # Linear basis
mod.loc += term.f(lin_sex, fname="s")
mod.scale += term.f(lin_sex, fname="g")

mod.build()

# ..............................................................................
# ---- Pre-optimization ----
# ..............................................................................

_ = mod.initialize(
    stopper=gs.Stopper(max_iter=5_000, patience=50),
    test_for_positive_definiteness=True,
)

# ..............................................................................
# ---- Sampling ----
# ..............................................................................

tic = time.time()
results = mod.run_mcmc(
    seed=params["fold"],
    warmup=WARMUP,
    posterior=THINNING * POSTERIOR,
    thinning_posterior=THINNING,
    num_chains=4,
    warm_start=False,
    apply_jitter=False,
    # cache_path=Path(jobdir) / "results.pickle",
)
toc = time.time()

samples = results.get_posterior_samples()

summary = gs.Summary(results)
errors = summary.error_df().reset_index()

diagnostics = (
    summary.to_dataframe()
    .reset_index()
    .loc[:, ["variable", "rhat", "ess_bulk", "ess_tail"]]
    .groupby("variable", as_index=False)
    .agg(
        ess_bulk_min=("ess_bulk", "min"),
        ess_bulk_median=("ess_bulk", "median"),
        ess_tail_min=("ess_tail", "min"),
        ess_tail_median=("ess_tail", "median"),
        rhat_max=("rhat", "max"),
        rhat_median=("rhat", "median"),
    )
)

seconds = toc - tic
minutes = seconds / 60

diagnostics["ess_bulk_min_per_minute"] = diagnostics["ess_bulk_min"] / minutes
diagnostics["ess_tail_min_per_minute"] = diagnostics["ess_tail_min"] / minutes
diagnostics["ess_bulk_median_per_minute"] = diagnostics["ess_bulk_median"] / minutes
diagnostics["ess_tail_median_per_minute"] = diagnostics["ess_tail_median"] / minutes

# gs.plot_trace(results, "ri(id)_coef", range(175, 181))

# ..............................................................................
# ---- CRPS ----
# ..............................................................................

newdata = {}
# newdata["response"] = test["cholst"].to_numpy()
newdata["sex"] = test["sex"].to_numpy()
newdata["age"] = test["age"].to_numpy()

key = jax.random.key(params["fold"])
meval = ptm.EvaluatePTM(mod, samples)
key, subkey = jax.random.split(key)
crps = meval.crps_sample(
    subkey,
    predictive_samples_n=1,
    newdata=newdata | {"response": test["cholst"].to_numpy()},
    subsamples_n=min(1000, POSTERIOR),
)

# ..............................................................................
# ---- WAIC ----
# ..............................................................................
meval = ptm.EvaluatePTM(mod, samples)
waic = float(meval.waic()["waic_deviance"].iloc[0])


# ..............................................................................
# ---- Summary of distribution analysis ----
# ..............................................................................

dist_summary = pd.DataFrame(
    {
        "waic": waic,
        "crps": crps,
    },
    index=[0],  # type: ignore
)

# ..............................................................................
# ---- Conditional quantiles ----
# ..............................................................................

age_grid = jnp.linspace(data["age"].min(), data["age"].max(), 100)  # type: ignore
age_grid = jnp.r_[age_grid, age_grid]
newdata_grid = {"age": age_grid}
newdata_grid["sex"] = jnp.full_like(age_grid, fill_value=-0.5)
newdata_grid["sex"] = newdata_grid["sex"].at[100:].set(0.5)

dist = mod.init_dist(
    ptm.util.subsample_tree(
        jax.random.key(params["fold"]), samples, min(1000, POSTERIOR)
    ),
    newdata=newdata_grid,
)
pred_samples = dist.sample((5,), seed=jax.random.key(params["fold"]))

probs = jnp.array([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
pred_quantiles = jnp.quantile(pred_samples, q=probs, axis=(0, 1, 2))

quantiles = pd.DataFrame(
    pred_quantiles.T, columns=pd.Index([str(p) for p in probs.squeeze()])
)
quantiles["age_std"] = age_grid
quantiles["sex"] = newdata_grid["sex"]
quantiles["sex"] = pd.Categorical(quantiles["sex"])
quantiles = quantiles.reset_index(names="n")

quantiles = quantiles.melt(
    id_vars=["n", "age_std", "sex"], value_name="cholst_quantile", var_name="prob_level"
)

# ..............................................................................
# ---- Location and scale effects of age ----
# ..............................................................................

newdata_grid.pop("sex")
loc_scale_samples = mod.graph.predict(
    samples, predict=["s(age)", "g(age)"], newdata=newdata_grid
)
loc_samples = loc_scale_samples["s(age)"]
scale_samples = loc_scale_samples["g(age)"]

loc_summary = (
    gs.SamplesSummary.from_array(loc_samples, name="loc").to_dataframe().reset_index()
)
scale_summary = (
    gs.SamplesSummary.from_array(scale_samples, name="scale")
    .to_dataframe()
    .reset_index()
)

loc_summary["age_std"] = age_grid
scale_summary["age_std"] = age_grid


# ..............................................................................
# ---- Save run information ----
# ..............................................................................

job = Path(jobdir).name
tid = datetime.now().strftime("%Y%m%d-%H%M%S")

dist_summary["fit_seconds"] = toc - tic

summaries = {
    "dist": dist_summary,
    "errors": errors,
    "diagnostics": diagnostics,
    "loc_summary": loc_summary,
    "scale_summary": scale_summary,
    "quantiles": quantiles,
}

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
