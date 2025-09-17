from pathlib import Path
import pandas as pd
import jax.numpy as jnp
import time
from datetime import datetime
import argparse

import liesel.goose as gs
import liesel_ptm as ptm
from liesel_ptm import ps, term
import jax

parser = argparse.ArgumentParser()
parser.add_argument("--jobdir", type=str, default="application-dbbmi/002-gaussian")
parser.add_argument("--jobrow", type=int, default=0)
parser.add_argument("--testing", type=bool, default=True)
args, _ = parser.parse_known_args()


MODEL = "gaussian"


jobdir = Path(args.jobdir)
jobrow = args.jobrow

params = pd.read_csv(jobdir / "params.csv").iloc[jobrow, :].to_dict()

THINNING = params["thinning"]
WARMUP = params["warmup"]
POSTERIOR = params["posterior"]

cache_path = None
if args.testing:
    THINNING = 1
    WARMUP = 200
    POSTERIOR = 20
    cache_path = Path(jobdir) / "results.pickle"

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
    test = data.iloc[0, :]  # effectively no test data

key = jax.random.key(params["fold"])

# ..............................................................................
# ---- Model ----
# ..............................................................................

mod = ptm.LocScalePTM.new_gaussian(
    response=train["bmi"].to_numpy(),
)

ps_age = ps(train["age"], nbases=20, xname="age")  # P-Spline basis
mod.loc += term.f_ig(ps_age, ig_concentration=1.0, ig_scale=0.001, fname="s")
mod.scale += term.f_ig(ps_age, ig_concentration=1.0, ig_scale=0.001, fname="g")


mod.build()

# ..............................................................................
# ---- Pre-optimization ----
# ..............................................................................

# manual initialization step
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
    num_chains=4,
    warm_start=False,
    apply_jitter=False,
    cache_path=cache_path,
)
toc = time.time()

samples = results.get_posterior_samples()

# ..............................................................................
# ---- Diagnostics ----
# ..............................................................................

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

# ..............................................................................
# ---- Log score on test data ----
# ..............................................................................
newdata = {}
newdata["age"] = test["age"].to_numpy()

meval = ptm.EvaluatePTM(mod, samples)  # type: ignore
log_score = meval.log_score(newdata=newdata | {"response": test["bmi"].to_numpy()})

# ..............................................................................
# ---- WAIC ----
# ..............................................................................
waic = float(meval.waic()["waic_deviance"].iloc[0])

# ..............................................................................
# ---- CRPS ----
# ..............................................................................
key, subkey = jax.random.split(key)
crps = meval.crps_sample(
    key=subkey,
    predictive_samples_n=1,
    newdata=newdata | {"response": test["bmi"].to_numpy()},
    subsamples_n=min(1000, POSTERIOR),
    n_chunk=500,
)

# ..............................................................................
# ---- Summary of distribution analysis ----
# ..............................................................................
dist_summary = pd.DataFrame(
    {
        "waic": waic,
        "log_score": log_score,
        "crps": crps,
    },
    index=[0],  # type: ignore
)

# ..............................................................................
# ---- Quantile curves data ----
# ..............................................................................
if ALL_DATA:  # only for full dataset
    age_grid = jnp.linspace(data["age"].min(), data["age"].max(), 150)  # type: ignore
    newdata_grid = {"age": age_grid}

    dist = mod.init_dist(samples, newdata=newdata_grid)
    probs = jnp.expand_dims(
        jnp.array([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]), (1, 2, 3)
    )
    pred_quantiles = dist.quantile(probs).mean(axis=(1, 2))

    quantiles = pd.DataFrame(
        pred_quantiles.T, columns=pd.Index([str(p) for p in probs.squeeze()])
    )
    quantiles["age_std"] = age_grid
    quantiles = quantiles.reset_index(names="n")

    quantiles = quantiles.melt(
        id_vars=["n", "age_std"], value_name="bmi_quantile", var_name="prob_level"
    )


# ..............................................................................
# ---- Location effect data ----
# ..............................................................................
if ALL_DATA:  # only for full dataset
    loc_samples = mod.loc.predict(samples, newdata=newdata_grid)

    loc_summary = (
        gs.SamplesSummary.from_array(loc_samples, name="loc")
        .to_dataframe()
        .reset_index()
    )
    loc_summary["age_std"] = age_grid

# ..............................................................................
# ---- Scale effect data ----
# ..............................................................................
if ALL_DATA:  # only for full dataset
    scale_samples = mod.scale.predict(samples, newdata=newdata_grid)
    scale_summary = (
        gs.SamplesSummary.from_array(scale_samples, name="scale")
        .to_dataframe()
        .reset_index()
    )

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
}

if ALL_DATA:
    summaries["quantile_curves"] = quantiles
    summaries["loc_summary"] = loc_summary
    summaries["scale_summary"] = scale_summary

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
