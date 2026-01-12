import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import click
import jax
import jax.numpy as jnp
import liesel.goose as gs
import liesel.model as lsl
import liesel_gam as gam
import liesel_ptm as ptm
import numpy as np
import pandas as pd
import smoothcon as sc
import tensorflow_probability.substrates.jax.bijectors as tfb
from liesel.contrib.splines import equidistant_knots

jax.config.update("jax_enable_x64", True)

model = "ptm-iwls-nuts"


@click.command()
@click.option(
    "--data_seed",
    type=int,
    required=True,
    help="Random seed for data generation.",
    default=2,
)
@click.option(
    "--data_type",
    type=str,
    required=True,
    help="Type of data to use.",
    default="gaussian",
)
@click.option("--ntest", type=int, default=5000, help="Number of test observations.")
@click.option(
    "--ntrain", type=int, default=1000, help="Number of training observations."
)
@click.option("--warmup", type=int, default=5000, help="Number of warmup iterations.")
@click.option(
    "--posterior", type=int, default=5_000, help="Number of posterior iterations."
)
@click.option("--jobid", type=str, required=True, default="test")
@click.option(
    "--jobdir",
    type=str,
    required=True,
    default="simulation/010-ptm-plots",
)
@click.option("--jobrow", type=int, required=True, default=0)
@click.option("--mcmc_strategy", type=str, required=True, default="iwls_fixed")
@click.option("--testing", type=bool, required=True, default=False)
@click.option("--apply_jitter", type=bool, required=True, default=False)
def run_one(
    data_seed,
    data_type,
    ntest,
    ntrain,
    warmup,
    posterior,
    jobid,
    jobdir,
    jobrow,
    mcmc_strategy,
    testing,
    apply_jitter,
):
    thinning = 1
    match mcmc_strategy:
        case "iwls_fixed" | "iwls-iwls_fixed":
            thinning = 10
        case "iwls-nuts":
            thinning = 5

    if testing:
        thinning = 1

    sys.path.append(jobdir)
    import utils

    logger = logging.getLogger(Path(jobdir).name)
    logger.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    if not logger.handlers:
        logger.addHandler(sh)

    lptm_logger = logging.getLogger("liesel_ptm")
    lptm_logger.setLevel(logging.DEBUG)
    if not lptm_logger.handlers:
        lptm_logger.addHandler(sh)

    finished = Path(jobdir) / "finished"
    finished.mkdir(parents=True, exist_ok=True)
    finfile = finished / str(jobrow)

    if finfile.exists():
        raise RuntimeError("Run is already finished")

    # Define paths
    data_path = Path(jobdir) / ".." / "data"
    out_path = Path(jobdir) / "out"
    out_path_dist = out_path / "dist"
    out_path_covariates = out_path / "covariates"
    out_path_errors = out_path / "errors"

    # Create directories if they don't exist
    out_path.mkdir(parents=True, exist_ok=True)
    out_path_dist.mkdir(parents=True, exist_ok=True)
    out_path_covariates.mkdir(parents=True, exist_ok=True)
    out_path_errors.mkdir(parents=True, exist_ok=True)

    def load_data(data_seed, data_type, train_or_test):
        data_filename = f"{data_type}-{data_seed:03d}.csv"
        data_filepath = data_path / data_type / train_or_test / data_filename
        return pd.read_csv(data_filepath)

    train = load_data(data_seed, data_type, "train")
    test = load_data(data_seed, data_type, "test")

    train = train.iloc[:ntrain, :]
    test = test.iloc[:ntest, :]

    key = jax.random.key(data_seed)

    # ..............................................................................
    # ---- Model ----
    # ..............................................................................

    a = -4.0
    b = 4.0
    nparam = 30
    knots = ptm.LogIncKnots(a, b, nparam=nparam)
    mod = ptm.LocScalePTM(
        response=train["y"].to_numpy(),
        knots=knots.knots,
        intercepts="pseudo_sample",
        to_float32=False,
    )

    trafo_scale = ptm.ScaleWeibull(
        value=1.0,
        scale=0.5,
        name="trafo0_scale",
        bijector=tfb.Exp(),
    )

    logger.warning(f"{trafo_scale.name}: {trafo_scale.value}")
    logger.warning(
        f"{trafo_scale.variance_param.name}: {trafo_scale.variance_param.value}"
    )
    logger.warning(
        f"{trafo_scale.variance_param.value_node[0].name}: {trafo_scale.variance_param.value_node[0].value}"
    )

    mod.trafo += ptm.PTMCoef.new_rw1_sumzero(
        knots=knots.knots,
        scale=trafo_scale,
        name="trafo0",
    )

    xknots = np.asarray(equidistant_knots(jnp.array([-2.0, 2.0]), n_param=20))
    sf = sc.SmoothFactory(train)
    smooths = []

    for i in range(4):
        smooth = sf(
            f"s(x{i}, bs='ps', k=20)",
            knots=xknots,
            diagonal_penalty=True,
            absorb_cons=True,
            scale_penalty=True,
        )
        smooths.append(smooth)

        x = train[f"x{i}"].to_numpy()
        basis = lsl.Var.new_obs(smooth(x), name=f"B(x{i})")

        mod.loc += gam.SmoothTerm.new_ig(
            basis=basis,
            penalty=smooth.penalty,
            ig_concentration=1.0,
            ig_scale=0.001,
            name=f"s(x{i})",
            variance_value=10.0,
        )

        mod.scale += gam.SmoothTerm.new_ig(
            basis=basis,
            penalty=smooth.penalty,
            ig_concentration=1.0,
            ig_scale=0.001,
            name=f"g(x{i})",
            variance_value=10.0,
        )

    logger.info("Building model")
    mod.build()

    # ..............................................................................
    # ---- Pre-optimization ----
    # ..............................................................................

    logger.info("Initialization")
    v = mod.graph.vars["trafo0_scale"].variance_param.value_node[0]
    logger.warning(f"Value of {v.name}: {v.value}")
    mod.initialize(
        stopper=gs.Stopper(max_iter=5_000, patience=50),
        test_for_positive_definiteness=True,
    )

    # ..............................................................................
    # ---- Sampling ----
    # ..............................................................................

    logger.info("Sampling")
    tic = time.time()
    results = mod.run_mcmc(
        seed=data_seed,
        warmup=warmup,
        posterior=thinning * posterior,
        thinning_posterior=thinning,
        num_chains=4,
        strategy=mcmc_strategy,
        warm_start=False,
        apply_jitter=apply_jitter,
    )
    toc = time.time()

    samples = results.get_posterior_samples()

    newdata = {}
    newdata["response"] = test["y"].to_numpy()
    for i in range(4):
        smooth = smooths[i]
        newdata[f"B(x{i})"] = smooth(test[f"x{i}"].to_numpy())  # type: ignore

    key, subkey = jax.random.split(key)
    rgrid = jnp.linspace(min(test["r"].min(), -4.0), max(test["r"].max(), 4.0), 150)
    r_dens_summary_samples = mod.summarise_trafo_by_samples(
        key=subkey,
        grid=rgrid,
        samples=samples,
        n=50,
    )

    r_dens_summary = mod.summarise_dist(
        samples=samples,
        loc=0.0,
        scale=1.0,
        grid=rgrid,
    )

    r_dens_summary_df = pd.DataFrame(
        {"pdf_hat": r_dens_summary["prob"].mean(axis=(0, 1))}
    )
    r_dens_summary_df["low"] = jnp.quantile(r_dens_summary["prob"], 0.05, axis=(0, 1))
    r_dens_summary_df["high"] = jnp.quantile(r_dens_summary["prob"], 0.95, axis=(0, 1))
    r_dens_summary_df["r"] = rgrid

    # ..............................................................................
    # ---- Location shift on test data ----
    # ..............................................................................

    newdata = {}
    for i in range(4):
        smooth = smooths[i]
        newdata[f"B(x{i})"] = smooth(test[f"x{i}"].to_numpy()[:100])  # type: ignore

    logger.info("Meanfuns")

    loc_dfs = [
        utils.covariate_df(
            "s",
            i,
            scale=False,
            newdata=newdata,
            test=test.iloc[:100, :],
            samples=samples,
        )
        for i in range(4)
    ]
    loc_summary = pd.concat([tup[0] for tup in loc_dfs], ignore_index=True)
    loc_samples_summary = pd.concat([tup[1] for tup in loc_dfs], ignore_index=True)

    # ..............................................................................
    # ---- Scaling on test data ----
    # ..............................................................................

    logger.info("Scalefuns")
    scale_dfs = [
        utils.covariate_df(
            "g",
            i,
            scale=False,
            newdata=newdata,
            test=test.iloc[:100, :],
            samples=samples,
        )
        for i in range(4)
    ]
    scale_summary = pd.concat([tup[0] for tup in scale_dfs], ignore_index=True)
    scale_samples_summary = pd.concat([tup[1] for tup in scale_dfs], ignore_index=True)

    # ..............................................................................
    # ---- Save run information ----
    # ..............................................................................
    logger.info("Saving")
    job = Path(jobdir).name
    tid = datetime.now().strftime("%Y%m%d-%H%M%S")

    summaries = {
        "r_dens_summary": r_dens_summary_df,
        "r_dens_summary_samples": r_dens_summary_samples,
        "loc_summary": loc_summary,
        "loc_samples_summary": loc_samples_summary,
        "scale_summary": scale_summary,
        "scale_samples_summary": scale_samples_summary,
    }

    for name, summary in summaries.items():
        summary["data_type"] = data_type
        summary["data_seed"] = data_seed
        summary["model"] = model
        summary["ntrain"] = ntrain
        summary["ntest"] = ntest
        summary["fit_seconds"] = toc - tic
        summary["job"] = job
        summary["run"] = tid
        summary["mcmc_strategy"] = mcmc_strategy
        summary["apply_jitter"] = apply_jitter
        identifier = f"{model}-{data_type}-{data_seed:03d}-n{ntrain}.csv"
        out_path_df = out_path / name
        out_path_df.mkdir(exist_ok=True, parents=True)
        fp = out_path_df / (name + "-" + identifier)
        summary.to_csv(fp, index=False)

    finfile.touch()


if __name__ == "__main__":
    run_one()
