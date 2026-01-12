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
from liesel.contrib.splines import equidistant_knots

model = "gaussian-iwls"


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
    default="simulation/007-gaussian",
)
@click.option("--jobrow", type=int, required=True, default=0)
@click.option("--thinning", type=int, required=True, default=10)
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
    thinning,
):
    sys.path.append(jobdir)
    import utils

    logger = logging.getLogger(Path(jobdir).name)
    logger.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(sh)

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

    mod = ptm.LocScalePTM.new_gaussian(
        response=train["y"].to_numpy(),
        loc_intercept_inference=gs.MCMCSpec(gs.IWLSKernel),
        scale_intercept_inference=gs.MCMCSpec(gs.IWLSKernel),
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
            variance_value=10.0,
            name=f"s(x{i})",
        )

        mod.scale += gam.SmoothTerm.new_ig(
            basis=basis,
            penalty=smooth.penalty,
            ig_concentration=1.0,
            ig_scale=0.001,
            variance_value=10.0,
            name=f"g(x{i})",
        )

    mod.build()

    # ..............................................................................
    # ---- Pre-optimization ----
    # ..............................................................................

    mod.initialize(
        stopper=gs.Stopper(max_iter=5_000, patience=50),
        test_for_positive_definiteness=True,
    )

    mod.setup_default_mcmc_kernels(
        strategy="iwls-nuts",
        locscale_kernel_kwargs={"initial_step_size": 1.0, "da_target_accept": 0.8},
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
        strategy="manual",
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

    # ..............................................................................
    # ---- KLD and log score on test data ----
    # ..............................................................................

    logger.info("KLD and Log score")
    newdata = test.loc[:, ["y", "x0", "x1", "x2", "x3"]].to_dict("list")
    newdata = {k: jnp.asarray(v) for k, v in newdata.items()}

    newdata = {}
    newdata["response"] = test["y"].to_numpy()
    for i in range(4):
        smooth = smooths[i]
        newdata[f"B(x{i})"] = smooth(test[f"x{i}"].to_numpy())  # type: ignore

    meval = ptm.EvaluatePTM(mod, samples)  # type: ignore

    kld = meval.kld(test["log_pdf"].to_numpy(), newdata=newdata.copy())
    log_score = meval.log_score(newdata=newdata.copy())

    # ..............................................................................
    # ---- WAIC ----
    # ..............................................................................
    logger.info("WAIC")
    waic = float(meval.waic()["waic_deviance"].iloc[0])

    # ..............................................................................
    # ---- CRPS on test data ----
    # ..............................................................................

    logger.info("CRPS")
    key, subkey = jax.random.split(key)
    crps = meval.crps(
        probs=jnp.linspace(0.005, 0.995, 25),
        newdata=newdata | test["y"].to_numpy(),
        k=20,
    ).mean()

    # ..............................................................................
    # ---- MAD on test data ----
    # ..............................................................................
    logger.info("MAD")
    cdf_mad, coverage, width = utils.cdf_mad_and_ci_stream(mod, samples, newdata, test)

    # ..............................................................................
    # ---- Summary of distribution analysis ----
    # ..............................................................................

    dist_summary = pd.DataFrame(
        {
            "waic": waic,
            "kld": kld,
            "log_score": log_score,
            "crps": crps,
            "cdf_mad": cdf_mad,
            "cdf_ci_coverage": coverage,
            "cdf_ci_width": width,
        },
        index=[0],  # type: ignore
    )

    # ..............................................................................
    # ---- Location shift on test data ----
    # ..............................................................................

    logger.info("Meanfuns")
    meanfuns_summary = pd.concat(
        [
            utils.eval_covariate_simple(
                "s", i, scale=False, newdata=newdata, test=test, samples=samples
            )
            for i in range(4)
        ],
        ignore_index=True,
    )

    # ..............................................................................
    # ---- Scaling on test data ----
    # ..............................................................................

    logger.info("Scalefuns")
    scalefuns_summary = pd.concat(
        [
            utils.eval_covariate_simple(
                "g", i, scale=False, newdata=newdata, test=test, samples=samples
            )
            for i in range(4)
        ],
        ignore_index=True,
    )

    covariates_summary = pd.concat(
        [
            meanfuns_summary,
            scalefuns_summary,
        ],
        axis=0,
    )

    # ..............................................................................
    # ---- Save run information ----
    # ..............................................................................
    logger.info("Saving")
    job = Path(jobdir).name
    tid = datetime.now().strftime("%Y%m%d-%H%M%S")

    dist_summary["data_type"] = data_type
    dist_summary["data_seed"] = data_seed
    dist_summary["model"] = model
    dist_summary["ntrain"] = ntrain
    dist_summary["ntest"] = ntest
    dist_summary["fit_seconds"] = toc - tic
    dist_summary["job"] = job
    dist_summary["run"] = tid

    covariates_summary["data_type"] = data_type
    covariates_summary["data_seed"] = data_seed
    covariates_summary["model"] = model
    covariates_summary["ntrain"] = ntrain
    covariates_summary["ntest"] = ntest
    covariates_summary["job"] = job
    covariates_summary["run"] = tid

    errors["data_type"] = data_type
    errors["data_seed"] = data_seed
    errors["model"] = model
    errors["ntrain"] = ntrain
    errors["ntest"] = ntest
    errors["job"] = job
    errors["run"] = tid

    diagnostics["data_type"] = data_type
    diagnostics["data_seed"] = data_seed
    diagnostics["model"] = model
    diagnostics["ntrain"] = ntrain
    diagnostics["ntest"] = ntest
    diagnostics["job"] = job
    diagnostics["run"] = tid

    # ..............................................................................
    # ---- Write results to disk ----
    # ..............................................................................

    identifier = f"{model}-{data_type}-{data_seed:03d}-n{ntrain}.csv"

    fp_dist = out_path_dist / ("dist-" + identifier)
    fp_covariates = out_path_covariates / ("covariates-" + identifier)
    fp_covariates_unscaled = (
        out_path / "covariates_unscaled" / ("covariates-" + identifier)
    )
    fp_diagnostics = out_path / "diagnostics" / ("diagnostics-" + identifier)
    fp_errors = out_path_errors / ("errors-" + identifier)

    fp_covariates_unscaled.parent.mkdir(exist_ok=True, parents=True)
    fp_diagnostics.parent.mkdir(exist_ok=True, parents=True)

    dist_summary.to_csv(fp_dist, index=False)
    covariates_summary.to_csv(fp_covariates, index=False)
    errors.to_csv(fp_errors, index=False)
    diagnostics.to_csv(fp_diagnostics, index=False)

    finfile.touch()


if __name__ == "__main__":
    run_one()
