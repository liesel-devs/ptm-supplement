import logging
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
    "--ntrain", type=int, default=20000, help="Number of training observations."
)
@click.option("--warmup", type=int, default=1000, help="Number of warmup iterations.")
@click.option(
    "--posterior", type=int, default=400, help="Number of posterior iterations."
)
@click.option("--jobid", type=str, required=True, default="test")
@click.option(
    "--jobdir",
    type=str,
    required=True,
    default="scaling/010-ptm",
)
@click.option("--jobrow", type=int, required=True, default=0)
@click.option("--mcmc_strategy", type=str, required=True, default="iwls-nuts")
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
            thinning = 1

    if testing:
        thinning = 1

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
    data_path = Path(jobdir) / ".." / ".." / "data" / "sim"
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

    data_seed = 1
    logger.info(f"Loading {data_seed=}, {data_type=}")
    train = load_data(data_seed, data_type, "train")
    while train.shape[0] < ntrain:
        data_seed += 1
        logger.info(f"Loading {data_seed=}, {data_type=}")
        train = pd.concat([train, load_data(data_seed, data_type, "train")])

    logger.info(f"{train.shape=}")

    test = load_data(data_seed, data_type, "test")

    train = train.iloc[:ntrain, :].reset_index(drop=True)
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
    # mod.graph.plot_vars(save_path=str(Path(jobdir) / "mod.png"), width=20, height=12)

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

    aprobs_list = []
    tinfos = results.get_posterior_transition_infos()
    for param in mod.graph.parameters:
        kernel = results.kernels_by_pos_key.expect(None)[param]
        aprob = tinfos[kernel].acceptance_prob.mean()
        posmoved = tinfos[kernel].position_moved.mean()
        aprobs_dict = {}
        aprobs_dict["acceptance_prob"] = aprob
        aprobs_dict["position_moved"] = posmoved
        aprobs_dict["kernel"] = results.kernel_classes.expect(None)[kernel].__name__
        aprobs_dict["variable"] = param
        aprobs_list.append(aprobs_dict)

    aprobs_summary = pd.DataFrame(aprobs_list)

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

    diagnostics = pd.merge(diagnostics, aprobs_summary, how="left")

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
    meval = ptm.EvaluatePTM(mod, samples)
    key, subkey = jax.random.split(key)
    crps = meval.crps_sample(
        key=subkey,
        predictive_samples_n=1,
        newdata=newdata,
        subsamples_n=min(1000, posterior),
        n_chunk=100,
    )

    # ..............................................................................
    # ---- Summary of distribution analysis ----
    # ..............................................................................

    dist_summary = pd.DataFrame(
        {
            "waic": waic,
            "kld": kld,
            "log_score": log_score,
            "crps": crps,
        },
        index=[0],  # type: ignore
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
    dist_summary["mcmc_strategy"] = mcmc_strategy
    dist_summary["apply_jitter"] = apply_jitter

    errors["data_type"] = data_type
    errors["data_seed"] = data_seed
    errors["model"] = model
    errors["ntrain"] = ntrain
    errors["ntest"] = ntest
    errors["job"] = job
    errors["run"] = tid
    errors["mcmc_strategy"] = mcmc_strategy
    errors["apply_jitter"] = apply_jitter

    diagnostics["data_type"] = data_type
    diagnostics["data_seed"] = data_seed
    diagnostics["model"] = model
    diagnostics["ntrain"] = ntrain
    diagnostics["ntest"] = ntest
    diagnostics["job"] = job
    diagnostics["run"] = tid
    diagnostics["mcmc_strategy"] = mcmc_strategy
    diagnostics["apply_jitter"] = apply_jitter

    # ..............................................................................
    # ---- Write results to disk ----
    # ..............................................................................

    identifier = f"{model}-{mcmc_strategy}-n{ntrain}.csv"

    fp_dist = out_path_dist / ("dist-" + identifier)
    fp_diagnostics = out_path / "diagnostics" / ("diagnostics-" + identifier)
    fp_errors = out_path_errors / ("errors-" + identifier)

    fp_diagnostics.parent.mkdir(exist_ok=True, parents=True)

    dist_summary.to_csv(fp_dist, index=False)
    errors.to_csv(fp_errors, index=False)
    diagnostics.to_csv(fp_diagnostics, index=False)

    finfile.touch()


if __name__ == "__main__":
    run_one()
