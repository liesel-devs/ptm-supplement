import time
from datetime import datetime
from pathlib import Path

import click
import jax
import jax.numpy as jnp
import liesel.goose as gs
import liesel_bctm as bctm
import pandas as pd
from liesel_ptm.waic import waic as waic_fn

model = "bctm-te"


@click.command()
@click.option(
    "--data_seed",
    type=int,
    required=True,
    help="Random seed for data generation.",
    default=1,
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
    "--ntrain", type=int, default=100, help="Number of training observations."
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
    default="sim2/005-bctm-te",
)
@click.option("--jobrow", type=int, required=True, default=0)
def run_one(
    data_seed, data_type, ntest, ntrain, warmup, posterior, jobid, jobdir, jobrow
):
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

    # ..............................................................................
    # ---- Model ----
    # ..............................................................................

    ymin = min(train["y"].min(), test["y"].min())
    ymax = max(train["y"].max(), test["y"].max())

    ctmb = (
        bctm.CTMBuilder(train)
        .add_intercept()
        .add_trafo_teprod_full(
            "y",
            "x0",
            (8, 8),
            a=1.0,
            b=0.001,
            positive_tranformation=jnp.exp,
            name="yx0",
            knot_boundaries=((ymin, ymax), (-2.0, 2.0)),
        )
        .add_trafo_teprod_full(
            "y",
            "x1",
            (8, 8),
            a=1.0,
            b=0.001,
            positive_tranformation=jnp.exp,
            name="yx1",
            knot_boundaries=((ymin, ymax), (-2.0, 2.0)),
        )
        .add_trafo_teprod_full(
            "y",
            "x2",
            (8, 8),
            a=1.0,
            b=0.001,
            positive_tranformation=jnp.exp,
            name="yx2",
            knot_boundaries=((ymin, ymax), (-2.0, 2.0)),
        )
        .add_trafo_teprod_full(
            "y",
            "x3",
            (8, 8),
            a=1.0,
            b=0.001,
            positive_tranformation=jnp.exp,
            name="yx3",
            knot_boundaries=((ymin, ymax), (-2.0, 2.0)),
        )
        .add_response("y")
    )

    ctm_model = ctmb.build_model()

    # ..............................................................................
    # ---- Sampling ----
    # ..............................................................................

    eb = gs.EngineBuilder(data_seed, num_chains=4)

    eb.set_model(gs.LieselInterface(ctm_model))
    eb.set_initial_values(ctm_model.state)

    nuts_params = []
    for group in ctm_model.groups().values():
        if group.sampled_params:  # type: ignore
            nuts_params += group.sampled_params  # type: ignore

    nuts = gs.NUTSKernel(
        nuts_params, da_target_accept=0.9, mm_diag=False, max_treedepth=8
    )
    eb.add_kernel(nuts)

    for group in ctm_model.groups().values():
        for kernel in group.gibbs_kernels():  # type: ignore
            eb.add_kernel(kernel)

        if hasattr(group, "mcmc_kernels"):
            for kernel in group.mcmc_kernels:  # type: ignore
                eb.add_kernel(kernel)

    eb.positions_included += ["z"]

    fast_warmup = 0.5
    fast_warmup_duration = fast_warmup * warmup
    init_duration = int(fast_warmup_duration / 2)
    term_duration = init_duration
    slow_warmup_duration = warmup - init_duration - term_duration
    warmup = slow_warmup_duration + init_duration + term_duration

    epochs = gs.stan_epochs(
        warmup_duration=warmup,
        posterior_duration=posterior,
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
    errors = summary.error_df().reset_index()

    # ..............................................................................
    # ---- KLD and log score on test data ----
    # ..............................................................................

    ytest = jnp.asarray(test["y"].to_numpy())
    newdata = test.loc[:, ["x0", "x1", "x2", "x3"]].to_dict("list")
    newdata = {f"y{k}": (ytest, jnp.asarray(v)) for k, v in newdata.items()}

    ctmp = bctm.ConditionalPredictions(samples, ctmb, **newdata)

    log_prob_samples = ctmp.log_prob()

    nsamples = log_prob_samples.shape[0] * log_prob_samples.shape[1]
    lppd_sum = jax.scipy.special.logsumexp(log_prob_samples, axis=(0, 1))
    lppd_i = lppd_sum - jnp.log(nsamples)

    log_score = -lppd_i.sum()
    kld = jnp.mean(test["log_pdf"].to_numpy() - lppd_i)

    # ..............................................................................
    # ---- WAIC ----
    # ..............................................................................

    ctmp_train = bctm.ConditionalPredictions(samples, ctmb)
    waic = waic_fn(ctmp_train.log_prob())

    waic = float(waic["waic_deviance"].iloc[0])

    # ..............................................................................
    # ---- MAD on test data ----
    # ..............................................................................

    cdf_samples = ctmp.cdf()
    cdf_true = jnp.expand_dims(test["cdf"].to_numpy(), (0, 1))
    cdf_mad = jnp.mean(jnp.abs(cdf_true - cdf_samples))

    cdf_low = jnp.quantile(cdf_samples, 0.05, axis=(0, 1))
    cdf_high = jnp.quantile(cdf_samples, 0.95, axis=(0, 1))

    in_ci = (cdf_low <= cdf_true) * (cdf_true <= cdf_high)
    coverage = jnp.mean(in_ci)
    width = jnp.mean(cdf_high - cdf_low)

    # ..............................................................................
    # ---- Summary of distribution analysis ----
    # ..............................................................................

    dist_summary = pd.DataFrame(
        {
            "waic": waic,
            "kld": kld,
            "log_score": log_score,
            "cdf_mad": cdf_mad,
            "cdf_ci_coverage": coverage,
            "cdf_ci_width": width,
        },
        index=[0],  # type: ignore
    )

    # ..............................................................................
    # ---- Save run information ----
    # ..............................................................................

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

    errors["data_type"] = data_type
    errors["data_seed"] = data_seed
    errors["model"] = model
    errors["ntrain"] = ntrain
    errors["ntest"] = ntest
    errors["job"] = job
    errors["run"] = tid

    # ..............................................................................
    # ---- Write results to disk ----
    # ..............................................................................

    identifier = f"{model}-{data_type}-{data_seed:03d}-n{ntrain}.csv"

    fp_dist = out_path_dist / ("dist-" + identifier)
    fp_errors = out_path_errors / ("errors-" + identifier)

    dist_summary.to_csv(fp_dist, index=False)
    errors.to_csv(fp_errors, index=False)

    finfile.touch()


if __name__ == "__main__":
    run_one()
