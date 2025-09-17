"""
Installation of liesel_bctm::

    pip install https://github.com/liesel-devs/liesel-bctm.git
"""

from pathlib import Path
import pandas as pd
import jax.numpy as jnp
import time
from datetime import datetime
import argparse
import plotnine as p9
import logging
import tensorflow_probability.substrates.jax.distributions as tfd
import properscoring as ps

import liesel.goose as gs
import jax
import liesel_bctm as bctm

from liesel_ptm.waic import waic as waic_fn
from liesel_ptm.var import ScaleInverseGamma
from liesel_ptm.kernel import setup_simple_ig_gibbs
from liesel_ptm import cache_results, util

logger = logging.getLogger("sim")
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())

parser = argparse.ArgumentParser()
parser.add_argument("--jobdir", type=str, default="application-fh/009-bctm")
parser.add_argument("--jobrow", type=int, default=0)
parser.add_argument("--testing", type=bool, default=True)
args, _ = parser.parse_known_args()


jobdir = Path(args.jobdir)
jobrow = args.jobrow

params = pd.read_csv(jobdir / "params.csv").iloc[jobrow, :].to_dict()

MODEL = params["model"]

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

train = data[data["fold"] != params["fold"]]
test = data[data["fold"] == params["fold"]]

ALL_DATA = params["fold"] == -1
if ALL_DATA:
    train = data
    test = data.iloc[0, :]  # effectively no test data


# ..............................................................................
# ---- Model ----
# ..............................................................................

ymin = data["cholst"].min()
ymax = data["cholst"].max()
xmin = data["age"].min()
xmax = data["age"].max()

ctmb = bctm.CTMBuilder(train)
ctmb = ctmb.add_intercept()

ctmb = ctmb.add_trafo_teprod_full(
    "cholst",
    "age",
    (9, 9),
    a=1.0,
    b=0.001,
    positive_tranformation=jnp.exp,
    name="te_cholst_age",
    knot_boundaries=((ymin, ymax), (xmin, xmax)),
)

ctmb = ctmb.add_linear_const("sex", name="sex")

if "ri" in MODEL:
    ri_tau = ScaleInverseGamma(
        value=1.0,
        concentration=1.0,
        scale=0.01,
        name="person_ri_tau",
    )
    ctmb = ctmb.add_random_intercept(
        "newid",
        tau=ri_tau,
        name="person_ri",
    )

ctmb = ctmb.add_response("cholst")

ctm_model = ctmb.build_model()

# ctm_model.plot_vars(save_path=jobdir / "model.png")

# ..............................................................................
# ---- Sampling ----
# ..............................................................................

eb = gs.EngineBuilder(params["fold"], num_chains=4)

interface = gs.LieselInterface(ctm_model)
eb.set_model(interface)
eb.set_initial_values(ctm_model.state)

nuts_params = []
for group in ctm_model.groups().values():
    if group.sampled_params:  # type: ignore
        nuts_params += group.sampled_params  # type: ignore

nuts = gs.NUTSKernel(nuts_params, da_target_accept=0.9, mm_diag=False, max_treedepth=9)
eb.add_kernel(nuts)

for group in ctm_model.groups().values():
    for kernel in group.gibbs_kernels():  # type: ignore
        eb.add_kernel(kernel)

    if hasattr(group, "mcmc_kernels"):
        for kernel in group.mcmc_kernels:  # type: ignore
            if kernel.position_keys[0] == ri_tau.variance_param.name:
                # don't include a NUTS kernel for the random intercept variance;
                # we add a Gibbs kernel below.
                continue
            eb.add_kernel(kernel)

if "ri" in MODEL:
    ndim = ctm_model.vars["person_ri_coef"].value.size

    gibbs_ri_tau2 = setup_simple_ig_gibbs(
        name=ri_tau.variance_param.name,
        interface=interface,
        a=ri_tau.variance_param.dist_node["concentration"].value,
        b=ri_tau.variance_param.dist_node["concentration"].value,
        penalty=jnp.eye(ndim),
        rank=ndim,
        coef="person_ri_coef",
    )
    eb.add_kernel(gibbs_ri_tau2)

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


tic = time.time()
results = cache_results(eb, jobdir / "results.pickle", use_cache=False)
toc = time.time()

samples = results.get_posterior_samples()

summary = gs.Summary(results, deselected=["z"])


# ..............................................................................
# ---- WAIC ----
# ..............................................................................
logger.info("Computing WAIC")
ctmp_train = bctm.ConditionalPredictions(samples, ctmb)
waic = waic_fn(ctmp_train.log_prob())

waic = float(waic["waic_deviance"].iloc[0])

# ..............................................................................
# ---- Patching RandomIntercept class for marginal predictions ----
# ..............................................................................

if "ri" in MODEL:

    def _u32_checksum(x):
        # Cheap, deterministic checksum that’s JIT/vmap friendly.
        # Avoids Python-side RNG/state. Works for float or int leaves.
        x = jnp.asarray(x)
        x = jnp.reshape(x, (-1,))
        # bring floats to an integer grid to stabilize the checksum
        xi = jnp.asarray(jnp.round(x * 1e6), dtype=jnp.int32)
        acc = jnp.uint32(0)
        acc ^= jnp.uint32(jnp.sum(xi))
        acc ^= jnp.uint32(xi.size)
        return acc

    def _tree_salt(tree):
        leaves = jax.tree_util.tree_leaves(tree)
        # Reduce with XOR across leaves
        salts = jnp.array([_u32_checksum(leaf) for leaf in leaves], dtype=jnp.uint32)
        return jnp.bitwise_xor.reduce(salts)

    def ppeval_ri(samples, x):
        """
        Draw one random intercept sample for the given posterior draw.

        Notes
        -----
        This function uses the provided PRNG `x` together with a
        checksum ("salt") derived from the contents of `samples`
        to construct a lane-unique random key. Under `vmap`, each
        (chain, sample) combination has different `samples` values,
        so the derived salt differs and `fold_in` produces distinct
        subkeys. This avoids the problem that, if all lanes received
        the same `key`, the randomness would otherwise be identical.
        """
        # Make the key lane-unique based on the (c, s)-specific samples
        key = x
        salt = _tree_salt(samples)
        key = jax.random.fold_in(key, salt)

        ri_scale_samples = jnp.sqrt(samples["person_ri_tau_square"])

        ri_sample = tfd.Normal(loc=0.0, scale=ri_scale_samples).sample(1, seed=key)
        return ri_sample[..., None]

    ctmb.pt[-1].ppeval = ppeval_ri


# ..............................................................................
# ---- CRPS ----
# ..............................................................................
N_NEWSAMPLES_PER_POSTERIOR_SAMPLE_CRPS = 1

if "ri" in MODEL:
    newid_basis_test = ctmb.pt[-1].basis_fn(test["newid"].to_numpy())

ngrid = test.shape[0]
ygrid = jnp.linspace(ymin, ymax, 500)
smooths_list = []
key = jax.random.key(1)
for i in range(ngrid):
    newdata = {}
    newdata["sex"] = test["sex"].to_numpy()[i]
    newdata["te_cholst_age"] = (ygrid, test["age"].to_numpy()[i])
    if "ri" in MODEL:
        key, subkey = jax.random.split(key)
        newdata["person_ri"] = subkey
    smooths_list.append(newdata)

pred_samples = bctm.summary.trafo_csample(
    key=jax.random.key(params["fold"]),
    n=N_NEWSAMPLES_PER_POSTERIOR_SAMPLE_CRPS,
    samples=samples,
    smooths_list=smooths_list,
    ygrid=ygrid,
    builder=ctmb,
)

nsamp, c, s, ntest = pred_samples.shape
pred_samples = jnp.reshape(pred_samples, shape=(nsamp * c * s, ntest))


def crps_streaming(pred_samples, observations, n_chunk):
    """
    Compute CRPS in chunks along the leading axis of pred_samples.
    Memory-safe computation for many samples when using ps.crps_ensemble.

    Parameters
    ----------
    pred_samples : jnp.ndarray, shape (nsamp, ntest)
        Ensemble predictions. Leading axis will be chunked.
    observations : array-like, shape (ntest,)
        Observed values to compare against.
    n_chunk : int
        Number of ensemble members per chunk.

    Returns
    -------
    Mean CRPS across all observations.
    """
    ntest = observations.shape[0]
    crps_vals = []

    for i in range(0, ntest, n_chunk):
        chunk = pred_samples[:, i : i + n_chunk]  # (nsamples, nchunk)
        crps_chunk = ps.crps_ensemble(
            observations[i : i + n_chunk], chunk.T
        )  # shape (nchunk,)
        crps_vals.append(crps_chunk.mean())

    return jnp.mean(jnp.stack(crps_vals))


crps = crps_streaming(pred_samples, test["cholst"].to_numpy(), 500)

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
# ---- Quantile curves ----
# ..............................................................................
age_grid = jnp.linspace(data["age"].min(), data["age"].max(), 100)  # type: ignore
age_grid = jnp.r_[age_grid, age_grid]
sex_grid = jnp.full_like(age_grid, fill_value=0.0)
sex_grid = sex_grid.at[100:].set(1.0)

ngrid = age_grid.size
ygrid = jnp.linspace(ymin, ymax, 500)
smooths_list = []
key = jax.random.key(1)
for i in range(ngrid):
    newdata = {}
    newdata["sex"] = sex_grid[i]
    newdata["te_cholst_age"] = (ygrid, age_grid[i])
    if "ri" in MODEL:
        key, subkey = jax.random.split(key)
        newdata["person_ri"] = subkey
    smooths_list.append(newdata)

nsamples = 5
pred_samples = bctm.summary.trafo_csample(
    key=jax.random.key(params["fold"]),
    n=nsamples,
    samples=util.subsample_tree(
        jax.random.key(params["fold"]), samples, min(POSTERIOR, 1000)
    ),
    smooths_list=smooths_list,
    ygrid=ygrid,
    builder=ctmb,
)


probs = jnp.array([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
pred_quantiles = jnp.quantile(pred_samples, q=probs, axis=(0, 1, 2))
del pred_samples

quantiles = pd.DataFrame(
    pred_quantiles.T, columns=pd.Index([str(p) for p in probs.squeeze()])
)
quantiles["age_std"] = age_grid
quantiles["sex"] = sex_grid
quantiles["sex"] = pd.Categorical(quantiles["sex"])
quantiles = quantiles.reset_index(names="n")

quantiles = quantiles.melt(
    id_vars=["n", "age_std", "sex"], value_name="cholst_quantile", var_name="prob_level"
)

quantiles_sorted = quantiles.sort_values(["sex", "prob_level", "age_std"])

#
quantiles_sorted["cholst_rollmean"] = quantiles_sorted.groupby(["sex", "prob_level"])[
    "cholst_quantile"
].transform(lambda s: s.rolling(window=5, min_periods=1).mean())


p = (
    p9.ggplot(quantiles_sorted)
    + p9.geom_point(p9.aes("age", "cholst"), data=data, alpha=0.3)
    + p9.geom_line(
        p9.aes("age_std", "cholst_rollmean", group="prob_level"), color="red"
    )
    + p9.labs(title="Quantiles")
    + p9.facet_wrap("~sex")
)

# ..............................................................................
# ---- Save run information ----
# ..............................................................................
logger.info("Saving run information")
job = Path(jobdir).name
tid = datetime.now().strftime("%Y%m%d-%H%M%S")

dist_summary["fit_seconds"] = toc - tic

summaries = {
    "dist": dist_summary,
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
