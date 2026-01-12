import jax.numpy as jnp
import jax
import pandas as pd


def _slice_newdata_for_chunk(nd, sl):
    """Slice first axis for any array in newdata that matches the N dimension; leave others as-is."""
    nd_chunk = {}
    for k, v in nd.items():
        v = jnp.asarray(v)
        # If it looks like an [N, ...] array, slice along axis 0
        if v.ndim >= 1 and sl.stop <= v.shape[0]:
            nd_chunk[k] = v[sl]
        else:
            nd_chunk[k] = v
    return nd_chunk


def cdf_mad_and_ci_stream(mod, samples, newdata, test, *, chunk_size: int = 512):
    """
    Memory-efficient & fast:
      - builds the distribution on CHUNKS of newdata (e.g., B(xi)[sl])
      - vmaps cdf over the chunk of y's
      - peak memory ~ O(S*C*chunk_size), not O(S*C*N)
    """
    # Prepare data
    nd_full = dict(newdata) if newdata is not None else {}
    y = jnp.asarray(nd_full.pop("response"))  # [N]
    cdf_true = jnp.asarray(test["cdf"].to_numpy())  # [N]
    N = y.shape[0]

    # Accumulators
    mad_total = jnp.array(0.0)
    cov_total = jnp.array(0.0)
    width_total = jnp.array(0.0)

    # Process in chunks
    for start in range(0, N, chunk_size):
        stop = min(start + chunk_size, N)
        sl = slice(start, stop)

        # Slice newdata so the model only builds tensors for this chunk
        nd = _slice_newdata_for_chunk(nd_full, sl)
        y_k = y[sl]  # [K]
        true_k = cdf_true[sl]  # [K]
        K = y_k.shape[0]

        # Build the (batched) distribution for this chunk only
        dist_k = mod.init_dist(
            samples, newdata=nd
        )  # should produce shapes only for K rows now

        # If dist.cdf is vectorized over y already and returns [S, C, K], use it directly.
        # Otherwise vmap it over K to get [K, S, C] and then reshape.
        try:
            cdf_sc_k = dist_k.cdf(y_k)  # either [S, C, K] or error if not vectorized
            if cdf_sc_k.ndim == 3:
                # [S, C, K] -> [K, S*C]
                cdf_km = jnp.reshape(jnp.moveaxis(cdf_sc_k, -1, 0), (K, -1))
            else:
                # Unexpected shape, fall back to vmap
                raise TypeError
        except Exception:
            # vmap over K obs: [K, S, C]
            cdf_ksc = jax.vmap(dist_k.cdf)(y_k)
            cdf_km = cdf_ksc.reshape(K, -1)  # [K, S*C]

        # --- MAD over K obs (mean over samples/chains inside each obs)
        mad_k = jnp.mean(jnp.abs(true_k[:, None] - cdf_km), axis=1)  # [K]
        mad_total = mad_total + jnp.sum(mad_k)  # scalar

        # --- 90% CI across samples/chains per obs
        low_k = jnp.quantile(cdf_km, 0.05, axis=1)  # [K]
        high_k = jnp.quantile(cdf_km, 0.95, axis=1)  # [K]
        cov_total = cov_total + jnp.sum((low_k <= true_k) & (true_k <= high_k))
        width_total = width_total + jnp.sum(high_k - low_k)

        # Nothing kept from this chunk beyond accumulated scalars

    Nf = jnp.asarray(N, dtype=mad_total.dtype)
    cdf_mad = mad_total / Nf
    coverage = cov_total / Nf
    width = width_total / Nf
    return float(cdf_mad), float(coverage), float(width)


def evaluate_covariate_effect(xnum: int, f: str, scale: bool, newdata, test, samples):
    suffix = ""
    match f:
        case "s":
            suffix = "loc"
        case "g":
            suffix = "scale"

    # --- Prepare fx (true function on x) and center/scale over N
    fx = test[f"f{xnum}_{suffix}"].to_numpy()
    fx = jnp.asarray(fx)
    if scale:
        fx_centered = (fx - fx.mean()) / fx.std()
    else:
        fx_centered = fx - fx.mean()  # center only
    # shape: [N]
    N = fx_centered.shape[0]

    # --- Inputs for predictions
    B = newdata[f"B(x{xnum})"]  # [N, P]
    B = jnp.asarray(B)
    coef = samples[f"{f}(x{xnum})_coef"]  # [S, C, P]
    coef = jnp.asarray(coef)
    S, C, P = coef.shape

    # Helper: compute prediction for a single observation n: [S, C]
    def pred_at_n(n, coef):
        # tensordot over P: (S,C,P) · (P,) -> (S,C)
        return jnp.tensordot(coef, B[n], axes=([2], [0]))

    # ---------- PASS 1: per-sample mean/std across n (for centering) ----------
    mean0 = jnp.zeros((S, C))
    m2_0 = jnp.zeros((S, C))
    count0 = jnp.array(0, dtype=jnp.int32)

    def body_stats(n, state):
        mean, m2, count = state
        y = pred_at_n(n, coef)  # [S, C]
        count_new = count + 1
        delta = y - mean
        mean_new = mean + delta / count_new
        m2_new = m2 + delta * (y - mean_new)
        return (mean_new, m2_new, count_new)

    mean_sc, m2_sc, ncount = jax.lax.fori_loop(0, N, body_stats, (mean0, m2_0, count0))
    var_sc = m2_sc / jnp.maximum(ncount, 1)  # ddof=0 to mirror jnp.std default
    std_sc = jnp.sqrt(jnp.maximum(var_sc, jnp.finfo(var_sc.dtype).eps))

    # ---------- PASS 2: accumulate metrics without storing [S,C,N] ----------
    bias_sum0 = jnp.array(0.0)
    mse_sum0 = jnp.array(0.0)
    var_sum0 = jnp.array(0.0)
    cov_count0 = jnp.array(0.0)
    width_sum0 = jnp.array(0.0)

    def body_metrics(n, acc):
        bias_sum, mse_sum, var_sum, cov_count, width_sum = acc

        y_sc = pred_at_n(n, coef)  # [S, C]
        y_sc = (y_sc - mean_sc) / std_sc  # center/scale per (S,C) sample
        fx_n = fx_centered[n]  # scalar

        diff = y_sc - fx_n  # [S, C]
        # mean over samples/chains
        bias_sum += diff.mean()  # scalar
        mse_sum += (diff * diff).mean()  # scalar
        var_sum += y_sc.var()  # var across (S,C), ddof=0

        # pointwise CIs across (S,C) for this n
        low_n = jnp.quantile(y_sc, q=0.05)  # scalar (flattened over S,C)
        high_n = jnp.quantile(y_sc, q=0.95)
        cov_count += (low_n <= fx_n) * (fx_n <= high_n)
        width_sum += high_n - low_n

        return (bias_sum, mse_sum, var_sum, cov_count, width_sum)

    bias_sum, mse_sum, var_sum, cov_count, width_sum = jax.lax.fori_loop(
        0, N, body_metrics, (bias_sum0, mse_sum0, var_sum0, cov_count0, width_sum0)
    )

    # Averages across observations
    bias = bias_sum / N
    mse = mse_sum / N
    var_ = var_sum / N
    in_ci = cov_count / N
    width = width_sum / N

    df = pd.DataFrame(
        {
            "xnum": xnum,
            "bias": bias,
            "var": var_,
            "mse": mse,
            "ci_coverage": in_ci,
            "ci_width": width,
            "parameter": "loc",
        },
        index=pd.Index([0]),
    )
    return df


def eval_covariate_simple(f: str, xnum, scale: bool, test, newdata, samples):
    suffix = ""
    match f:
        case "s":
            suffix = "loc"
        case "g":
            suffix = "scale"

    fx = test[f"f{xnum}_{suffix}"].to_numpy()
    if scale:
        fx_centered = (fx - fx.mean()) / fx.std()
    else:
        fx_centered = fx - fx.mean()

    fx_centered = jnp.expand_dims(fx_centered, (0, 1))

    B = newdata[f"B(x{xnum})"]
    coef = samples[f"{f}(x{xnum})_coef"]
    pred = jnp.einsum("np,...p->...n", B, coef)  # [S, C, N]

    if scale:
        pred_centered = (pred - pred.mean(axis=-1, keepdims=True)) / pred.std(
            axis=-1, keepdims=True
        )
    else:
        pred_centered = pred - pred.mean(axis=-1, keepdims=True)

    bias = (pred_centered - fx_centered).mean(axis=(0, 1)).mean()
    var = pred_centered.var(axis=(0, 1)).mean()
    mse = ((pred_centered - fx_centered) ** 2).mean()

    low = jnp.quantile(pred_centered, q=0.05, axis=(0, 1))
    high = jnp.quantile(pred_centered, q=0.95, axis=(0, 1))

    in_ci = ((low <= fx_centered) * (fx_centered <= high)).mean()
    width = (high - low).mean()

    df = pd.DataFrame(
        {
            "xnum": xnum,
            "bias": bias,
            "var": var,
            "mse": mse,
            "ci_coverage": in_ci,
            "ci_width": width,
            "parameter": suffix,
        },
        index=pd.Index([0]),
    )
    return df
