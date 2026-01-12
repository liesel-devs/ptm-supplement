#' Calculates the function s(delta) for a location-scale parametric
#' transformation model.
#'
#' @param shape Vector of shape parameters.
#' @param l Degree of the spline used in the model. Cubic splines (the default)
#'  use l = 3.
#'
#' @return Evaluation of s(delta), a positive real scalar.
#' @export
sfn <- function(shape, l = 3) {
  pm1 <- length(shape)

  shape <- exp(shape)
  numerator <- sum(shape[c(1, pm1)] / 6) +
    sum((5 / 6) * shape[c(2, pm1 - 1)]) +
    sum(shape[-c(1, 2, pm1 - 1, pm1)])

  numerator / (pm1 - l + 1)
}

#' Computes the average slope of a B-spline based on its coefficients.
#'
#' @param knots Vector of spline knots.
#' @param coef Vector of spline coefficients.
#' @param l Order of the spline. The default is `l = 3` for a cubic spline.
#'
#' @return Average slope of the spline, a scalar.
#' @export
avg_slope_general <- function(knots, coef, l = 3) {
  dk <- knots |> diff() |> mean()
  p <- length(coef)

  dcoef <- diff(coef)

  numerator <- sum(dcoef[-c(1, 2, p - 2, p - 1)]) +
    sum(dcoef[c(1, p - 1)] / 6) +
    sum(5 * dcoef[c(2, p - 2)] / 6)

  numerator / (dk * (p - l))
}

#' Computes increasing spline coefficients for a monotonically increasing
#' spline based on an input vector of log-increments in adjacent coeffients.
#'
#' @param log_coef_increments Vector of log-increments in spline coefficients.
#' @param constant Intercept-coefficient. Scalar
#'
#' @return Vector of increasing spline coefficients.
#' @export
mi_coef <- function(log_coef_increments, constant = 0) {
  c(constant, cumsum(exp(log_coef_increments)))
}

#' Draws random samples of the delta (shape) parameters from a random walk prior.
#'
#' @param p Number of parameters in the model.
#' @param tau Scale parameter of the random walk
#' @param nsamples Number of samples of size `p-1` to draw.
#'
#' @return Matrix of size `(p-1) x nsamples`
#' @export
rdelta_rw <- function(p, tau = 0.2, nsamples = 1) {
  D <- diff(diag(p - 1))
  K <- crossprod(D)

  eig <- eigen(K)
  G1 <- eig$vectors[, (p - 1), drop = FALSE]
  G2 <- eig$vectors[, 1:(p - 2)]

  G <- eig$vectors
  Omega.sqrt.inv <- diag(ifelse(eig$values > 1e-12, 1 / sqrt(eig$values), 0))
  Z <- (G %*% Omega.sqrt.inv)[, 1:(p - 2)]
  z <- stats::rnorm(nsamples * (p - 2)) |> matrix(ncol = nsamples)

  delta <- tau * (Z %*% z)

  delta
}

#' Create equidistant knots.
#'
#' @param x Input vector
#' @param p Desired number of parameters.
#' @param l Order of the spline. Defaults to `l=3` for cubic splines.
#' @param extend_range_factor Slightly extends the range over `min(x)` and `max(x)`,
#'      by setting `xlow <- min(x) - (max(x) - min(x)) * extend_range_factor`
#'
#' @return Vector of knots. Includes the outer knots. Will be of length `p + l + 1`.
#' @export
create_knots <- function(x, p = 10, l = 3, extend_range_factor = 0.01) {
  # m: Defines the order of the spline for which the knots are created.
  #    m = 2 means cubic splines.
  # k: Number of parameters.
  # extend_range_factor: Factor to stretch the range of x a little for
  #                      compatibility with splines of order 0

  m <- l - 1

  xrange <- max(x) - min(x)
  xlow <- min(x) - xrange * extend_range_factor
  xup <- max(x) + xrange * extend_range_factor

  dx <- (xup - xlow) / (p - m - 1)
  n_knots <- p + m + 2
  knots <- seq(xlow - dx * (m + 1), xup + dx * (m + 1), length.out = n_knots)

  knots
}

#' Helper functions for smoothly extrapolating a B-spline.
#'
#' These functions provide a smooth transition from the slope at the lowest
#' point of the spline to a straight line.
#'
#' @param x A numeric vector of values at which to evaluate the B-spline
#'   functions or derivatives.
#' @param knots A numeric vector of knot positions.
#' @param coef A numeric vector of spline coefficients.
#' @param target_deriv The slope of the extrapolation at `knots[ord] - eps`. If
#'   `NULL` (default), the average slope of the spline is used.
#' @param eps The distance over which the slope of the B-spline should be
#'   transitioned towards the `target_deriv`.
#' @param ord A positive integer giving the order of the spline function. This
#'   is the number of coefficients in each piecewise polynomial segment, thus a
#'   cubic spline has order 4. Defaults to 4.
#'
#' @return A numeric vector of `length(x)`.
#' @export
extrapolate_left_transition <- function(
  x,
  knots,
  coef,
  target_deriv = NULL,
  eps = 3 * mean(diff(knots)),
  ord = 4
) {
  kmin <- knots[ord]
  deriv_left <- basis_dot_deriv(kmin, knots, coef, ord = ord) |> drop()

  target_deriv <- ifelse(
    is.null(target_deriv),
    avg_slope_general(knots, coef, ord - 1),
    target_deriv
  )

  unshifted_extrapolation <- function(x) {
    polyn <- x * kmin - (x^2) / 2
    t1 <- (target_deriv / eps) * polyn
    t2 <- deriv_left * (x - polyn / eps)

    t1 + t2
  }

  bdot_min <- basis_dot(kmin, knots, coef) |> drop()
  constant <- bdot_min - unshifted_extrapolation(kmin)

  unshifted_extrapolation(x) + constant
}

extrapolate_left_transition_deriv <- function(
  x,
  knots,
  coef,
  target_deriv = NULL,
  eps = 3 * mean(diff(knots)),
  ord = 4
) {
  kmin <- knots[ord]
  deriv_left <- basis_dot_deriv(kmin, knots, coef, ord = ord) |> drop()

  target_deriv <- ifelse(
    is.null(target_deriv),
    avg_slope_general(knots, coef, ord - 1),
    target_deriv
  )

  (1 - (kmin - x) / eps) * deriv_left + (kmin - x) / eps
}


extrapolate_right_transition_deriv <- function(
  x,
  knots,
  coef,
  target_deriv = NULL,
  eps = 3 * mean(diff(knots)),
  ord = 4
) {
  kmax <- sort(knots, decreasing = TRUE)[ord]
  deriv_right <- basis_dot_deriv(kmax, knots, coef) |> drop()
  target_deriv <- ifelse(
    is.null(target_deriv),
    avg_slope_general(knots, coef, ord - 1),
    target_deriv
  )

  (1 - (x - kmax) / eps) * deriv_right + (x - kmax) / eps
}

#' @rdname extrapolate_left_transition
extrapolate_right_transition <- function(
  x,
  knots,
  coef,
  target_deriv = NULL,
  eps = 3 * mean(diff(knots)),
  ord = 4
) {
  kmax <- sort(knots, decreasing = TRUE)[ord]
  deriv_right <- basis_dot_deriv(kmax, knots, coef) |> drop()
  target_deriv <- ifelse(
    is.null(target_deriv),
    avg_slope_general(knots, coef, ord - 1),
    target_deriv
  )

  unshifted_extrapolation <- function(x) {
    polyn <- (x^2) / 2 - x * kmax
    t1 <- (target_deriv / eps) * polyn
    t2 <- deriv_right * (x - polyn / eps)

    t1 + t2
  }

  bdot_max <- basis_dot(kmax, knots, coef) |> drop()
  constant <- bdot_max - unshifted_extrapolation(kmax)

  unshifted_extrapolation(x) + constant
}

#' Evaluates a B-spline or its derivative.
#'
#' @param x A numeric vector of values at which to evaluate the B-spline functions or derivatives.
#' @param knots A numeric vector of knot positions.
#' @param coef A numeric vector of spline coefficients.
#' @param ord A positive integer giving the order of the spline function. This is the number of coefficients in each piecewise polynomial segment, thus a cubic spline has order 4. Defaults to 4.
#'
#' @return A numeric vector of `length(x)`.
#' @export
basis_dot <- function(x, knots, coef, ord = 4) {
  B <- splines::splineDesign(knots, x, ord = ord, outer.ok = TRUE)
  B %*% coef
}

#' @rdname basis_dot
basis_dot_deriv <- function(x, knots, coef, ord = 4) {
  B <- splines::splineDesign(knots, x, ord = ord, derivs = 1, outer.ok = TRUE)
  B %*% coef
}


#' A B-spline with linear extrapolation.
#'
#'
#' @details Between the minimum and maximum knots `knots[ord]` and `sort(knots,
#' decreasing = TRUE)[4]`, the output is given by [basis_dot()]. At the
#' boundaries of this segment, there are transition segments of width `eps`,
#' which provide a smooth transition of the function's derivient towards
#' `target_deriv`. Beyond the transition segments, the function is linear with
#' slope given by `target_deriv`.
#'
#' @param x A numeric vector of values at which to evaluate the B-spline
#'   functions or derivatives.
#' @param knots A numeric vector of knot positions.
#' @param coef A numeric vector of spline coefficients.
#' @param target_deriv The slope of the extrapolation at `knots[ord] - eps`. If
#'   `NULL` (default), the average slope of the spline is used.
#' @param eps The distance over which the slope of the B-spline should be
#'   transitioned towards the `target_deriv`.
#' @param ord A positive integer giving the order of the spline function. This
#'   is the number of coefficients in each piecewise polynomial segment, thus a
#'   cubic spline has order 4. Defaults to 4.
#'
#' @seealso [extrapolate_left_transition()]
#' @seealso [extrapolate_right_transition()]
#'
#' @return A numeric vector of `length(x)`.
#' @export
#'
#' @examples
#' p <- 20
#' x <- seq(-4, 4, length.out = 300)
#' knots <- create_knots(x, p)
#' coef <- rdelta_rw(p, tau = 1) |>
#'   drop() |>
#'   mi_coef()
#'
#' fx <- basis_dot_extrap(x, knots, coef)
basis_dot_extrap <- function(
  x,
  knots,
  coef,
  target_deriv = NULL,
  eps = 3 * mean(diff(knots)),
  ord = 4
) {
  knots <- sort(knots)
  avg_slope <- avg_slope_general(knots, coef)
  kmin <- knots[ord]
  kmax <- sort(knots, decreasing = TRUE)[ord]

  target_deriv <- ifelse(
    is.null(target_deriv),
    avg_slope_general(knots, coef, ord - 1),
    target_deriv
  )

  # start points of purely linear extrapolation
  yl <- extrapolate_left_transition(kmin - eps, knots, coef, target_deriv, eps)
  yr <- extrapolate_right_transition(kmax + eps, knots, coef, target_deriv, eps)

  left_linear <- yl - target_deriv * (kmin - eps - x)

  left_transition <- extrapolate_left_transition(
    x,
    knots,
    coef,
    target_deriv,
    ord = ord,
    eps = eps
  )

  right_transition <- extrapolate_right_transition(
    x,
    knots,
    coef,
    target_deriv,
    ord = ord,
    eps = eps
  )

  right_linear <- yr + target_deriv * (x - (kmax + eps))

  if (eps > 0) {
    value <- dplyr::case_when(
      x < (kmin - eps) ~ left_linear,
      (kmin - eps) <= x & x < kmin ~ left_transition,
      kmin <= x & x <= kmax ~ basis_dot(x, knots, coef, ord = ord),
      kmax < x & x <= (kmax + eps) ~ right_transition,
      (kmax + eps) < x ~ right_linear
    )
  } else {
    value <- dplyr::case_when(
      x < kmin ~ target_deriv * x,
      kmin <= x & x <= kmax ~ basis_dot(x, knots, coef, ord = ord),
      kmax < x ~ target_deriv * x
    )
  }

  value
}


basis_dot_extrap_deriv <- function(
  x,
  knots,
  coef,
  target_deriv = NULL,
  eps = 3 * mean(diff(knots)),
  ord = 4
) {
  knots <- sort(knots)
  avg_slope <- avg_slope_general(knots, coef)
  kmin <- knots[ord]
  kmax <- sort(knots, decreasing = TRUE)[ord]

  target_deriv <- ifelse(
    is.null(target_deriv),
    avg_slope_general(knots, coef, ord - 1),
    target_deriv
  )

  # start points of purely linear extrapolation
  yl <- extrapolate_left_transition(kmin - eps, knots, coef, target_deriv, eps)
  yr <- extrapolate_right_transition(kmax + eps, knots, coef, target_deriv, eps)

  left_linear <- target_deriv

  left_transition <- extrapolate_left_transition_deriv(
    x,
    knots,
    coef,
    target_deriv,
    ord = ord,
    eps = eps
  )

  right_transition <- extrapolate_right_transition_deriv(
    x,
    knots,
    coef,
    target_deriv,
    ord = ord,
    eps = eps
  )

  right_linear <- target_deriv

  value <- dplyr::case_when(
    x < (kmin - eps) ~ left_linear,
    (kmin - eps) <= x & x < kmin ~ left_transition,
    kmin <= x & x <= kmax ~ basis_dot_deriv(x, knots, coef, ord = ord),
    kmax < x & x <= (kmax + eps) ~ right_transition,
    (kmax + eps) < x ~ right_linear
  )

  value
}

find_alpha <- function(knots, coef) {
  a <- knots[4]
  B <- splines::splineDesign(knots, a)
  fx_at_a <- B %*% coef

  a - fx_at_a
}


# --------------------------------------------------------------------------- #
# PTM density
# --------------------------------------------------------------------------- #

dptm <- function(x, delta, knots, lambda, log = FALSE, m = 0, s = 1) {
  x <- s * x + m

  coef_tmp <- mi_coef(delta)
  sval <- avg_slope_general(knots, coef_tmp)
  coef_tmp <- coef_tmp / sval
  alpha <- find_alpha(knots, coef_tmp) |> drop()
  coef <- coef_tmp + alpha

  fx <- basis_dot_extrap(x, knots, coef, eps = lambda) |> drop()
  fx_deriv <- basis_dot_extrap_deriv(x, knots, coef, eps = lambda) |> drop()

  log_pdf <- dnorm(fx, log = TRUE) +
    log(pmax(fx_deriv, .Machine$double.eps)) +
    log(s)
  if (log) {
    return(log_pdf)
  }

  exp(log_pdf) |> drop()
}


# --------------------------------------------------------------------------- #
# Mean and variance of PTM density
# --------------------------------------------------------------------------- #

mean_ptm <- function(delta, knots, lambda, m = 0, s = 1, grid_x = NULL) {
  if (is.null(grid_x)) {
    grid_x <- seq(-10, 10, length.out = 2001)
  }
  dx <- diff(grid_x)[1] # uniform grid spacing assumed
  dvals <- dptm(grid_x, delta, knots, lambda, m = m, s = s)
  sum(grid_x * dvals) * dx
}


var_ptm <- function(delta, knots, lambda, m = 0, s = 1, grid_x = NULL) {
  if (is.null(grid_x)) {
    grid_x <- seq(-10, 10, length.out = 2001)
  }
  dx <- diff(grid_x)[1] # uniform grid spacing assumed
  dvals <- dptm(grid_x, delta, knots, lambda, m = m, s = s)
  m <- sum(grid_x * dvals) * dx
  sum((grid_x - m)^2 * dvals) * dx
}

# --------------------------------------------------------------------------- #
# Standardized PTM density
# --------------------------------------------------------------------------- #

std_dptm <- function(
  x,
  delta,
  knots,
  lambda,
  niter = 10,
  tol = 0.01,
  grid_x = NULL
) {
  m <- 0
  s <- 1
  m_diff <- Inf
  s_diff <- Inf
  dptm_ <- function(...) dptm(..., m = m, s = s)

  if (is.null(grid_x)) {
    grid_x <- seq(-10, 10, length.out = 2001)
  }

  # center
  i <- 0
  while ((m_diff > tol) & i < niter) {
    m_new <- mean_ptm(delta, knots, lambda, m = m, s = s, grid_x = grid_x)
    m_diff <- abs(m_new)
    m <- m_new + m

    i <- i + 1
  }

  # scale
  j <- 0
  while ((s_diff > tol) & j < niter) {
    s_new <- sqrt(var_ptm(delta, knots, lambda, m = m, s = s, grid_x = grid_x))
    s_diff <- abs(1 - s_new)
    s <- s_new * s

    j <- j + 1
  }

  no_convergence <- (m_diff > tol) | (s_diff > tol)
  if (no_convergence) {
    message("No convergence.")
    message("Mean iterations: ", i, " Mean diff: ", m_diff)
    message("Scale iterations: ", j, " Scale diff: ", s_diff)
  }

  pdf = dptm(x, delta, knots, lambda, m = m, s = s)
  list(
    pdf = pdf,
    m = m,
    s = s,
    m_std = m_new,
    s_std = s_new,
    iter_m = i,
    iter_s = j,
    converged = !no_convergence
  )
}
