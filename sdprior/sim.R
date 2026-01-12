library(tidyverse)
library(furrr)
source("sdprior/functions.R")


# --------------------------------------------------------------------------- #
# Check
# --------------------------------------------------------------------------- #

nbases <- 30
a <- -4
b <- 4
knots <- create_knots(c(a, b), p = nbases)
tau <- 15
delta <- rdelta_rw(p = nbases, tau = tau, nsamples = 1) |> drop()
lambda <- diff(knots) |> mean()
x <- seq(-8, 8, length.out = 801)

std_dptm(x, delta, knots, lambda = lambda, niter = 10)

par(mfrow = c(1, 2))
plot(x, dptm(x, delta, knots, lambda = lambda), type = "l")
abline(v = mean_ptm(delta, knots, lambda))
plot(x, std_dptm(x, delta, knots, lambda = lambda, niter = 10)$pdf, type = "l")
lines(x, dnorm(x), col = "red")

# --------------------------------------------------------------------------- #
# Total variation distance
# --------------------------------------------------------------------------- #

# Fixed grid for integration
grid_x <- seq(-10, 10, length.out = 2001)
dx <- diff(grid_x)[1] # uniform grid spacing

tv_ptm <- function(delta, knots, lambda) {
  std_density <- std_dptm(
    grid_x,
    delta,
    knots,
    lambda,
    niter = 20,
    grid_x = grid_x
  )
  diff_vals <- abs(std_density$pdf - dnorm(grid_x))
  iae <- sum(diff_vals) * dx
  tv <- iae / 2
  list(
    tv = tv,
    m = std_density$m,
    s = std_density$s,
    m_std = std_density$m_std,
    s_std = std_density$s_std,
    iter_m = std_density$iter_m,
    iter_s = std_density$iter_s,
    converged = std_density$converged
  )
}


tv_ptm_unstd <- function(delta, knots, lambda) {
  std_density <- dptm(grid_x, delta, knots, lambda)
  diff_vals <- abs(std_density - dnorm(grid_x))
  iae <- sum(diff_vals) * dx
  tv <- iae / 2
  tv
}


# check
tv_ptm(delta, knots, lambda)
tv_ptm_unstd(delta, knots, lambda)

# --------------------------------------------------------------------------- #
# Simulation loop functions
# --------------------------------------------------------------------------- #

tv_delta <- function(nsamples_delta, tau2, knots, lambda) {
  map_dfr(1:nsamples_delta, function(i) {
    delta <- rdelta_rw(p = nbases, tau = sqrt(tau2), nsamples = 1) |> drop()
    tv_result <- tv_ptm(delta, knots, lambda)
    tv_result_unstd <- tv_ptm_unstd(delta, knots, lambda)

    df <- tibble(
      tv = tv_result$tv,
      tv_unstd = tv_result_unstd,
      lambda = lambda,
      idelta = i,
      nbases = nbases,
      mean = tv_result$m,
      var = (tv_result$s)^2,
      mean_std = tv_result$m_std,
      var_std = (tv_result$s_std)^2,
      iter_m = tv_result$iter_m,
      iter_s = tv_result$iter_s,
      converged = tv_result$converged,
    )

    df
  })
}

tv_tau2 <- function(nsamples_tau, nsamples_delta, knots, lambda, theta) {
  map_dfr(1:nsamples_tau, function(i) {
    tau2 <- rweibull(1, shape = 0.5, scale = theta)
    df <- tv_delta(nsamples_delta, tau2, knots, lambda)
    df$tau2 <- tau2

    df
  })
}

tv_lambda <- function(lambda, nsamples_tau, nsamples_delta, knots, theta) {
  map_dfr(lambda, function(lambda) {
    df <- tv_tau2(nsamples_tau, nsamples_delta, knots, lambda, theta)
    df$lambda <- lambda
    df
  })
}

tv_theta <- function(
  theta,
  lambda,
  nsamples_tau,
  nsamples_delta,
  knots,
  .options
) {
  furrr::future_map_dfr(
    theta,
    function(theta) {
      df <- tv_lambda(lambda, nsamples_tau, nsamples_delta, knots, theta)
      df$theta <- theta

      df
    },
    .options = .options
  )
}

# --------------------------------------------------------------------------- #
# Run simulation
# --------------------------------------------------------------------------- #
nbases <- 30
a <- -4
b <- 4
knots <- create_knots(c(a, b), p = nbases)

theta <- 2^seq(-10, 4)
lambda <- c(0, diff(knots) |> mean(), 1, Inf)
nsamples_tau <- 1
nsamples_delta <- 1

# set.seed(2404)
plan(multisession, workers = 10)

result <- tv_theta(
  theta,
  lambda,
  nsamples_tau,
  nsamples_delta,
  knots,
  .options = furrr_options(seed = 2404)
)

write_csv(result, "sdprior/data/sdprior_sim.csv")
