#!/usr/bin/env Rscript

# Load required libraries
suppressPackageStartupMessages({
  library(optparse)
  library(tidyverse)
  library(fs)
  library(scoringutils)
  library(tictoc) # for timing

  library(tram)
  library(gamlss)
  library(rsample)
})

# ..............................................................................
# ---- Function definitions ----
# ..............................................................................

# The following functions are copied from the code accompanying the
# TAMLS paper:

# Siegfried, S., Kook, L., & Hothorn, T. (2023).
# Distribution-free location-scale regression.
# The American Statistician, 77(4), 345–356.
# https://doi.org/10.1080/00031305.2023.2203177

## ----TAMLS-model--------------------------------------------------------------

`coef<-` <- mlt::`coef<-` ## masked

## extract expressions for gamlss.family
e <- expression(
  d <- data.frame(y = y, m = mu, s = sigma),
  mf <- mlt(._mff, data = d, fixed = c("m" = 1, "scl_s" = 1), scale = TRUE), # theta = ._start
  if ((i / 5) %% 1 == 0) {
    print(logLik(mf))
  },
  i <<- i + 1,
  # print(logLik(mf)),
  trm <- predict(mf, newdata = d, type = "trafo"),
  tr <- predict(
    mf,
    newdata = data.frame(y = y, s = sigma, m = 0),
    type = "trafo"
  )
)


# ..............................................................................
# ---- Model fitting function ----
# ..............................................................................

fit_tamls <- function(data, basis_order) {
  train <- data

  ## gamlss.dist for transformation models (scale_shift = FALSE)
  TM <- function(mu.link = "identity", sigma.link = "identity") {
    mstats <- checklink("mu.link", "TM", substitute(mu.link), c("identity"))
    dstats <- checklink(
      "sigma.link",
      "TM",
      substitute(sigma.link),
      c("identity")
    )
    structure(
      list(
        family = c("TM", "trafo"),
        parameters = list(mu = TRUE, sigma = TRUE),
        nopar = 2,
        type = "Continuous",
        mu.link = as.character(substitute(mu.link)),
        sigma.link = as.character(substitute(sigma.link)),
        mu.linkfun = mstats$linkfun,
        sigma.linkfun = dstats$linkfun,
        mu.linkinv = mstats$linkinv,
        sigma.linkinv = dstats$linkinv,
        mu.dr = mstats$mu.eta,
        sigma.dr = dstats$mu.eta,
        dldm = function(y, mu, sigma) {
          eval(e)
          return(trm)
        },
        d2ldm2 = function(y, mu, sigma) -1,
        dldd = function(y, mu, sigma) {
          eval(e)
          return(-trm * 1 / 2 * tr + 1 / 2)
        },
        d2ldd2 = function(y, mu, sigma) {
          eval(e)
          return(1 / 2 * (1 / 2 * tr * mu - tr^2))
        },
        d2ldmdd = function(y, mu, sigma) {
          eval(e)
          return(1 / 2 * tr)
        },
        G.dev.incr = function(y, mu, sigma) {
          eval(e)
          return(-2 * mf$logliki(coef(as.mlt(mf)), weights = weights(mf)))
        },
        rqres = expression(NA),
        mu.initial = expression({
          mu <- rep(0, length(y))
        }),
        sigma.initial = expression({
          sigma <- rep(0, length(y))
        }),
        mu.valid = function(mu) TRUE,
        sigma.valid = function(sigma) TRUE,
        y.valid = function(y) TRUE,
        mean = function(mu, sigma) return(NA),
        variance = function(mu, sigma) return(NA)
      ),
      class = c("gamlss.family", "family")
    )
  }

  ## get coefficients basis functions
  refit <- function(model, data) {
    mu <- predict(model, what = "mu")
    sigma <- predict(model, what = "sigma")
    y <- data$y
    eval(e)
    return(mf)
  }
  # ..............................................................................
  # ---- model setup ----
  # ..............................................................................

  # adds dummy columns to train data frame
  train <- train |> mutate(m = 0, s = exp(0))

  ## ----TAMLS-model-setup--------------------------------------------------------
  ## support & thetas
  OR <- 10
  log_first <- FALSE

  mf <<- BoxCox(
    y ~ 1,
    data = train,
    order = OR,
    log_first = log_first
  ) ## thetas

  ._mff <<- BoxCox(
    y ~ m | s,
    data = train,
    model_only = TRUE,
    order = OR,
    log_first = log_first
  ) ## support

  ._start <<- coef(as.mlt(mf))
  mlt(._mff, data = train)

  # ..............................................................................
  # ---- Model fit ----
  # ..............................................................................

  i <<- 0
  tic()
  mTM <- gamlss(
    formula = y ~
      1 +
      pb(x0, inter = 17) + # P-Splines with 20 parameters
      pb(x1, inter = 17) +
      pb(x2, inter = 17) +
      pb(x3, inter = 17),
    sigma.fo = ~ 0 +
      pb(x0, inter = 17) +
      pb(x1, inter = 17) +
      pb(x2, inter = 17) +
      pb(x3, inter = 17),
    data = train,
    family = TM(),
    control = gamlss.control(n.cyc = 400, c.crit = 0.001)
  )
  mlt_TM <- refit(mTM, train) ## fitted mlt model
  timing <- toc(quiet = TRUE)

  return(list(mlt_model = mlt_TM, gamlss_model = mTM, timing = timing))
}

# --------------------------------------------------------------------------- #
# Data import
# --------------------------------------------------------------------------- #

load_data <- function(data_seed, data_type, train_or_test) {
  data_filename <- paste0(data_type, "-", sprintf("%03d", data_seed), ".csv")
  data_filepath <- fs::path(data_path, data_type, train_or_test, data_filename)
  read_csv(data_filepath, show_col_types = FALSE)
}

# data_seed <- 1
# data_type <- "mixture"
# train_or_test <- "train"
# ntest <- 102
# ntrain <- 301

run_one <- function(
  data_seed,
  data_type,
  ntrain,
  ntest,
  nbootstrap,
  out_path_dist,
  out_path_covariates
) {
  model <- "tamls"
  train <- load_data(data_seed, data_type, "train")
  test <- load_data(data_seed, data_type, "test")

  train <- train[1:ntrain, ]
  test <- test[1:ntest, ]

  fit <- fit_tamls(train, basis_order = 10)

  mTM <- fit$gamlss_model
  mlt_TM <- fit$mlt_model
  timing <- fit$timing

  mTM$call$data <- train

  # ..............................................................................
  # ---- KLD and log score on test data ----
  # ..............................................................................

  # browser()
  m_predicted <- predict(mTM, what = "mu", newdata = test)
  s_predicted <- predict(mTM, what = "sigma", newdata = test)

  logpdf_predicted <- predict(
    mlt_TM,
    newdata = data.frame(y = test$y, m = m_predicted, s = s_predicted),
    type = "logdensity"
  )

  log_score <- sum(-logpdf_predicted)
  kld <- mean(test$log_pdf - logpdf_predicted)

  # ..............................................................................
  # ---- MAD on test data ----
  # ..............................................................................

  cdf_predicted <- predict(
    mlt_TM,
    newdata = data.frame(y = test$y, m = m_predicted, s = s_predicted),
    type = "distribution"
  )

  cdf_mad <- mean(abs(test$cdf - cdf_predicted))

  # ..............................................................................
  # ---- Bootstrapped uncertainty quantification for CDF ----
  # ..............................................................................

  tic()
  bootstrap_uncertainty <- function(data) {
    fit <- fit_tamls(data, basis_order = 10)
    mTM <- fit$gamlss_model
    mlt_TM <- fit$mlt_model

    mTM$call$data <- data

    m_predicted <- predict(mTM, what = "mu", newdata = test)
    s_predicted <- predict(mTM, what = "sigma", newdata = test)

    cdf_predicted <- predict(
      mlt_TM,
      newdata = data.frame(y = test$y, m = m_predicted, s = s_predicted),
      type = "distribution"
    )

    m_predicted <- predict(mTM, what = "mu", newdata = test, type = "terms")
    s_predicted <- predict(mTM, what = "sigma", newdata = test, type = "terms")

    i <- 1:ncol(m_predicted)

    m_predicted <- map_dfc(i, function(i) {
      (m_predicted[, i] - mean(m_predicted[, i])) / sd(m_predicted[, i])
    })

    s_predicted <- map_dfc(i, function(i) {
      (s_predicted[, i] - mean(s_predicted[, i])) / sd(s_predicted[, i])
    })

    names(m_predicted) <- paste0("m", 0:3)
    names(s_predicted) <- paste0("s", 0:3)

    bind_cols(m_predicted, s_predicted, cdf = cdf_predicted)
  }

  train_bootstraps <- rsample::bootstraps(train, times = nbootstrap)

  bootstrap_df <- map_dfr(1:nrow(train_bootstraps), function(i) {
    x <- train_bootstraps$splits[[i]]
    df <- bootstrap_uncertainty(as.data.frame(x)) |>
      mutate(bootstrap_i = i)

    df
  })

  cdfs_long <- bootstrap_df |>
    select(cdf, bootstrap_i) |>
    group_by(bootstrap_i) |>
    mutate(obs = row_number()) |>
    ungroup() |>
    group_by(obs) |>
    mutate(
      ci_low = quantile(cdf, 0.05),
      ci_high = quantile(cdf, 0.95)
    ) |>
    mutate(obs = as.integer(obs)) |>
    ungroup()

  cdf_calibration <- test |>
    select(cdf) |>
    rename(cdf_true = cdf) |>
    mutate(obs = row_number()) |>
    right_join(cdfs_long, by = "obs") |>
    mutate(
      in_ci = ci_low <= cdf_true & cdf_true <= ci_high,
    ) |>
    summarise(coverage = mean(in_ci), width = mean(ci_high - ci_low))

  timing_bootstrap <- toc(quiet = TRUE)

  # ..............................................................................
  # ---- CRPS ----
  # ..............................................................................

  predict_quantiles_on_test <- function(object, newdata, quant.probs) {
    m <- object
    test <- newdata
    ntest <- nrow(test)
    prediction <- matrix(NA_real_, nrow = length(quant.probs), ncol = ntest)

    for (i in 1:ntest) {
      pred <- predict(
        m,
        type = "quantile",
        newdata = test[i, ],
        prob = quant.probs
      )
      prediction[, i] <- pred |> as.numeric()
    }

    prediction
  }

  quantile_predicted <- predict_quantiles_on_test(
    mlt_TM,
    data.frame(y = test$y, m = m_predicted, s = s_predicted),
    quant.probs = test$cdf
  )

  probs <- seq(0.005, 0.995, length.out = 25)

  q <- predict_quantiles_on_test(
    mlt_TM,
    data.frame(y = test$y, m = m_predicted, s = s_predicted),
    probs
  ) # (nprobs, ntest)

  quantile_scores <- scoringutils::quantile_score(
    observed = test$y,
    predicted = q |> t(),
    quantile_level = probs,
    weigh = TRUE
  )

  crps <- quantile_scores |> mean()

  # ..............................................................................
  # ---- Summary of distribution analysis ----
  # ..............................................................................

  dist_summary <- tibble(
    kld,
    log_score,
    cdf_mad,
    crps
  ) |>
    mutate(cdf_ci_coverage = cdf_calibration$coverage) |>
    mutate(cdf_ci_width = cdf_calibration$width)

  # ..............................................................................
  # ---- Location functions ----
  # ..............................................................................

  pred_meanfun <- function(test, xnum) {
    x_name <- paste0("x", xnum)
    x <- test[[x_name]]
    fx <- test[[paste0("f", xnum, "_", "loc")]]
    fx_centered <- fx - mean(fx)
    fx_centered <- fx_centered / sd(fx_centered)

    fx_true_df <- tibble(fx_true = fx_centered) |>
      mutate(n = row_number())

    fx_df <- bootstrap_df |>
      select(paste0("m", xnum), bootstrap_i) |>
      group_by(bootstrap_i) |>
      mutate(n = row_number()) |>
      ungroup()

    fx_df <- fx_df |>
      left_join(fx_true_df, by = "n") |>
      rename(draw = bootstrap_i)

    fx_df["value"] <- fx_df[paste0("m", xnum)]

    fx_df <- fx_df |>
      select(-paste0("m", xnum)) |>
      mutate(xnum = xnum) |>
      rename(fx = fx_true)

    fx_df
  }

  meanfuns_df <- pred_meanfun(test, 0) |>
    bind_rows(pred_meanfun(test, 1)) |>
    bind_rows(pred_meanfun(test, 2)) |>
    bind_rows(pred_meanfun(test, 3))

  meanfuns_mse <- meanfuns_df |>
    group_by(xnum, n) |>
    mutate(
      bias = mean(value - fx),
      var = var(value)
    ) |>
    ungroup() |>
    group_by(xnum) |>
    summarise(
      bias = mean(bias),
      var = mean(var),
      mse = mean((fx - value)^2)
    )

  # Calibration
  meanfuns_calibration <- meanfuns_df |>
    group_by(n, xnum, fx) |>
    summarise(q05 = quantile(value, 0.05), q95 = quantile(value, 0.95)) |>
    mutate(in_ci = q05 <= fx & fx <= q95) |>
    ungroup() |>
    group_by(xnum) |>
    summarise(coverage = mean(in_ci), width = mean(q95 - q05)) |>
    identity()

  meanfuns_summary <- meanfuns_mse |>
    left_join(meanfuns_calibration, by = "xnum") |>
    mutate(parameter = "loc")

  # ..............................................................................
  # ---- Scale functions ----
  # ..............................................................................

  pred_scalefun <- function(test, xnum) {
    x_name <- paste0("x", xnum)
    x <- test[[x_name]]
    fx <- test[[paste0("f", xnum, "_", "loc")]]
    fx_centered <- fx - mean(fx)
    fx_centered <- fx_centered / sd(fx_centered)

    fx_true_df <- tibble(fx_true = fx_centered) |>
      mutate(n = row_number())

    fx_df <- bootstrap_df |>
      select(paste0("s", xnum), bootstrap_i) |>
      group_by(bootstrap_i) |>
      mutate(n = row_number()) |>
      ungroup()

    fx_df <- fx_df |>
      left_join(fx_true_df, by = "n") |>
      rename(draw = bootstrap_i)

    fx_df["value"] <- fx_df[paste0("s", xnum)]

    fx_df <- fx_df |>
      select(-paste0("s", xnum)) |>
      mutate(xnum = xnum) |>
      rename(fx = fx_true)

    fx_df
  }

  scalefuns_df <- pred_scalefun(test, 0) |>
    bind_rows(pred_scalefun(test, 1)) |>
    bind_rows(pred_scalefun(test, 2)) |>
    bind_rows(pred_scalefun(test, 3))

  scalefuns_mse <- scalefuns_df |>
    group_by(xnum, n) |>
    mutate(
      bias = mean(value - fx),
      var = var(value)
    ) |>
    ungroup() |>
    group_by(xnum) |>
    summarise(
      bias = mean(bias),
      var = mean(var),
      mse = mean((fx - value)^2)
    )

  # Calibration
  scalefuns_calibration <- scalefuns_df |>
    group_by(n, xnum, fx) |>
    summarise(q05 = quantile(value, 0.05), q95 = quantile(value, 0.95)) |>
    mutate(in_ci = q05 <= fx & fx <= q95) |>
    ungroup() |>
    group_by(xnum) |>
    summarise(coverage = mean(in_ci), width = mean(q95 - q05)) |>
    identity()

  scalefuns_summary <- scalefuns_mse |>
    left_join(scalefuns_calibration, by = "xnum") |>
    mutate(parameter = "scale")

  # ..............................................................................
  # ---- Summary of covariates analysis ----
  # ..............................................................................

  covariates_summary <- bind_rows(meanfuns_summary, scalefuns_summary) |>
    rename(ci_width = width) |>
    rename(ci_coverage = coverage)

  # ..............................................................................
  # ---- Save run information ----
  # ..............................................................................

  tid <- format(Sys.time(), "%Y%m%d-%H%M%S")
  job <- fs::path(out_path_dist, "..", "..") |>
    fs::path_real() |>
    fs::path_file()

  dist_summary <- dist_summary |>
    mutate(
      data_type = data_type,
      data_seed = data_seed,
      model = model,
      ntrain = ntrain,
      ntest = ntest,
      fit_seconds = timing$toc - timing$tic,
      bootstrap_seconds = timing_bootstrap$toc - timing_bootstrap$tic,
      run = tid,
      job = job
    )

  covariates_summary <- covariates_summary |>
    mutate(
      data_type = data_type,
      data_seed = data_seed,
      model = model,
      ntrain = ntrain,
      ntest = ntest,
      run = tid,
      job = job
    )

  # ..............................................................................
  # ---- Write results to disk ----
  # ..............................................................................

  identifier <- paste0(
    model,
    "-",
    data_type,
    "-",
    sprintf("%03d", data_seed),
    "-",
    "n",
    ntrain,
    ".csv"
  )

  fp_dist <- fs::path(out_path_dist, paste0("dist-", identifier))
  fp_covariates <- fs::path(
    out_path_covariates,
    paste0("covariates-", identifier)
  )

  write_csv(dist_summary, fp_dist)
  write_csv(covariates_summary, fp_covariates)

  return(NULL)
}


# ..............................................................................
# ---- Run analysis ----
# ..............................................................................

# Define command line options
option_list <- list(
  # parameters from params.csv
  make_option(
    "--data_seed",
    type = "integer",
    help = "Data seed",
    metavar = "integer"
  ),
  make_option(
    "--data_type",
    type = "character",
    help = "Data type",
    metavar = "character"
  ),
  make_option(
    "--ntrain",
    type = "integer",
    help = "Number of training observations",
    metavar = "integer"
  ),

  # default parameters
  make_option(
    "--jobid",
    type = "character",
    default = "test",
    help = "Job ID [default %default]"
  ),
  make_option("--jobdir", type = "character", help = "Job Directory"),
  make_option(
    "--jobrow",
    type = "integer",
    help = "Job Index",
    default = 0
  )
)

# Parse arguments
opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)


out_path <- fs::path(opt$jobdir, "out")
data_path <- fs::path(opt$jobdir, "..", "data")

out_path_dist <- fs::path(out_path, "dist")
out_path_covariates <- fs::path(out_path, "covariates")

fs::dir_create(out_path)
fs::dir_create(out_path_dist)
fs::dir_create(out_path_covariates)

run_one(
  data_seed = opt$data_seed,
  data_type = opt$data_type,
  ntrain = opt$ntrain,
  ntest = 5000,
  nbootstrap = 100,
  out_path_dist = out_path_dist,
  out_path_covariates = out_path_covariates
)

finished_path <- fs::path(opt$jobdir, "finished")
fs::dir_create(finished_path)
finished_file <- fs::path(finished_path, opt$jobrow)
fs::file_create(finished_file)
