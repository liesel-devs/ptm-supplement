#!/usr/bin/env Rscript

# Load required libraries
suppressPackageStartupMessages({
  library(optparse)
  library(tidyverse)
  library(fs)
  library(scoringutils)
  library(tictoc) # for timing

  library(DDPstar)
})

# --------------------------------------------------------------------------- #
# Data import
# --------------------------------------------------------------------------- #

load_data <- function(data_seed, data_type, train_or_test) {
  data_filename <- paste0(data_type, "-", sprintf("%03d", data_seed), ".csv")
  data_filepath <- fs::path(
    data_path,
    data_type,
    train_or_test,
    data_filename
  )
  read_csv(data_filepath, show_col_types = FALSE)
}

run_one <- function(
  data_seed,
  data_type,
  ntrain,
  ntest,
  nsave,
  nburn,
  nskip,
  out_path_dist,
  out_path_covariates
) {
  model <- "ddpstar"
  print("Starting run")

  train <- load_data(data_seed, data_type, "train")
  test <- load_data(data_seed, data_type, "test")

  train <- train[1:ntrain, ]
  test <- test[1:ntest, ]

  # --------------------------------------------------------------------------- #
  # DDPstar model
  # --------------------------------------------------------------------------- #
  mcmc <- mcmccontrol(nsave = nsave, nburn = nburn, nskip = nskip)

  prior <- priorcontrol(
    # hyperparameters for stick-breaking process
    aalpha = 2,
    balpha = 2,

    # hyperparameters for sigma_l^2
    a = 2,
    b = NA, # auto-initialization

    L = 20 # number of mixture components
  )

  print("Starting model fit")
  tic()
  m <- DDPstar(
    formula = y ~
      f(x0, nseg = 20) +
      f(x1, nseg = 20) +
      f(x2, nseg = 20) +
      f(x3, nseg = 20),
    data = train,
    standardise = FALSE,
    mcmc = mcmc,
    prior = prior,
    compute.WAIC = TRUE
  )
  timing <- toc(quiet = TRUE)
  print("Finished model fit")

  # ..............................................................................
  # ---- WAIC on training data ----
  # ..............................................................................

  waic <- m$WAIC$WAIC

  # ..............................................................................
  # ---- KLD and Log Score on test data ----
  # ..............................................................................

  predict_density_on_test <- function(object, newdata) {
    m <- object
    test <- newdata
    ntest <- nrow(test)
    prediction <- matrix(NA_real_, nrow = ntest, ncol = m$mcmc$nsave)

    # I compute predictions individually here to avoid inefficiencies
    # I want only the specific predictions for each row of the test data frame
    # but when using den.grid, predict.DDPStar would evaluate the full grid
    # for each row of the test dataframe, thereby greatly inflating the
    # number of evaluations.

    for (i in 1:ntest) {
      pred <- predict(
        m,
        what = "denfun",
        newdata = test[i, ],
        den.grid = test$y[i]
      )
      prediction[i, ] <- pred$denfun
    }

    prediction |> t()
  }

  print("predicting density on test")
  pdf_test <- predict_density_on_test(m, test) # (nsamples, ntest)

  pdf_summary <- pdf_test |>
    t() |>
    log() |>
    as.data.frame() |>
    as_tibble() |>
    mutate(n = row_number()) |>
    pivot_longer(
      starts_with("V"),
      names_to = "draw",
      values_to = "log_pdf_sample",
      names_prefix = "V"
    ) |>
    group_by(n) |>
    summarise(
      log_pdf_predict = matrixStats::logSumExp(log_pdf_sample) - log(n()),
    ) |>
    mutate(log_pdf_true = test$log_pdf) |>
    ungroup() |>
    summarise(
      kld = mean(log_pdf_true - log_pdf_predict),
      log_score = sum(-log_pdf_predict)
    )

  kld <- pdf_summary$kld
  log_score <- pdf_summary$log_score

  # ..............................................................................
  # ---- Conditional response CDF on test data ----
  # ..............................................................................

  print("predicting CDF on test")
  predict_cdf_on_test <- function(object, newdata) {
    m <- object
    test <- newdata
    ntest <- nrow(test)
    prediction <- matrix(NA_real_, nrow = ntest, ncol = m$mcmc$nsave)

    for (i in 1:ntest) {
      pred <- predict(
        m,
        what = "probfun",
        newdata = test[i, ],
        q.value = test$y[i]
      )
      prediction[i, ] <- pred$probfun
    }

    prediction |> t()
  }

  cdf_test <- predict_cdf_on_test(m, test) # (nsamples, ntest)

  cdf_mad <- cdf_test |>
    t() |>
    as.data.frame() |>
    as_tibble() |>
    mutate(n = row_number()) |>
    mutate(cdf_true = test$cdf) |>
    pivot_longer(
      starts_with("V"),
      names_to = "draw",
      values_to = "cdf_sample",
      names_prefix = "V"
    ) |>
    summarise(mad_cdf = mean(abs(cdf_true - cdf_sample))) |>
    pull(mad_cdf)

  cdf_calibration <- cdf_test |>
    t() |>
    as.data.frame() |>
    as_tibble() |>
    mutate(n = row_number()) |>
    pivot_longer(
      starts_with("V"),
      names_to = "draw",
      values_to = "cdf_sample",
      names_prefix = "V"
    ) |>
    group_by(n) |>
    summarise(
      q05 = quantile(cdf_sample, 0.05),
      q95 = quantile(cdf_sample, 0.95)
    ) |>
    mutate(cdf_true = test$cdf) |>
    mutate(in_ci = q05 <= cdf_true & cdf_true <= q95) |>
    summarise(coverage = mean(in_ci), width = mean(q95 - q05)) |>
    identity()

  # ..............................................................................
  # ---- CRPS ----
  # ..............................................................................
  print("predicting quantiles on test")

  probs <- seq(0.005, 0.995, length.out = 25)

  pred <- predict(
    m,
    what = "quantfun",
    newdata = test,
    quant.probs = probs
  )

  q <- aperm(pred$quantfun, c(2, 3, 1))

  ntest <- nrow(test)
  nMCMC <- dim(q)[1]
  quantile_scores <- matrix(NA, nrow = nMCMC, ncol = ntest)
  for (j in 1:nMCMC) {
    quantile_scores[j, ] <- scoringutils::quantile_score(
      observed = test$y,
      predicted = q[j, , ] |> t(),
      quantile_level = probs,
      weigh = TRUE
    )
  }

  crps <- quantile_scores |> mean()

  # ..............................................................................
  # ---- Summary of distribution analysis ----
  # ..............................................................................

  dist_summary <- tibble(
    waic,
    kld,
    log_score,
    cdf_mad,
    crps
  ) |>
    mutate(cdf_ci_coverage = cdf_calibration$coverage) |>
    mutate(cdf_ci_width = cdf_calibration$width)

  # ..............................................................................
  # ---- Mean function on test data ----
  # ..............................................................................

  print("predicting mean functions on test")
  pred_meanfun <- function(object, test, xnum) {
    m <- object

    x_name <- paste0("x", xnum)
    x <- test[[x_name]]
    fx <- test[[paste0("f", xnum, "_", "loc")]]
    # fx_mean <- test[[paste0("f", xnum, "_", "loc", "_", "mean")]]
    fx_mean <- mean(fx)
    fx_centered <- fx - fx_mean

    newdata <- data.frame(
      x0 = rep(0, times = nrow(test)),
      x1 = 0,
      x2 = 0,
      x3 = 0
    )
    newdata[x_name] <- test[[x_name]]

    pred <- predict(
      m,
      what = "regfun",
      reg.select = 1:4,
      newdata = newdata
    )

    # center in each iteration
    fpred_centered <- apply(pred$regfun, 2, function(x) x - mean(x))

    fx_df <- fpred_centered |>
      as.data.frame() |>
      as_tibble() |>
      mutate(
        n = row_number(),
        x = x,
        fx = fx_centered
      ) |>

      pivot_longer(
        starts_with("V"),
        names_to = "draw",
        values_to = "value",
        names_prefix = "V"
      ) |>
      mutate(xnum = xnum)

    fx_df
  }

  meanfuns_df <- pred_meanfun(m, test, 0) |>
    bind_rows(pred_meanfun(m, test, 1)) |>
    bind_rows(pred_meanfun(m, test, 2)) |>
    bind_rows(pred_meanfun(m, test, 3))

  # MSE
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
  # ---- Scale function on test data ----
  # ..............................................................................

  print("predicting scale functions on test")
  pred_scalefun <- function(object, test, xnum) {
    m <- object

    x_name <- paste0("x", xnum)
    x <- test[[x_name]]
    fx <- test[[paste0("f", xnum, "_", "scale")]]
    # fx_mean <- test[[paste0("f", xnum, "_", "scale", "_", "mean")]]
    fx_mean <- mean(fx)
    fx_centered <- fx - fx_mean

    newdata <- data.frame(
      x0 = rep(0, times = nrow(test)),
      x1 = 0,
      x2 = 0,
      x3 = 0
    )
    newdata[x_name] <- test[[x_name]]

    pred <- predict(
      m,
      what = "varfun",
      reg.select = 1:4,
      newdata = newdata
    )

    pred$varfun <- pred$varfun |> sqrt() |> log()

    # center in each iteration
    fpred_centered <- apply(pred$varfun, 2, function(x) x - mean(x))

    fx_df <- fpred_centered |>
      as.data.frame() |>
      as_tibble() |>
      mutate(
        n = row_number(),
        x = x,
        fx = fx_centered
      ) |>

      pivot_longer(
        starts_with("V"),
        names_to = "draw",
        values_to = "value",
        names_prefix = "V"
      ) |>
      mutate(xnum = xnum)

    fx_df
  }

  scalefuns_df <- pred_scalefun(m, test, 0) |>
    bind_rows(pred_scalefun(m, test, 1)) |>
    bind_rows(pred_scalefun(m, test, 2)) |>
    bind_rows(pred_scalefun(m, test, 3))

  # MSE
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
      run = tid,
      job = job,
      time = timing$toc - timing$tic
    )

  covariates_summary <- covariates_summary |>
    mutate(
      data_type = data_type,
      data_seed = data_seed,
      model = model,
      ntrain = ntrain,
      ntest = ntest,
      run = tid,
      job = job,
      time = timing$toc - timing$tic
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
  print("Writing")
  print(fp_dist)
  write_csv(dist_summary, fp_dist)
  print("Writing")
  print(fp_covariates)
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
    metavar = "integer",
    default = 1
  ),
  make_option(
    "--data_type",
    type = "character",
    help = "Data type",
    metavar = "character",
    default = "mixture"
  ),
  make_option(
    "--ntrain",
    type = "integer",
    help = "Number of training observations",
    metavar = "integer",
    default = 1000
  ),

  # default parameters
  make_option(
    "--jobid",
    type = "character",
    default = "test",
    help = "Job ID [default %default]"
  ),
  make_option(
    "--jobdir",
    type = "character",
    help = "Job Directory",
    default = "sim2/001-ddpstar"
  ),
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
data_path <- fs::path(opt$jobdir, "..", "simulation", "data")

out_path_dist <- fs::path(out_path, "dist")
out_path_covariates <- fs::path(out_path, "covariates")

fs::dir_create(out_path)
fs::dir_create(out_path_dist)
fs::dir_create(out_path_covariates)

run_one(
  data_seed = opt$data_seed,
  data_type = opt$data_type,
  ntrain = opt$ntrain,
  ntest = 1000,
  nsave = 15000,
  nskip = 10,
  nburn = 5000,
  out_path_dist = out_path_dist,
  out_path_covariates = out_path_covariates
)

finished_path <- fs::path(opt$jobdir, "finished")
fs::dir_create(finished_path)
finished_file <- fs::path(finished_path, opt$jobrow)
fs::file_create(finished_file)
