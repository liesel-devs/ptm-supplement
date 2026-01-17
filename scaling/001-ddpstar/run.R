#!/usr/bin/env Rscript

# Load required libraries
suppressPackageStartupMessages({
  library(optparse)
  library(tidyverse)
  library(fs)
  library(scoringutils)
  library(tictoc) # for timing

  library(DDPstar)
  library(posterior) # for effective sample sizes
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

  # data_seed <- 1
  # data_type <- "mixture"
  # ntrain <- 103
  # ntest <- 101
  # nsave <- 102
  # nburn <- 105
  # nskip <- 1

  train <- load_data(data_seed, data_type, "train")

  while (nrow(train) < ntrain) {
    data_seed <- data_seed + 1
    train <- bind_rows(train, load_data(data_seed, data_type, "train"))
  }

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

  ess_beta <- array(m$fit$beta, c(1, dim(m$fit$beta))) |>
    aperm(c(2, 1, 3, 4)) |>
    apply(c(3, 4), posterior::ess_bulk)

  min_ess_beta <- min(ess_beta)

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
  # ---- Summary of distribution analysis ----
  # ..............................................................................

  dist_summary <- tibble(
    waic,
    kld,
    log_score,
  )

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
      time = timing$toc - timing$tic,
      min_ess_beta = min_ess_beta,
      min_ess_per_s = min_ess_beta / time
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
  print("Writing")
  print(fp_dist)
  write_csv(dist_summary, fp_dist)
  print("Writing")

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
    default = "scaling/001-ddpstar"
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
data_path <- fs::path(opt$jobdir, "..", "..", "data", "sim")

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
  nsave = 2000 * 4, # other Bayesian models use 4 chains, saving 8000 posterior samples
  nskip = 10,
  nburn = 1000,
  out_path_dist = out_path_dist,
  out_path_covariates = out_path_covariates
)

finished_path <- fs::path(opt$jobdir, "finished")
fs::dir_create(finished_path)
finished_file <- fs::path(finished_path, opt$jobrow)
fs::file_create(finished_file)
