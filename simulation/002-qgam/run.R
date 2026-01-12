#!/usr/bin/env Rscript

# Load required libraries
suppressPackageStartupMessages({
  library(optparse)
  library(tidyverse)
  library(fs)
  library(scoringutils)
  library(tictoc) # for timing

  library(qgam)
  library(mgcViz)
})

# --------------------------------------------------------------------------- #
# Data import
# --------------------------------------------------------------------------- #

load_data <- function(data_seed, data_type, train_or_test) {
  data_filename <- paste0(data_type, "-", sprintf("%03d", data_seed), ".csv")
  data_filepath <- fs::path(data_path, data_type, train_or_test, data_filename)
  read_csv(data_filepath, show_col_types = FALSE)
}

run_one <- function(
  data_seed,
  data_type,
  ntrain,
  ntest,
  out_path_dist,
  out_path_covariates
) {
  model <- "qgam"

  train <- load_data(data_seed, data_type, "train")
  test <- load_data(data_seed, data_type, "test")

  train <- train[1:ntrain, ]
  test <- test[1:ntest, ]

  # --------------------------------------------------------------------------- #
  # model
  # --------------------------------------------------------------------------- #

  probs <- seq(0.005, 0.995, length.out = 25)
  # probs <- c(0.5, 0.6)

  tic()
  fit <- mqgam(
    form = list(
      y ~
        s(x0, k = 20, bs = "ps") +
        s(x1, k = 20, bs = "ps") +
        s(x2, k = 20, bs = "ps") +
        s(x3, k = 20, bs = "ps"),
      ~ s(x0, k = 7, bs = "ps") +
        s(x1, k = 7, bs = "ps") +
        s(x2, k = 7, bs = "ps") +
        s(x3, k = 7, bs = "ps")
    ),
    data = train,
    qu = probs
  )
  timing <- toc(quiet = TRUE)

  fit_viz <- mgcViz::getViz(fit)

  summary(fit)
  plot(mgcViz::getViz(fit)[[1]]) # visualize quantiles

  # ..............................................................................
  # ---- CRPS approximation ----
  # ..............................................................................

  pred <- sapply(fit_viz, predict, newdata = test)
  crps <- scoringutils::quantile_score(test$y, pred, probs, weigh = TRUE) |>
    mean()

  # ..............................................................................
  # ---- Summary of distribution analysis ----
  # ..............................................................................

  dist_summary <- tibble(
    crps
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

  write_csv(dist_summary, fp_dist)

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
  make_option("--jobdir", type = "character", help = "Job Directory")
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
  out_path_dist = out_path_dist,
  out_path_covariates = out_path_covariates
)
