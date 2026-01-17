#!/usr/bin/env Rscript

# Load required libraries
suppressPackageStartupMessages({
  library(optparse)
  library(tidyverse)
  library(fs)
  library(scoringRules)
  library(tictoc) # for timing

  library(skewsamp)
  library(logger)

  library(GpGp) # for Gaussian processes
})


# --------------------------------------------------------------------------- #
# Data import
# --------------------------------------------------------------------------- #

load_data <- function(data_seed, data_type, train_or_test) {
  data_filename <- paste0(data_type, "-", sprintf("%03d", data_seed), ".csv")
  data_filepath <- fs::path(data_path, data_type, train_or_test, data_filename)
  read_csv(data_filepath, show_col_types = FALSE)
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
    default = 500
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
    default = "scaling/004-kowal"
  )
)

# Parse arguments
opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)


out_path <- fs::path(opt$jobdir, "out")
data_path <- fs::path(opt$jobdir, "..", "..", "data", "sim")

out_path_dist <- fs::path(out_path, "dist")
out_path_covariates <- fs::path(out_path, "covariates")
out_path_log <- fs::path(out_path, "log")

fs::dir_create(out_path)
fs::dir_create(out_path_dist)
fs::dir_create(out_path_covariates)
fs::dir_create(out_path_log)

# Source files:
source(fs::path(opt$jobdir, "kowal/source_sba.R"))
source(fs::path(opt$jobdir, "kowal/helper_funs.R"))
source(fs::path(opt$jobdir, "kowal/slice.R"))
source(fs::path(opt$jobdir, "kowal/run.R"))

run_one(
  data_seed = opt$data_seed,
  data_type = opt$data_type,
  ntrain = opt$ntrain,
  ntest = 5000,
  nsave = 2000,
  out_path_dist = out_path_dist,
  out_path_log = out_path_log
)

# seeds <- 1:100
# data_types <- c("gaussian", "mixture", "skewnorm", "unif", "u-shaped", "ptm")
# ntrains <- c(250, 500, 1000)

# # seeds <- 1:2
# # data_types <- c("gaussian", "mixture")
# # ntrains <- c(250)

# conditions <- expand_grid(seeds, data_types, ntrains) |>
#   mutate(i = row_number())

# plan(multisession, workers = 10)

# res <- furrr::future_map(
#   1:nrow(conditions),
#   function(i) {
#     run_one(
#       data_seed = conditions$seeds[i],
#       data_type = conditions$data_types[i],
#       ntrain = conditions$ntrains[i],
#       ntest = 5000,
#       nsave = 15000,
#       nskip = 15,
#       nburn = 5000
#     )
#   },
#   .options = furrr_options(seed = 2404)
# )
