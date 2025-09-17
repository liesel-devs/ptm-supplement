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


# Define command line options
option_list <- list(
  # parameters from params.csv
  make_option(
    "--fold",
    type = "integer",
    metavar = "integer",
    default = 1
  ),
  make_option(
    "--jobrow",
    type = "character",
    default = 1,
    help = "Job ID [default %default]"
  ),
  make_option(
    "--jobdir",
    type = "character",
    help = "Job Directory",
    default = "application-dbbmi/007-kowal"
  )
)

# Parse arguments
opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

params <- read_csv(fs::path(opt$jobdir, "params.csv"))[opt$jobrow, ]


out_path <- fs::path(opt$jobdir, "out")
data_path <- fs::path(opt$jobdir, "..", "..", "data")

out_path_dist <- fs::path(out_path, "dist")
out_path_log <- fs::path(out_path, "log")

fs::dir_create(out_path)
fs::dir_create(out_path_dist)
fs::dir_create(out_path_log)

model <- "kowal"
data <- read_csv(fs::path(data_path, "dbbmi.csv"))

bmi_mean <- data$bmi |> mean()
bmi_sd <- data$bmi |> sd()
age_mean <- data$age |> mean()
age_sd <- data$age |> sd()

data <- data |>
  mutate(bmi = (bmi - bmi_mean) / bmi_sd) |>
  mutate(age = (age - age_mean) / age_sd) |>
  mutate(y = bmi)

train <- data |>
  filter(fold != params$fold)

test <- data |>
  filter(fold == params$fold)

ALL_DATA = params$fold == -1
if (ALL_DATA) {
  train <- data
  test <- data[1:2, ]
}


# Source files:
source(fs::path(opt$jobdir, "kowal/source_sba.R"))
source(fs::path(opt$jobdir, "kowal/helper_funs.R"))
source(fs::path(opt$jobdir, "kowal/slice.R"))


identifier <- paste0(
  model,
  "-",
  "fold",
  params$fold,
  ".log"
)


logfile <- fs::path(out_path_log, identifier)
log_appender(appender_file(logfile))
log_info("Run started.")

log_info("Starting to fit model.")
tic()
m <- sbgp(
  y = train$bmi,
  locs = train$age,
  X = cbind(1, train[, "age"]),
  nsave = 5000,
  approx_g = FALSE,
  locs_test = test$age,
  X_test = cbind(1, test[, "age"])
)
timing <- toc(quiet = TRUE)
log_info("Model fit complete.")

# ..............................................................................
# ---- CRPS on test data ----
# ..............................................................................
log_info("Starting CRPS computation.")
crps <- mean(crps_sample(test$bmi, t(m$post_ypred)))
log_info("CRPS computation finished.")


# ..............................................................................
# ---- Log Score on test data ----
# ..............................................................................

log_score <- scoringRules::logs_sample(test$bmi, t(m$post_ypred)) |> sum()

# ..............................................................................
# ---- Quantile curves plot ----
# ..............................................................................

age_seq <- seq(min(data$age), max(data$age), length.out = 150)

log_info("Starting to fit model.")
tic()
m <- sbgp(
  y = train$bmi,
  locs = train$age,
  X = cbind(1, train[, "age"]),
  nsave = 5000,
  approx_g = FALSE,
  locs_test = age_seq,
  X_test = cbind(1, age_seq)
)
timing <- toc(quiet = TRUE)
log_info("Model fit complete.")

probs <- c(0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)
pred <- apply(t(m$post_ypred), 1, quantile, probs = probs)

predictions <- pred |>
  t() |>
  as_tibble() |>
  mutate(i = row_number()) |>
  mutate(age = age_seq) |>
  pivot_longer(
    -c(i, age),
    names_to = "alpha",
    values_to = "predicted_quantile"
  ) |>
  mutate(alpha = str_remove(alpha, "%")) |>
  mutate(alpha_num = as.numeric(alpha) * 0.01)


p <- data |>
  ggplot() +
  geom_point(
    aes(age, bmi),
    alpha = 0.05
  ) +
  geom_line(
    data = predictions,
    aes(age, predicted_quantile, group = alpha, color = alpha_num),
  ) +
  geom_smooth(
    data = predictions,
    aes(age, predicted_quantile, group = alpha, color = alpha_num),
    se = FALSE
  ) +
  NULL
p


# ..............................................................................
# ---- Summary of distribution analysis ----
# ..............................................................................

dist_summary <- tibble(
  crps,
  log_score
)

# ..............................................................................
# ---- Save run information ----
# ..............................................................................

tid <- format(Sys.time(), "%Y%m%d-%H%M%S")
job <- fs::path(out_path_dist, "..", "..") |>
  fs::path_real() |>
  fs::path_file()

# Add fit time
dist_summary <- dist_summary %>%
  mutate(fit_seconds = timing$toc - timing$tic)

summaries <- list(
  dist = dist_summary
)

if (ALL_DATA) {
  summaries[["predictions"]] = predictions
}

# Add common columns to each tibble
summaries <- map(
  summaries,
  ~ .x |>
    mutate(
      model = model,
      job = job,
      run = tid,
      fold = params$fold
    )
)

for (i in seq_along(summaries)) {
  name_ <- names(summaries)[i]

  identifier <- paste0(
    name_,
    "-",
    model,
    "-",
    "fold",
    params$fold,
    "-",
    "row",
    opt$jobrow,
    ".csv"
  )

  fs::dir_create(fs::path(out_path, name_))
  write_csv(summaries[[i]], fs::path(out_path, name_, identifier))
}

finished_path <- fs::path(opt$jobdir, "finished")
fs::dir_create(finished_path)
finished_file <- fs::path(finished_path, opt$jobrow)
fs::file_create(finished_file)
