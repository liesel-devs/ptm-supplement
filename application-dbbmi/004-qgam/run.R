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
    default = "application-dbbmi/004-qgam"
  ),
  make_option(
    "--testing",
    type = "logical",
    default = TRUE
  )
)

# Parse arguments
opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

params <- read_csv(fs::path(opt$jobdir, "params.csv"))[opt$jobrow, ]


out_path <- fs::path(opt$jobdir, "out")
data_path <- fs::path(opt$jobdir, "..", "..", "data")

out_path_dist <- fs::path(out_path, "dist")

fs::dir_create(out_path)
fs::dir_create(out_path_dist)

model <- "qgam"

data <- read_csv(fs::path(data_path, "dbbmi.csv"))

bmi_mean <- data$bmi |> mean()
bmi_sd <- data$bmi |> sd()
age_mean <- data$age |> mean()
age_sd <- data$age |> sd()

data <- data |>
  mutate(bmi = (bmi - bmi_mean) / bmi_sd) |>
  mutate(age = (age - age_mean) / age_sd)

train <- data |>
  filter(fold != params$fold)

test <- data |>
  filter(fold == params$fold)

ALL_DATA = params$fold == -1
if (ALL_DATA) {
  train <- data
  test <- data[1:2, ]
}

# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #

if (opt$testing) {
  probs <- seq(0.005, 0.995, length.out = 2)
  train <- train[1:200, ]
} else {
  probs <- seq(0.005, 0.995, length.out = 25)
}
# probs <- c(0.5, 0.6)

tic()
fit <- mqgam(
  form = list(
    bmi ~ s(age, k = 20, bs = "ps"),
    ~ s(age, k = 7, bs = "ps")
  ),
  data = train,
  qu = probs
)
timing <- toc(quiet = TRUE)

fit_viz <- mgcViz::getViz(fit)

# summary(fit)
# plot(mgcViz::getViz(fit)[[10]]) # visualize quantiles

# ..............................................................................
# ---- CRPS approximation ----
# ..............................................................................

pred <- sapply(fit_viz, predict, newdata = test)
crps <- scoringutils::quantile_score(test$bmi, pred, probs, weigh = TRUE) |>
  mean()

# ..............................................................................
# ---- Quantile score ----
# ..............................................................................

probs <- c(0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)

fit <- mqgam(
  form = list(bmi ~ s(age, k = 20, bs = "ps"), ~ s(age, k = 7, bs = "ps")),
  data = train,
  qu = probs
)

fit_viz <- mgcViz::getViz(fit)

pred <- sapply(fit_viz, predict, newdata = data.frame(age = test$age))

quantile_score_array <- 2 *
  ((pinLoss(test$bmi, pred, probs, add = FALSE)))

quantile_score_mean <- colMeans(quantile_score_array)
quantile_score_df <- tibble(
  quantile_score_mean,
  prob = names(quantile_score_mean)
)


# ..............................................................................
# ---- Plot ----
# ..............................................................................

age_seq <- with(data, seq(min(age), max(age), length.out = 150))
pred <- sapply(fit_viz, predict, newdata = data.frame(age = age_seq))

predictions <- pred |>
  as_tibble() |>
  mutate(i = row_number()) |>
  mutate(age = age_seq) |>
  pivot_longer(-c(i, age), names_to = "alpha", values_to = "predicted_quantile")


p <- data |>
  ggplot() +
  geom_point(
    aes(age, bmi),
    alpha = 0.05
  ) +
  geom_line(
    data = predictions |> mutate(alpha_num = as.numeric(alpha)),
    aes(age, predicted_quantile, group = alpha, color = alpha_num)
  ) +
  NULL


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

# Add fit time
dist_summary <- dist_summary %>%
  mutate(fit_seconds = timing$toc - timing$tic)

summaries <- list(
  dist = dist_summary,
  quantile_score = quantile_score_df
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
