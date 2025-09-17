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
    default = "application-fh/010-qgam"
  ),
  make_option(
    "--model",
    type = "character",
    default = "default"
  )
)

# Parse arguments
opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

params <- read_csv(fs::path(opt$jobdir, "params.csv"))[opt$jobrow, ]


out_path <- fs::path(opt$jobdir, "out")
img_path <- fs::path(opt$jobdir, "img")
data_path <- fs::path(opt$jobdir, "..", "..", "data")

out_path_dist <- fs::path(out_path, "dist")

fs::dir_create(out_path)
fs::dir_create(img_path)
fs::dir_create(out_path_dist)

model <- params$model

data <- read_csv(fs::path(data_path, "framingham.csv"))

data$newid <- factor(data$newid)

cholst_mean <- data$cholst |> mean()
cholst_sd <- data$cholst |> sd()
data$age_at_start <- data$age
data$age <- data$age_at_start + data$year
age_mean <- data$age |> mean()
age_sd <- data$age |> sd()

data <- data |>
  mutate(cholst = (cholst - cholst_mean) / cholst_sd) |>
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

probs <- seq(0.005, 0.995, length.out = 25)
# probs <- c(0.5, 0.6)

if (str_detect(model, "ri")) {
  form <- list(
    cholst ~ s(age, k = 20, bs = "ps") + sex + s(newid, bs = "re"),
    ~ s(age, k = 7, bs = "ps") + sex
  )
} else {
  form <- list(
    cholst ~ s(age, k = 20, bs = "ps") + sex,
    ~ s(age, k = 7, bs = "ps") + sex
  )
}


tic()
fit <- mqgam(
  form = form,
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
crps <- scoringutils::quantile_score(test$cholst, pred, probs, weigh = TRUE) |>
  mean()

# ..............................................................................
# ---- Quantile score ----
# ..............................................................................

probs <- c(0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)

fit <- mqgam(
  form = form,
  data = train,
  qu = probs
)

fit_viz <- mgcViz::getViz(fit)

pred <- sapply(
  fit_viz,
  predict,
  newdata = test
)

quantile_score_array <- 2 *
  ((pinLoss(test$cholst, pred, probs, add = FALSE)))

quantile_score_mean <- colMeans(quantile_score_array)
quantile_score_df <- tibble(
  quantile_score_mean,
  prob = names(quantile_score_mean)
)


# ..............................................................................
# ---- Plot ----
# ..............................................................................

age_seq <- with(data, seq(min(age), max(age), length.out = 150))
sex_seq <- rep(c(0, 1), each = length(age_seq))
age_seq <- c(age_seq, age_seq)
pred <- sapply(
  fit_viz,
  predict,
  newdata = data.frame(age = age_seq, sex = sex_seq, newid = test$newid[1])
)

predictions <- pred |>
  as_tibble() |>
  mutate(i = row_number()) |>
  mutate(age = age_seq) |>
  mutate(sex = sex_seq) |>
  pivot_longer(
    -c(i, age, sex),
    names_to = "alpha",
    values_to = "predicted_quantile"
  )


p <- data |>
  ggplot() +
  geom_point(
    aes(age, cholst),
    alpha = 0.05
  ) +
  geom_line(
    data = predictions |> mutate(alpha_num = as.numeric(alpha)),
    aes(age, predicted_quantile, group = alpha, color = alpha_num)
  ) +
  facet_wrap(~sex) +
  # scale_color_gradient2(midpoint = 0.5, low = "blue", mid = "yellow", high = "red") +
  # scale_color_brewer("RdYlBu") +
  NULL

ggsave(fs::path(
  img_path,
  paste0("quantiles-", "fold", params$fold, ".png")
))

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
