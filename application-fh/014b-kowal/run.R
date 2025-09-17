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
    default = "application-fh/015b-kowal"
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
out_path_log <- fs::path(out_path, "log")

fs::dir_create(out_path)
fs::dir_create(img_path)
fs::dir_create(out_path_dist)
fs::dir_create(out_path_log)

model <- "kowal"
data <- read_csv(fs::path(data_path, "framingham.csv"))

cholst_mean <- data$cholst |> mean()
cholst_sd <- data$cholst |> sd()
data$age_at_start <- data$age
data$age <- data$age_at_start + data$year
age_mean <- data$age |> mean()
age_sd <- data$age |> sd()


data <- data |>
  mutate(cholst = (cholst - cholst_mean) / cholst_sd) |>
  mutate(age = (age - age_mean) / age_sd) |>
  mutate(y = cholst)

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

plot(train$age, train$cholst)

log_info("Starting to fit model.")
tic()
m <- sbgp(
  y = train$cholst,
  locs = train$age,
  X = cbind(1, train[, c("age", "sex")]),
  nsave = 5000,
  approx_g = TRUE,
  emp_bayes = FALSE, # necessary for successful fitting here
  locs_test = test$age,
  X_test = cbind(1, test[, c("age", "sex")])
)
timing <- toc(quiet = TRUE)
log_info("Model fit complete.")

# ..............................................................................
# ---- CRPS on test data ----
# ..............................................................................
log_info("Starting CRPS computation.")
crps <- mean(crps_sample(test$cholst, t(m$post_ypred)))
log_info("CRPS computation finished.")


# ..............................................................................
# ---- Log Score on test data ----
# ..............................................................................

log_score <- scoringRules::logs_sample(test$cholst, t(m$post_ypred)) |> sum()

# ..............................................................................
# ---- Quantile curves plot ----
# ..............................................................................

age_seq <- with(data, seq(min(age), max(age), length.out = 100))
sex_seq <- rep(c(0, 1), each = length(age_seq))
age_seq <- c(age_seq, age_seq)
newdata <- data.frame(age = age_seq, sex = sex_seq)

log_info("Starting to fit model.")
tic()
m <- sbgp(
  y = train$cholst,
  locs = train$age,
  X = cbind(1, train[, c("age", "sex")]),
  nsave = 5000,
  emp_bayes = FALSE,
  approx_g = TRUE,
  locs_test = age_seq,
  X_test = cbind(1, newdata)
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
  mutate(sex = sex_seq) |>
  pivot_longer(
    -c(i, age, sex),
    names_to = "alpha",
    values_to = "predicted_quantile"
  ) |>
  mutate(alpha = str_remove(alpha, "%")) |>
  mutate(alpha_num = as.numeric(alpha) * 0.01)


p <- data |>
  ggplot() +
  geom_point(
    aes(age, cholst),
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
  facet_wrap(~sex) +
  NULL
p

ggsave(fs::path(
  img_path,
  paste0("quantiles-", "fold", params$fold, ".png")
))


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
