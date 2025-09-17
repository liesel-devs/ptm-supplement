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
    default = "application-dbbmi/005-ddpstar"
  ),
  make_option(
    "--testing",
    type = "logical",
    default = TRUE
  )
)

NSAVE <- 15000
NBURN <- 5000
NSKIP <- 10


# Parse arguments
opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

params <- read_csv(fs::path(opt$jobdir, "params.csv"))[opt$jobrow, ]


out_path <- fs::path(opt$jobdir, "out")
data_path <- fs::path(opt$jobdir, "..", "..", "data")

out_path_dist <- fs::path(out_path, "dist")

fs::dir_create(out_path)
fs::dir_create(out_path_dist)

model <- "ddpstar"

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

if (opt$testing) {
  NSAVE <- 150
  NBURN <- 50
  NSKIP <- 1
  train <- train[1:200, ]
  test <- test[1:10, ]
}

# --------------------------------------------------------------------------- #
# DDPstar model
# --------------------------------------------------------------------------- #
mcmc <- mcmccontrol(nsave = NSAVE, nburn = NBURN, nskip = NSKIP)

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
  formula = bmi ~ f(age, nseg = 20),
  data = train,
  standardise = FALSE, # data was standardised before
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
# ---- Log Score on test data ----
# ..............................................................................

predict_density_on_test <- function(object, newdata) {
  m <- object
  test <- newdata
  ntest <- nrow(test)
  prediction <- matrix(NA_real_, nrow = ntest, ncol = m$mcmc$nsave)

  # I compute predictions individually here to avoid inefficiencoies
  # I want only the specific predictions for each row of the test data frame
  # but when using den.grid, predict.DDPStar would evaluate the full grid
  # for each row of the test dataframe, thereby greatly inflating the
  # number of evaluations.

  for (i in 1:ntest) {
    pred <- predict(
      m,
      what = "denfun",
      newdata = test[i, ],
      den.grid = test$bmi[i]
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
  ungroup() |>
  summarise(
    log_score = sum(-log_pdf_predict)
  )

log_score <- pdf_summary$log_score


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
    observed = test$bmi,
    predicted = q[j, , ] |> t(),
    quantile_level = probs,
    weigh = TRUE
  )
}

crps <- quantile_scores |> mean()

# ..............................................................................
# ---- Quantile score ----
# ..............................................................................

probs <- c(0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)
pred <- predict(
  m,
  what = "quantfun",
  newdata = test,
  quant.probs = probs
)

q <- aperm(pred$quantfun, c(2, 3, 1))

ntest <- nrow(test)
nMCMC <- dim(q)[1]
quantile_scores <- matrix(NA, nrow = nMCMC, ncol = length(probs))
for (j in 1:nMCMC) {
  quan_scores_this_sample <- 2 *
    (qgam::pinLoss(
      test$bmi,
      q[j, , ] |> t(),
      probs,
      add = FALSE
    ) |>
      colMeans())

  quantile_scores[j, ] <- quan_scores_this_sample
}

quantile_score_df <- quantile_scores |>
  t() |>
  as_tibble() |>
  add_column(prob = probs) |>
  pivot_longer(
    starts_with("V"),
    names_to = "sample",
    values_to = "quantile_score",
    names_prefix = "V"
  ) |>
  group_by(prob) |>
  summarise(
    quantile_score_mean = mean(quantile_score),
    quantile_score_sd = sd(quantile_score)
  )


# ..............................................................................
# ---- Plot ----
# ..............................................................................

age_seq <- with(data, seq(min(age), max(age), length.out = 150))

pred <- predict(
  m,
  what = "quantfun",
  newdata = data.frame(age = age_seq),
  quant.probs = probs
)

pred <- pred$quantfun |> apply(c(1, 3), mean)

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
  # scale_color_gradient2(midpoint = 0.5, low = "blue", mid = "yellow", high = "red") +
  # scale_color_brewer("RdYlBu") +
  NULL

# ..............................................................................
# ---- Summary of distribution analysis ----
# ..............................................................................

dist_summary <- tibble(
  crps,
  log_score,
  waic
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
