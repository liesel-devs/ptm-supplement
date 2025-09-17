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
    default = "application-fh/012-ddpstar"
  ),
  make_option(
    "--model",
    type = "character",
    default = "default"
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

if (str_detect(model, "ri")) {
  form <- cholst ~ f(age, nseg = 20) + sex + rae(newid)
} else {
  form <- cholst ~ f(age, nseg = 20) + sex
}

print("Starting model fit")
tic()
m <- DDPstar(
  formula = form,
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
      den.grid = test$cholst[i]
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
    observed = test$cholst,
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
      test$cholst,
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
sex_seq <- rep(c(0, 1), each = length(age_seq))
age_seq <- c(age_seq, age_seq)

pred <- predict(
  m,
  what = "quantfun",
  newdata = data.frame(age = age_seq, sex = sex_seq),
  quant.probs = probs
)

pred <- pred$quantfun |> apply(c(1, 3), mean)

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
