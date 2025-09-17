#!/usr/bin/env Rscript

# Load required libraries
suppressPackageStartupMessages({
  library(optparse)
  library(tidyverse)
  library(fs)
  library(scoringutils)
  library(tictoc) # for timing

  library(tram)
  library(gamlss)
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
    default = "application-dbbmi/006-tamls"
  ),
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

model <- "tamls"

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

## ----TAMLS-model--------------------------------------------------------------

`coef<-` <- mlt::`coef<-` ## masked

## extract expressions for gamlss.family
e <- expression(
  d <- data.frame(y = y, m = mu, s = sigma),
  mf <- mlt(._mff, data = d, fixed = c("m" = 1, "scl_s" = 1), scale = TRUE), # theta = ._start
  if ((i / 5) %% 1 == 0) {
    print(logLik(mf))
  },
  i <<- i + 1,
  # print(logLik(mf)),
  trm <- predict(mf, newdata = d, type = "trafo"),
  tr <- predict(
    mf,
    newdata = data.frame(y = y, s = sigma, m = 0),
    type = "trafo"
  )
)


# ..............................................................................
# ---- Model fitting function ----
# ..............................................................................

fit_tamls <- function(data, basis_order) {
  train <- data

  ## gamlss.dist for transformation models (scale_shift = FALSE)
  TM <- function(mu.link = "identity", sigma.link = "identity") {
    mstats <- checklink("mu.link", "TM", substitute(mu.link), c("identity"))
    dstats <- checklink(
      "sigma.link",
      "TM",
      substitute(sigma.link),
      c("identity")
    )
    structure(
      list(
        family = c("TM", "trafo"),
        parameters = list(mu = TRUE, sigma = TRUE),
        nopar = 2,
        type = "Continuous",
        mu.link = as.character(substitute(mu.link)),
        sigma.link = as.character(substitute(sigma.link)),
        mu.linkfun = mstats$linkfun,
        sigma.linkfun = dstats$linkfun,
        mu.linkinv = mstats$linkinv,
        sigma.linkinv = dstats$linkinv,
        mu.dr = mstats$mu.eta,
        sigma.dr = dstats$mu.eta,
        dldm = function(y, mu, sigma) {
          eval(e)
          return(trm)
        },
        d2ldm2 = function(y, mu, sigma) -1,
        dldd = function(y, mu, sigma) {
          eval(e)
          return(-trm * 1 / 2 * tr + 1 / 2)
        },
        d2ldd2 = function(y, mu, sigma) {
          eval(e)
          return(1 / 2 * (1 / 2 * tr * mu - tr^2))
        },
        d2ldmdd = function(y, mu, sigma) {
          eval(e)
          return(1 / 2 * tr)
        },
        G.dev.incr = function(y, mu, sigma) {
          eval(e)
          return(
            -2 * mf$logliki(coef(as.mlt(mf)), weights = weights(mf))
          )
        },
        rqres = expression(NA),
        mu.initial = expression({
          mu <- rep(0, length(y))
        }),
        sigma.initial = expression({
          sigma <- rep(0, length(y))
        }),
        mu.valid = function(mu) TRUE,
        sigma.valid = function(sigma) TRUE,
        y.valid = function(y) TRUE,
        mean = function(mu, sigma) return(NA),
        variance = function(mu, sigma) return(NA)
      ),
      class = c("gamlss.family", "family")
    )
  }

  ## get coefficients basis functions
  refit <- function(model, data) {
    mu <- predict(model, what = "mu")
    sigma <- predict(model, what = "sigma")
    y <- data$y
    eval(e)
    return(mf)
  }
  # ..............................................................................
  # ---- model setup ----
  # ..............................................................................

  # adds dummy columns to train data frame
  train <- train |> mutate(m = 0, s = exp(0))

  ## ----TAMLS-model-setup--------------------------------------------------------
  ## support & thetas
  OR <- 10
  log_first <- FALSE

  mf <<- BoxCox(
    y ~ 1,
    data = train,
    order = OR,
    log_first = log_first
  ) ## thetas

  ._mff <<- BoxCox(
    y ~ m | s,
    data = train,
    model_only = TRUE,
    order = OR,
    log_first = log_first
  ) ## support

  ._start <<- coef(as.mlt(mf))
  mlt(._mff, data = train)

  # ..............................................................................
  # ---- Model fit ----
  # ..............................................................................

  i <<- 0
  tic()
  mTM <- gamlss(
    formula = y ~
      1 +
        pb(age, inter = 17),
    sigma.fo = ~ 0 +
      pb(age, inter = 17),
    data = train,
    family = TM(),
    control = gamlss.control(n.cyc = 20, c.crit = 0.1)
  )
  mlt_TM <- refit(mTM, train) ## fitted mlt model
  timing <- toc(quiet = TRUE)

  return(list(mlt_model = mlt_TM, gamlss_model = mTM, timing = timing))
}

# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #

print("Starting model fit")
tic()
fit <- fit_tamls(train, basis_order = 10)
timing <- toc(quiet = TRUE)
print("Finished model fit")

mTM <- fit$gamlss_model
mlt_TM <- fit$mlt_model
timing <- fit$timing

mTM$call$data <- train

# ..............................................................................
# ---- KLD and log score on test data ----
# ..............................................................................

# browser()
m_predicted <- predict(mTM, what = "mu", newdata = test)
s_predicted <- predict(mTM, what = "sigma", newdata = test)

logpdf_predicted <- predict(
  mlt_TM,
  newdata = data.frame(y = test$bmi, m = m_predicted, s = s_predicted),
  type = "logdensity"
)

log_score <- sum(-logpdf_predicted)


# ..............................................................................
# ---- CRPS ----
# ..............................................................................

predict_quantiles_on_test <- function(object, newdata, quant.probs) {
  m <- object
  test <- newdata
  ntest <- nrow(test)
  prediction <- matrix(NA_real_, nrow = length(quant.probs), ncol = ntest)

  for (i in 1:ntest) {
    pred <- predict(
      m,
      type = "quantile",
      newdata = test[i, ],
      prob = quant.probs
    )
    prediction[, i] <- pred |> as.numeric()
  }

  prediction
}

probs <- seq(0.005, 0.995, length.out = 25)

q <- predict_quantiles_on_test(
  mlt_TM,
  data.frame(y = test$bmi, m = m_predicted, s = s_predicted),
  probs
) # (nprobs, ntest)

quantile_scores <- scoringutils::quantile_score(
  observed = test$bmi,
  predicted = q |> t(),
  quantile_level = probs,
  weigh = TRUE
)

crps <- quantile_scores |> mean()

# ..............................................................................
# ---- Quantile score ----
# ..............................................................................

probs <- c(0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)
q <- predict_quantiles_on_test(
  mlt_TM,
  data.frame(y = test$bmi, m = m_predicted, s = s_predicted),
  probs
) # (nprobs, ntest)

quantile_score_array <- 2 *
  qgam::pinLoss(
    test$bmi,
    q |> t(),
    probs,
    add = FALSE
  )

quantile_score_mean <- colMeans(quantile_score_array)
quantile_score_df <- tibble(
  quantile_score_mean,
  prob = names(quantile_score_mean)
)


# ..............................................................................
# ---- Plot ----
# ..............................................................................

age_seq <- with(data, seq(min(age), max(age), length.out = 150))
newdata <- data.frame(age = age_seq)
m_predicted <- predict(mTM, what = "mu", newdata = newdata)
s_predicted <- predict(mTM, what = "sigma", newdata = newdata)

pred <- predict(
  mlt_TM,
  type = "quantile",
  newdata = data.frame(m = m_predicted, s = s_predicted),
  prob = probs
)

predictions <- pred |>
  t() |>
  as_tibble() |>
  mutate(i = row_number()) |>
  mutate(age = age_seq) |>
  pivot_longer(
    -c(i, age),
    names_to = "alpha",
    values_to = "predicted_quantile"
  )


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
