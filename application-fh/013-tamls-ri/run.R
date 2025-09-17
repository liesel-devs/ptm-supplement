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
    default = "application-fh/014-tamls-ri"
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

  if (str_detect(params$model, "ri")) {
    form_mu <- y ~
      1 +
        pb(age, inter = 17) +
        sex +
        re(random = ~ 1 | newid)
    form_sigma <- ~ 0 +
      pb(age, inter = 17) +
      sex
  } else {
    form_mu <- y ~
      1 +
        pb(age, inter = 17) +
        sex
    form_sigma <- ~ 0 +
      pb(age, inter = 17) +
      sex
  }

  i <<- 0
  tic()
  mTM <- gamlss(
    formula = form_mu,
    sigma.fo = form_sigma,
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

m_predicted <- predict(mTM, what = "mu", newdata = test, type = "terms")
m_predicted[, 3] <- 0 # set random intercept column to zero
m_predicted <- rowSums(m_predicted) + mTM$mu.coefficients[1]

# first element: between-cluster stddev
# second element: within-cluster stddev
re_stddev_components <- as.numeric(nlme::VarCorr(mTM$mu.coefSmo[[2]])[,
  "StdDev"
])

s_predicted <- predict(mTM, what = "sigma", newdata = test)

logpdf_predicted <- predict(
  mlt_TM,
  newdata = data.frame(y = test$cholst, m = m_predicted, s = s_predicted),
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


# first element: between-cluster stddev
# second element: within-cluster stddev
re_stddev_components <- as.numeric(nlme::VarCorr(mTM$mu.coefSmo[[2]])[,
  "StdDev"
])

nsamples <- 1000
ngrid <- nrow(test)
pred_samples <- matrix(NA, nrow = ngrid, ncol = nsamples)

for (i in 1:nsamples) {
  for (j in 1:ngrid) {
    mj <- as.numeric(m_predicted[j] + rnorm(1, sd = re_stddev_components[1]))
    sj <- s_predicted[j]
    uij <- runif(1)

    predij <- predict_quantiles_on_test(
      mlt_TM,
      data.frame(m = mj, s = sj),
      uij
    ) |>
      as.numeric()
    pred_samples[j, i] <- predij
  }
}

crps <- scoringRules::crps_sample(
  y = test$cholst,
  dat = pred_samples
) |>
  mean()

# ..............................................................................
# ---- Plot ----
# ..............................................................................

age_seq <- with(data, seq(min(age), max(age), length.out = 100))
sex_seq <- rep(c(0, 1), each = length(age_seq))
age_seq <- c(age_seq, age_seq)
newdata <- data.frame(age = age_seq, sex = sex_seq, newid = test$newid[1])

m_predicted <- predict(mTM, what = "mu", newdata = newdata, type = "terms")
m_predicted[, 3] <- 0 # set random intercept column to zero
m_predicted <- rowSums(m_predicted) + mTM$mu.coefficients[1]

s_predicted <- predict(mTM, what = "sigma", newdata = newdata)

nsamples <- 1000
ngrid <- nrow(newdata)
pred_samples <- matrix(NA, nrow = ngrid, ncol = nsamples)

for (i in 1:nsamples) {
  for (j in 1:ngrid) {
    mj <- as.numeric(m_predicted[j] + rnorm(1, sd = re_stddev_components[1]))
    sj <- s_predicted[j]
    uij <- runif(1)

    predij <- predict_quantiles_on_test(
      mlt_TM,
      data.frame(m = mj, s = sj),
      uij
    ) |>
      as.numeric()
    pred_samples[j, i] <- predij
  }
}

probs <- c(0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)
pred <- apply(pred_samples, 1, quantile, probs = probs)

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
  )


p <- data |>
  ggplot() +
  geom_point(
    aes(age, cholst),
    alpha = 0.05
  ) +
  # geom_line(
  #   data = predictions |> mutate(alpha_num = as.numeric(alpha)),
  #   aes(age, predicted_quantile, group = alpha, color = alpha_num)
  # ) +
  geom_smooth(
    data = predictions |> mutate(alpha_num = as.numeric(alpha)),
    aes(age, predicted_quantile, group = alpha, color = alpha_num),
    se = FALSE
  ) +
  facet_wrap(~sex) +
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
