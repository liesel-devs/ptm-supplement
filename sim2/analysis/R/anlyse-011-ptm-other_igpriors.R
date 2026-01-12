library(tidyverse)

current_file <- rstudioapi::getActiveDocumentContext()$path
current_dir <- dirname(current_file)
setwd(current_dir)


# ..............................................................................
# ---- Covariates ----
# ..............................................................................

df <- read_csv(fs::path("../data/011-ptm-other_igpriors/covariates.csv"))


df <- df |>
  mutate(tau2_prior = str_glue("{ig_a}, {ig_b}"))


df |>
  group_by(tau2_prior, data_type, ntrain, xnum, data_seed) |>
  summarise(n = n())


df |>
  distinct(tau2_prior, data_type, ntrain, xnum, data_seed, .keep_all = TRUE) |>
  group_by(tau2_prior, data_type, ntrain) |>
  summarise(n = n())


cov_summary <- df |>
  distinct(
    tau2_prior,
    parameter,
    data_type,
    ntrain,
    xnum,
    data_seed,
    .keep_all = TRUE
  ) |>
  group_by(tau2_prior, parameter) |>
  summarise(
    mse = mean(mse),
    ci_coverage = mean(ci_coverage),
    ci_width = mean(ci_width)
  )

cov_summary

# ..............................................................................
# ---- Dist ----
# ..............................................................................

df <- read_csv(fs::path("../data/011-ptm-other_igpriors/dist.csv"))


df <- df |>
  mutate(tau2_prior = str_glue("{ig_a}, {ig_b}"))


dist_summary <- df |>
  group_by(tau2_prior) |>
  summarise(
    waic = mean(waic),
    kld = mean(kld),
    log_score = mean(log_score),
    crps = mean(crps),
    cdf_mad = mean(cdf_mad),
    cdf_ci_coverage = mean(cdf_ci_coverage),
    cdf_ci_width = mean(cdf_ci_width),
  )

dist_summary
