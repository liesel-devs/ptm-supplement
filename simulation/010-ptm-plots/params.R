library(tidyverse)

data_seed <- 1:20
data_type <- c("ptm")
ntrain <- c(250, 500, 1000)
mcmc_strategy <- c("iwls-nuts")
apply_jitter <- c(FALSE)


params <- expand_grid(data_seed, data_type, ntrain, mcmc_strategy, apply_jitter)

data_seed <- 1:2
data_type <- c("gaussian", "mixture", "skewnorm")
ntrain <- c(250, 500, 1000)
params2 <- expand_grid(
  data_seed,
  data_type,
  ntrain,
  mcmc_strategy,
  apply_jitter
)

params <- bind_rows(params, params2)

# Get the directory of the currently active file
current_file <- rstudioapi::getActiveDocumentContext()$path
current_dir <- dirname(current_file)

write_csv(params, fs::path(current_dir, "params.csv"))
