library(tidyverse)

data_seed <- 1:100
data_type <- c("gaussian", "mixture", "skewnorm", "ptm")
ntrain <- c(250, 500)
mcmc_strategy <- c("iwls-nuts")
apply_jitter <- c(FALSE)
igprior <- c("1,0.005", "0.01,0.01")

params <- expand_grid(
  data_seed,
  data_type,
  ntrain,
  mcmc_strategy,
  apply_jitter,
  igprior
)

# Get the directory of the currently active file
current_file <- rstudioapi::getActiveDocumentContext()$path
current_dir <- dirname(current_file)

write_csv(params, fs::path(current_dir, "params.csv"))
