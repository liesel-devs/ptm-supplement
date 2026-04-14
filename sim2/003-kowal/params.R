library(tidyverse)

data_seed <- 1:100
data_type <- c("gaussian", "mixture", "skewnorm", "ptm")
ntrain <- c(250, 500, 1000)

# data_seed <- 1:2
# data_type <- c("gaussian", "mixture")
# ntrain <- c(250)

params <- expand_grid(data_seed, data_type, ntrain)

# Get the directory of the currently active file
current_file <- rstudioapi::getActiveDocumentContext()$path
current_dir <- dirname(current_file)

write_csv(params, fs::path(current_dir, "params.csv"))
