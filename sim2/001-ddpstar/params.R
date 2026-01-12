library(tidyverse)

data_seed <- 1:100
data_type <- c("ptm")
ntrain <- c(250, 500, 1000)

params <- expand_grid(data_seed, data_type, ntrain)

current_file <- rstudioapi::getActiveDocumentContext()$path
current_dir <- dirname(current_file)

write_csv(params, fs::path(current_dir, "params.csv"))
