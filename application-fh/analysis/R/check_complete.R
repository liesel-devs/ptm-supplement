library(tidyverse)
library(fs)

data_dir <- path(path_wd(), "application-fh", "analysis", "data")

# ..............................................................................
# ---- Dist data ----
# ..............................................................................

dist <- read_csv(path(data_dir, "dist.csv"))

dist |>
  group_by(job) |>
  summarise(n = n())

dist |>
  group_by(model) |>
  summarise(n = n())

loc <- read_csv(path(data_dir, "loc_summary.csv"))

loc |>
  group_by(job) |>
  summarise(n = n())

loc_samples <- read_csv(path(data_dir, "loc_samples_summary.csv"))

loc_samples |>
  group_by(job) |>
  summarise(n = n())

# expecting 11 for
# (check) ddpstar
# (check) qgam
# (check) tamls
# (check) kowal
# bctm

# gaussian: 1 less than expected (full-data batch)

# ecpecting 66 for ptm
# got 60: 6 less than expected (full-data batch is missing)

# ..............................................................................
# ---- Predictions ----
# ..............................................................................

pred <- read_csv(path(data_dir, "predictions.csv"))

pred |>
  group_by(model, fold) |>
  summarise(n = n())

pred <- read_csv(path(data_dir, "quantile_curves.csv"))

pred |>
  group_by(model, fold) |>
  summarise(n = n())


pred <- read_csv(path(data_dir, "quantiles.csv"))

pred |>
  group_by(model) |>
  summarise(n = n())

# missing:
# ptm
# gaussian
# bctm
