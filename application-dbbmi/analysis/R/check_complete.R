library(tidyverse)
library(fs)

data_dir <- path(path_wd(), "application-dbbmi", "analysis", "data")

# ..............................................................................
# ---- Dist data ----
# ..............................................................................

dist <- read_csv(path(data_dir, "dist.csv"))

dist |>
  group_by(model) |>
  summarise(n = n())

# ..............................................................................
# ---- Predictions ----
# ..............................................................................

pred <- read_csv(path(data_dir, "predictions.csv"))

pred |>
  group_by(model, fold) |>
  summarise(n = n())
