library(tidyverse)
library(fs)

data_dir <- path(path_wd(), "application-dbbmi", "analysis", "data")
dist <- read_csv(path(data_dir, "dist.csv")) |>
  filter(!str_detect(job, "001c")) |>
  filter(!str_detect(job, "001d")) |>
  identity()

dist |> distinct(job)

# ..............................................................................
# ---- PTM Variants ----
# ..............................................................................

dist_ptm <- dist |>
  filter(str_detect(model, "ptm")) |>
  mutate(model = "PTM") |>
  group_by(model, a, b, trafo_target_slope) |>
  summarise(log_score = mean(log_score, na.rm = TRUE), crps = mean(crps)) |>
  ungroup() |>
  arrange(log_score) |>
  mutate(
    trafo_target_slope = ifelse(
      trafo_target_slope == "identity",
      "$\\lambda = 0.1(b-a)$",
      "$\\lambda \\to \\infty$"
    )
  ) |>
  identity()

dist_tab <- dist_ptm |>
  mutate(log_score = round(log_score, 2)) |>
  mutate(crps = round(crps, 3)) |>
  kbl(
    booktabs = TRUE,
    linesep = c(""),
    col.names = c(
      "Model",
      "a",
      "b",
      "$\\lambda$",
      "Log Score $\\downarrow$",
      "CRPS $\\downarrow$"
    ),
    escape = FALSE,
    format = "latex"
  ) |>
  kable_styling(full_width = TRUE) |>
  column_spec(1, width = "3em") |>
  column_spec(2, width = "3em") |>
  column_spec(3, width = "3em") |>
  identity()

cat(dist_tab, file = path(out_dir, "db_dist_ptm.tex"))


dist |>
  group_by(model, a, b, trafo_target_slope) |>
  summarise(log_score = mean(log_score, na.rm = TRUE), crps = mean(crps)) |>
  ungroup() |>
  arrange(log_score)
