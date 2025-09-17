library(tidyverse)
library(fs)
library(kableExtra)

data_dir <- path(path_wd(), "application-fh", "analysis", "data")
out_dir <- path(path_wd(), "application-fh", "analysis", "out")
dir_create(out_dir)

diagnostics <- read_csv(path(data_dir, "diagnostics.csv")) |>
  filter(!str_detect(model, "ptm") | str_detect(model, "77-id")) |>
  mutate(
    model_label = case_when(
      model == "gaussian-linear" ~ "Gaussian (lin) +RI",
      model == "gaussian-linear-nori" ~ "Gaussian (lin)",
      model == "gaussian-nonlinear-nori" ~ "Gaussian (nonlin)",
      model == "gaussian-nonlinear" ~ "Gaussian (nonlin) +RI",
      model == "ptm-linear-77-id" ~ "PTM (lin) +RI",
      model == "ptm-linear-nori-77-id" ~ "PTM (lin)",
      model == "ptm-nonlinear-nori-77-id" ~ "PTM (nonlin)",
      model == "ptm-nonlinear-77-id" ~ "PTM (nonlin) +RI"
    )
  ) |>
  mutate(
    model_label = factor(
      model_label,
      levels = c(
        "PTM (lin)",
        "PTM (lin) +RI",
        "PTM (nonlin)",
        "PTM (nonlin) +RI",
        "Gaussian (lin)",
        "Gaussian (lin) +RI",
        "Gaussian (nonlin)",
        "Gaussian (nonlin) +RI"
      )
    )
  )

diagnostics_summary <- diagnostics |>
  group_by(variable, model_label) |>
  summarise(
    ess_bulk_min = mean(ess_bulk_min),
    ess_tail_min = mean(ess_tail_min),
    rhat_max = mean(rhat_max),
  ) |>
  ungroup() |>
  group_by(model_label) |>
  summarise(
    ess_bulk_min = min(ess_bulk_min),
    ess_tail_min = min(ess_tail_min),
    rhat_max = max(rhat_max),
  )

diagnostics_tab <- diagnostics_summary |>
  mutate(across(starts_with("ess"), ~ round(., 0) |> format(big.mark = " "))) |>
  mutate(across(starts_with("rhat"), ~ round(., 3))) |>
  kbl(
    booktabs = TRUE,
    linesep = c("", "", "", "\\addlinespace"),
    col.names = c(
      "Model",
      "Minimum ESS (Bulk)",
      "Minimum ESS (Tail)",
      "Maximum $\\hat{R}$"
    ),
    format = "latex",
    align = c("lrrc"),
    escape = FALSE
  ) |>
  kable_styling(full_width = TRUE) |>
  identity()

cat(diagnostics_tab, file = path(out_dir, "fh_diagnostics.tex"))


errors <- read_csv(path(data_dir, "errors.csv")) |>
  filter(!str_detect(model, "ptm") | str_detect(model, "77-id")) |>
  filter(phase == "posterior") |>
  mutate(
    model_label = case_when(
      model == "gaussian-linear" ~ "Gaussian (lin) +RI",
      model == "gaussian-linear-nori" ~ "Gaussian (lin)",
      model == "gaussian-nonlinear-nori" ~ "Gaussian (nonlin)",
      model == "gaussian-nonlinear" ~ "Gaussian (nonlin) +RI",
      model == "ptm-linear-77-id" ~ "PTM (lin) +RI",
      model == "ptm-linear-nori-77-id" ~ "PTM (lin)",
      model == "ptm-nonlinear-nori-77-id" ~ "PTM (nonlin)",
      model == "ptm-nonlinear-77-id" ~ "PTM (nonlin) +RI"
    )
  ) |>
  mutate(
    model_label = factor(
      model_label,
      levels = c(
        "PTM (lin)",
        "PTM (lin) +RI",
        "PTM (nonlin)",
        "PTM (nonlin) +RI",
        "Gaussian (lin)",
        "Gaussian (lin) +RI",
        "Gaussian (nonlin)",
        "Gaussian (nonlin) +RI"
      )
    )
  ) |>
  identity()

errors_summary <- errors |>
  group_by(model_label, error_msg) |>
  summarise(
    rel = mean(relative)
  ) |>
  filter(error_msg != "divergent transition + maximum tree depth")

errors_tab <- errors_summary |>
  mutate(across(starts_with("rel"), ~ round(., 3))) |>
  kbl(
    booktabs = TRUE,
    linesep = c(
      "",
      "\\addlinespace",
      "",
      "\\addlinespace",
      "",
      "\\addlinespace",
      "",
      "\\addlinespace",
      "",
      "",
      "",
      "",
      ""
    ),
    col.names = c(
      "Model",
      "Note",
      "Rel. freq."
    ),
    format = "latex",
    align = c("lrl")
  ) |>
  kable_styling(full_width = TRUE) |>
  column_spec(1, width = "10em") |>
  column_spec(2, width = "15em") |>
  identity()

cat(errors_tab, file = path(out_dir, "fh_errors.tex"))
