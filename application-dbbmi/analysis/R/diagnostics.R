library(tidyverse)
library(fs)
library(kableExtra)

data_dir <- path(path_wd(), "application-dbbmi", "analysis", "data")

diagnostics <- read_csv(path(data_dir, "diagnostics.csv"))
out_dir <- path(path_wd(), "application-dbbmi", "analysis", "out")
dir_create(out_dir)

diagnostics <- diagnostics |>
  filter(fold != -1) |>
  identity()


diagnostics <- diagnostics |>
  mutate(
    model = case_when(
      str_detect(model, "ptm") ~ "PTM",
      model == "gaussian" ~ "Gaussian"
    )
  ) |>
  mutate(
    lambda = ifelse(
      trafo_target_slope == "continue_linearly",
      "$\\to \\infty$",
      "0.1(b-a)"
    )
  )

diagnostics_summary <- diagnostics |>
  group_by(variable, model, a, b, lambda) |>
  summarise(
    ess_bulk_min = mean(ess_bulk_min),
    ess_tail_min = mean(ess_tail_min),
    rhat_max = mean(rhat_max),
  ) |>
  ungroup() |>
  group_by(model, a, b, lambda) |>
  summarise(
    ess_bulk_min = min(ess_bulk_min),
    ess_tail_min = min(ess_tail_min),
    rhat_max = max(rhat_max),
  )


diagnostics_tab <- diagnostics_summary |>
  ungroup() |>
  mutate(across(c(a, b, lambda), ~ ifelse(is.na(.), "-", as.character(.)))) |>
  mutate(across(starts_with("ess"), ~ round(., 0) |> format(big.mark = " "))) |>
  mutate(across(starts_with("rhat"), ~ round(., 3))) |>
  kbl(
    booktabs = TRUE,
    col.names = c(
      "Model",
      "a",
      "b",
      "$\\lambda$",
      "Min. ESS (Bulk)",
      "Min. ESS (Tail)",
      "Max. $\\hat{R}$"
    ),
    format = "latex",
    align = c("lccc"),
    escape = FALSE
  ) |>
  kable_styling(full_width = TRUE) |>
  column_spec(2:3, width = "3em") |>
  column_spec(4, width = "5em") |>
  identity()

cat(diagnostics_tab, file = path(out_dir, "db_diagnostics.tex"))


errors <- read_csv(path(data_dir, "errors.csv")) |>
  filter(phase == "posterior") |>
  filter(fold != -1) |>
  mutate(
    model = case_when(
      model == "ptm" ~ "PTM",
      model == "gaussian" ~ "Gaussian"
    )
  ) |>
  mutate(
    lambda = ifelse(
      trafo_target_slope == "continue_linearly",
      "$\\to \\infty$",
      "0.1(b-a)"
    )
  )

errors_summary <- errors |>
  group_by(model, a, b, lambda, error_msg) |>
  summarise(
    rel = mean(relative)
  )


errors_tab <- errors_summary |>
  ungroup() |>
  mutate(across(c(a, b, lambda), ~ ifelse(is.na(.), "-", as.character(.)))) |>
  mutate(across(starts_with("rel"), ~ round(., 3))) |>
  kbl(
    booktabs = TRUE,
    linesep = c(
      "\\addlinespace",
      "",
      "",
      "",
      "\\addlinespace",
      "",
      "",
      "",
      "\\addlinespace"
    ),
    col.names = c(
      "Model",
      "a",
      "b",
      "$\\lambda$",
      "Note",
      "Rel. freq."
    ),
    format = "latex",
    escape = FALSE,
    align = c("lcccrc")
  ) |>
  kable_styling(full_width = TRUE) |>
  column_spec(1, width = "4em") |>
  column_spec(2:3, width = "3em") |>
  column_spec(4, width = "5em") |>
  column_spec(5, width = "13em") |>
  identity()

cat(errors_tab, file = path(out_dir, "db_errors.tex"))
