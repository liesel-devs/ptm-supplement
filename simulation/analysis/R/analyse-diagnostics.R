library(tidyverse)
library(fs)
library(kableExtra)

# Get the directory of the currently active file
current_file <- rstudioapi::getActiveDocumentContext()$path
current_dir <- dirname(current_file)

data_dir <- path(current_dir, "..", "data")
out_dir <- path(current_dir, "..", "out")
dir_create(out_dir)

diagnostics <- read_csv(path(data_dir, "diagnostics.csv")) |>
  mutate(
    model = case_when(
      str_detect(job, "jitter") ~ "PTM-jitter",
      str_detect(job, "ptm") ~ "PTM",
      TRUE ~ model
    )
  )

# ..............................................................................
# ---- Diagnostics Table (jitter vs. no jitter) ----
# ..............................................................................

diag_tab <- diagnostics |>
  filter(str_detect(job, "ptm")) |>
  filter(!is.na(kernel)) |>
  glimpse() |>
  group_by(model, kernel, ntrain) |>
  summarise(
    ap = mean(acceptance_prob),
    rhat = mean(rhat_max),
    ess_bulk_per_minute = mean(ess_bulk_min_per_minute),
    ess_tail_per_minute = mean(ess_tail_min_per_minute),
    ess_bulk = mean(ess_bulk_min),
    ess_tail = mean(ess_tail_min)
  ) |>
  mutate(
    ap = round(ap, 2) |> format(nsmall = 2),
    rhat = round(rhat, 2) |> format(nsmall = 2)
  ) |>
  mutate(across(starts_with("ess"), ~ round(., 1) |> format(nsmall = 1))) |>
  kbl(
    booktabs = TRUE,
    linesep = c("", "", "\\addlinespace"),
    col.names = c(
      "Model",
      "MCMC",
      "$N_{\\text{train}}$",
      "$\\alpha$",
      "Max. $\\hat{R}$",
      "Bulk / Min.",
      "Tail / Min.",
      "Bulk",
      "Tail"
    ),
    format = "latex",
    # align = c("rccccc"),
    escape = FALSE
  ) |>
  kable_styling(full_width = TRUE) |>
  add_header_above(c(
    " " = 5,
    "Minimum Effective Sample Size" = 4
  )) |>
  column_spec(1, width = "7em") |>
  identity()

cat(diag_tab, file = path(out_dir, "sim2_diag.tex"))

# ..............................................................................
# ---- MCMC Errors Table (jitter vs. no jitter) ----
# ..............................................................................

errors <- read_csv(path(data_dir, "errors.csv")) |>
  mutate(
    model = case_when(
      str_detect(job, "ptm") & !str_detect(job, "jitter") ~ "ptm",
      str_detect(job, "ptm") & str_detect(job, "jitter") ~ "ptm-jitter",
      TRUE ~ model
    )
  ) |>
  mutate(
    model = case_when(
      model == "ptm" ~ "PTM",
      model == "ptm-jitter" ~ "PTM-jitter",
      TRUE ~ model
    )
  )


errors_summary <- errors |>
  filter(str_detect(job, "ptm")) |>
  filter(!is.na(kernel)) |>
  filter(phase == "posterior") |>
  glimpse() |>
  group_by(model, error_msg) |>
  summarise(
    rel = mean(relative)
  ) |>
  filter(error_msg != "divergent transition + maximum tree depth") |>
  identity()


errors_tab <- errors_summary |>
  mutate(across(starts_with("rel"), ~ round(., 3))) |>
  kbl(
    booktabs = TRUE,
    linesep = c(
      "",
      "",
      "\\addlinespace"
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

cat(errors_tab, file = path(out_dir, "sim2_errors.tex"))
