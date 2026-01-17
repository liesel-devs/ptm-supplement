library(tidyverse)
library(fs)
library(kableExtra)

data_dir <- path(path_wd(), "application-fh", "analysis", "data")
out_dir <- path(path_wd(), "application-fh", "analysis", "out")
dir_create(out_dir)

qcrps_fh <- read_csv(fs::path(data_dir, "quantile_crps_fh.csv"))
qcrps_db <- read_csv(fs::path(data_dir, "quantile_crps_db.csv"))

# ..............................................................................
# ---- Analysis ----
# ..............................................................................

qcrps_fh |>
  filter(model == "kowal") |>
  filter(fold > -1) |>
  group_by(job) |>
  summarise(quantile_crps = mean(quantile_crps))

qcrps_db |>
  filter(model == "kowal") |>
  filter(fold > -1) |>
  group_by(job) |>
  summarise(quantile_crps = mean(quantile_crps))

# no relevant difference, so we keep 007-kowal, like in the other results
# reported in the paper
qcrps_db <- qcrps_db |>
  filter(!(job %in% c("007b-kowal-noX")))

# ..............................................................................
# ---- Short version: Main Paper ----
# ..............................................................................

qcrps_db_short <- qcrps_db |>
  filter(fold >= 1) |>
  filter(!str_starts(model, "ptm") | model == "ptm-47-id") |>
  mutate(model = ifelse(str_starts(model, "ptm"), "ptm", model)) |>
  group_by(model) |>
  summarise(
    quantile_crps = mean(quantile_crps, na.rm = TRUE),
  ) |>
  arrange(quantile_crps)

qcrps_gauss_db <- qcrps_db_short |>
  filter(model == "gaussian") |>
  pull(quantile_crps)

qcrps_db_short <- qcrps_db_short |>
  mutate(normalized_qcrps_db = (quantile_crps / qcrps_gauss_db))


qcrps_fh_short <- qcrps_fh |>
  filter(fold >= 1) |>
  filter(!str_starts(model, "ptm") | str_detect(model, "77-id")) |>
  mutate(
    model = case_when(
      model == "ptm-nonlinear-nori-77-id" ~ "ptm-nonlin",
      model == "ptm-nonlinear-77-id" ~ "ptm-nonlin-ri",
      model == "ptm-linear-77-id" ~ "ptm-lin-ri",
      model == "ptm-linear-nori-77-id" ~ "ptm-lin",
      model == "gaussian-linear" ~ "gaussian-lin-ri",
      model == "gaussian-nonlinear" ~ "gaussian-nonlin-ri",
      model == "gaussian-linear-nori" ~ "gaussian-lin",
      model == "gaussian-nonlinear-nori" ~ "gaussian-nonlin",
      TRUE ~ model
    )
  ) |>
  group_by(model) |>
  summarise(
    quantile_crps = mean(quantile_crps, na.rm = TRUE),
  ) |>
  arrange(quantile_crps)


qcrps_gauss_fh <- qcrps_fh_short |>
  filter(model == "gaussian-lin") |>
  pull(quantile_crps)

qcrps_fh_short <- qcrps_fh_short |>
  mutate(normalized_qcrps_fh = (quantile_crps / qcrps_gauss_fh))


qcrps_fh_ri <- qcrps_fh_short |>
  filter(str_detect(model, "ri")) |>
  rename(quantile_crps_ri_fh = quantile_crps) |>
  rename(normalized_qcrps_ri_fh = normalized_qcrps_fh) |>
  mutate(model = str_remove(model, "-ri"))

qcrps_fh_nori <- qcrps_fh_short |>
  filter(!str_detect(model, "ri")) |>
  rename(quantile_crps_nori_fh = quantile_crps) |>
  rename(normalized_qcrps_nori_fh = normalized_qcrps_fh)

qcrps_fh_wide <- qcrps_fh_nori |>
  left_join(qcrps_fh_ri, by = "model")


qcrps_db_wide <- qcrps_db_short |>
  rename(quantile_crps_db = quantile_crps) |>
  rename(normalized_qcrps_db = normalized_qcrps_db) |>
  mutate(
    model = case_when(
      model == "ptm" ~ "ptm-nonlin",
      model == "gaussian" ~ "gaussian-nonlin",
      TRUE ~ model
    )
  )


dist_short <- qcrps_fh_wide |>
  add_row(model = "kowal") |>
  left_join(qcrps_db_wide, by = "model") |>
  mutate(across(everything(), ~ ifelse(is.nan(.), NA, .))) |>
  relocate(
    model,
    quantile_crps_db,
    normalized_qcrps_db,
    quantile_crps_nori_fh,
    normalized_qcrps_nori_fh,
    quantile_crps_ri_fh,
    normalized_qcrps_ri_fh
  ) |>
  mutate(
    order = case_when(
      model == "ptm-nonlin" ~ 1,
      model == "ptm-lin" ~ 2,
      model == "gaussian-nonlin" ~ 3,
      model == "gaussian-lin" ~ 4,
      model == "bctm" ~ 5,
      model == "ddpstar" ~ 6,
      model == "kowal" ~ 7,
      model == "qgam" ~ 8,
      model == "tamls" ~ 9
    )
  ) |>
  arrange(order) |>
  select(-order)


dist_short <- dist_short |>
  mutate(across(
    starts_with("quantile_crps"),
    ~ ifelse(is.na(.), "-", format(round(., 4), nsmall = 4))
  )) |>
  mutate(across(
    starts_with("normalized"),
    ~ ifelse(is.na(.), "-", format(round(., 3), nsmall = 3))
  ))


# dist_short$log_score_db[6] <- cell_spec(dist_short$log_score_db[6], bold = T)

(dist_short_tab <- dist_short |>
  mutate(
    model = case_when(
      model == "ptm-nonlin" ~ "PTM (nonlin)",
      model == "ptm-lin" ~ "PTM (lin)",
      model == "gaussian-nonlin" ~ "Gaussian (nonlin)",
      model == "gaussian-lin" ~ "Gaussian (lin)",
      model == "bctm" ~ "BCTM",
      model == "ddpstar" ~ "DDPstar",
      model == "kowal" ~ "SBGP",
      model == "qgam" ~ "QGAM",
      model == "tamls" ~ "TAMLS"
    )
  ) |>
  kbl(
    booktabs = TRUE,
    linesep = c("", "\\addlinespace", "", "\\addlinespace", rep("", times = 5)),
    col.names = c(
      "Model",
      "QCRPS",
      "Scaled",
      "QCRPS",
      "Scaled",
      "QCRPS",
      "Scaled"
    ),
    format = "latex",
    align = c("rcccccc")
  ) |>
  kable_styling(full_width = TRUE) |>
  add_header_above(c(
    " " = 3,
    "Without RI" = 2,
    "With RI" = 2
  )) |>
  add_header_above(c(
    " " = 1,
    "4th Dutch Growth Study" = 2,
    "Framingham Heart Study" = 4
  )) |>
  column_spec(1, width = "9em") |>
  identity())


cat(dist_short_tab, file = path(out_dir, "fh_db_quantile_crps.tex"))
