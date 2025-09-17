library(tidyverse)
library(fs)
library(kableExtra)

data_dir_fh <- path(path_wd(), "application-fh", "analysis", "data")
data_dir_db <- path(path_wd(), "application-dbbmi", "analysis", "data")

dist_fh <- read_csv(path(data_dir_fh, "dist.csv"))

dist_db <- read_csv(path(data_dir_db, "dist.csv"))

out_dir <- path(path_wd(), "application-fh", "analysis", "out")
dir_create(out_dir)

dist_fh |>
  filter(model == "kowal") |>
  filter(fold > -1) |>
  group_by(job) |>
  summarise(crps = mean(crps), log_score = mean(log_score))

dist_db |>
  filter(model == "kowal") |>
  filter(fold > -1) |>
  group_by(job) |>
  summarise(crps = mean(crps), log_score = mean(log_score))

dist_fh <- dist_fh |>
  filter(!(job %in% c("015b-kowal", "015c-kowal")))

dist_db <- dist_db |>
  filter(!(job %in% c("007b-kowal")))

# ..............................................................................
# ---- Short version: Main Paper ----
# ..............................................................................

dist_db_short <- dist_db |>
  filter(fold >= 1) |>
  filter(!str_starts(model, "ptm") | model == "ptm-47-id") |>
  mutate(model = ifelse(str_starts(model, "ptm"), "ptm", model)) |>
  group_by(model) |>
  summarise(
    crps = mean(crps, na.rm = TRUE),
    log_score = mean(log_score, na.rm = TRUE)
  ) |>
  arrange(log_score)


dist_fh_short <- dist_fh |>
  filter(fold >= 1) |>
  filter(!str_starts(model, "ptm") | str_detect(model, "77-id")) |>
  # mutate(model = ifelse(str_starts(model, "ptm"), "ptm", model)) |>
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
    crps = mean(crps, na.rm = TRUE),
    log_score = mean(log_score, na.rm = TRUE)
  ) |>
  arrange(log_score)


dist_fh_ri <- dist_fh_short |>
  filter(str_detect(model, "ri")) |>
  rename(crps_ri_fh = crps, log_score_ri_fh = log_score) |>
  mutate(model = str_remove(model, "-ri"))

dist_fh_nori <- dist_fh_short |>
  filter(!str_detect(model, "ri")) |>
  rename(crps_nori_fh = crps, log_score_nori_fh = log_score)

dist_fh_wide <- dist_fh_nori |>
  left_join(dist_fh_ri, by = "model")


dist_db_wide <- dist_db_short |>
  rename(crps_db = crps, log_score_db = log_score) |>
  mutate(
    model = case_when(
      model == "ptm" ~ "ptm-nonlin",
      model == "gaussian" ~ "gaussian-nonlin",
      TRUE ~ model
    )
  )


dist_short <- dist_fh_wide |>
  left_join(dist_db_wide, by = "model") |>
  mutate(across(everything(), ~ ifelse(is.nan(.), NA, .))) |>
  relocate(
    model,
    log_score_db,
    crps_db,
    log_score_nori_fh,
    crps_nori_fh,
    crps_ri_fh
  ) |>
  select(-log_score_ri_fh) |>
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
    starts_with("crps"),
    ~ ifelse(is.na(.), "-", format(round(., 3), nsmall = 3))
  )) |>
  mutate(across(
    starts_with("log"),
    ~ ifelse(is.na(.), "-", format(round(., 1), nsmall = 1))
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
      "Log Score",
      "CRPS",
      "Log Score",
      "CRPS",
      "CRPS"
    ),
    format = "latex",
    align = c("rccccc")
  ) |>
  kable_styling(full_width = TRUE) |>
  add_header_above(c(
    " " = 3,
    "Without RI" = 2,
    "With RI" = 1
  )) |>
  add_header_above(c(
    " " = 1,
    "4th Dutch Growth Study" = 2,
    "Framingham Heart Study" = 3
  )) |>
  column_spec(1, width = "9em") |>
  identity())


cat(dist_short_tab, file = path(out_dir, "fh_db_scores.tex"))
