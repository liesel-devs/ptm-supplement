library(tidyverse)
library(fs)
library(latex2exp)
library(kableExtra)


# Get the directory of the currently active file
current_file <- rstudioapi::getActiveDocumentContext()$path
current_dir <- dirname(current_file)

data_dir <- path(current_dir, "..", "data")
out_dir <- path(current_dir, "..", "out")
dir_create(out_dir)

dist <- read_csv(path(data_dir, "dist.csv"))

cbPalette <- c(
  "#E69F00", # orangy yellow
  "#56B4E9", # light blue
  "#009E73", # green
  "#999999", # grey
  # "#F0E442", # bright yellow
  "#0072B2", # blue
  "#D55E00", # orangy red
  "#CC79A7" # saturated pink
)

options(ggplot2.discrete.colour = cbPalette)
options(ggplot2.discrete.fill = cbPalette)

# ..............................................................................
# ---- Compare the SBGP / Kowal model variants ----
# ..............................................................................

dist |>
  filter(model == "kowal") |>
  group_by(job) |>
  summarise(
    kld = mean(kld, na.rm = T),
    cdf_mad = mean(cdf_mad, na.rm = T),
    # log_score = mean(log_score, na.rm = T),
    crps = mean(crps, na.rm = T),
    cdf_ci_coverage = mean(cdf_ci_coverage, na.rm = TRUE),
    cdf_ci_width = mean(cdf_ci_width, na.rm = TRUE),
  )


dist |>
  filter(model == "kowal") |>
  group_by(job) |>
  summarise(n = n())


# exclude less performant SBGP / kowal variant
# 003b performans worse than 003
# 003c has only a small number of finished jobs, most of them errored out
dist <- dist |>
  filter(!str_detect(job, "003b")) |>
  filter(!str_detect(job, "003c"))


# ..............................................................................
# ---- PTM with vs. without jittering ----
# ..............................................................................

dist |>
  filter(str_detect(model, "ptm")) |>
  group_by(job) |>
  summarise(
    kld = mean(kld, na.rm = T),
    cdf_mad = mean(cdf_mad, na.rm = T),
    # log_score = mean(log_score, na.rm = T),
    crps = mean(crps, na.rm = T),
    cdf_ci_coverage = mean(cdf_ci_coverage, na.rm = TRUE),
    cdf_ci_width = mean(cdf_ci_width, na.rm = TRUE),
  )

dist <- dist |>
  mutate(
    model = case_when(
      str_detect(job, "ptm") & !str_detect(job, "jitter") ~ "ptm",
      str_detect(job, "ptm") & str_detect(job, "jitter") ~ "ptm-jitter",
      TRUE ~ model
    )
  )

dist |> distinct(model)

# ..............................................................................
# ---- Compute summary, name models ----
# ..............................................................................

dist <- dist |>
  mutate(
    # because dddpstar was evaluated only on 1000 test observations for
    # performance reasons, we need to scale the log score for comparability
    log_score = ifelse(model == "ddpstar", log_score * 5, log_score)
  ) |>
  mutate(N_train = ntrain) |>
  mutate(
    data_type = case_when(
      data_type == "gaussian" ~ "Gaussian",
      data_type == "mixture" ~ "Mixture",
      data_type == "skewnorm" ~ "Skewnorm",
      data_type == "ptm" ~ "PTM"
    )
  ) |>
  mutate(
    model = case_when(
      model == "bctm-locshift" ~ "BCTM-LS",
      model == "bctm-te" ~ "BCTM-TE",
      model == "ddpstar" ~ "DDPstar",
      model == "gaussian-iwls" ~ "Gaussian",
      model == "kowal" ~ "SBGP",
      model == "tamls" ~ "TAMLS",
      model == "ptm" ~ "PTM",
      model == "ptm-jitter" ~ "PTM-jitter",
      model == "qgam" ~ "QGAM"
    )
  ) |>
  identity()


distsu <- dist |>
  group_by(model, ntrain, data_type, job) |>
  summarise(
    kld = mean(kld, na.rm = T),
    cdf_mad = mean(cdf_mad, na.rm = T),
    # log_score = mean(log_score, na.rm = T),
    crps = mean(crps, na.rm = T),
    cdf_ci_coverage = mean(cdf_ci_coverage, na.rm = TRUE),
    cdf_ci_width = mean(cdf_ci_width, na.rm = TRUE),
    waic = mean(waic)
  ) |>
  ungroup()


# ..............................................................................
# ---- Highly aggregated table ----
# ..............................................................................

distsu_agg <- distsu |>
  group_by(model) |>
  summarise(
    kld = mean(kld),
    crps = mean(crps),
    waic = mean(waic),
    cdf_mad = mean(cdf_mad),
    cdf_ci_coverage = mean(cdf_ci_coverage, na.rm = TRUE),
    cdf_ci_width = mean(cdf_ci_width, na.rm = TRUE),
  ) |>
  arrange(kld)

distsu_tab <- distsu_agg |>
  filter(model != "PTM-jitter") |>
  mutate(waic = ifelse(is.na(waic), "-", round(waic))) |>
  mutate(across(
    c(kld, crps, cdf_mad, cdf_ci_coverage, cdf_ci_width),
    ~ ifelse(
      is.infinite(.) | is.nan(.) | is.na(.),
      "-",
      round(., 3) |> format(nsmall = 3)
    )
  )) |>
  kbl(
    booktabs = TRUE,
    linesep = c(""),
    col.names = c(
      "Model",
      "KLD $\\downarrow$",
      "CRPS $\\downarrow$",
      "WAIC $\\downarrow$",
      "MAD $\\downarrow$",
      "Coverage",
      "Width"
    ),
    format = "latex",
    align = c("rccccc"),
    escape = FALSE
  ) |>
  kable_styling(full_width = TRUE) |>
  add_header_above(c(
    " " = 5,
    "CDF CI" = 2
  )) |>
  identity()

cat(distsu_tab, file = path(out_dir, "sim2_dist.tex"))


# ..............................................................................
# ---- Highly aggregated table (only ntrain 250 and 500) ----
# ..............................................................................
# For comparison to other IG prior specifications in PTM

distsu_agg.subset <- distsu |>
  filter(ntrain <= 500) |>
  group_by(model) |>
  summarise(
    kld = mean(kld),
    crps = mean(crps),
    waic = mean(waic),
    cdf_mad = mean(cdf_mad),
    cdf_ci_coverage = mean(cdf_ci_coverage, na.rm = TRUE),
    cdf_ci_width = mean(cdf_ci_width, na.rm = TRUE),
  ) |>
  arrange(kld)

distsu_tab.subset <- distsu_agg.subset |>
  filter(model != "PTM-jitter") |>
  mutate(waic = ifelse(is.na(waic), "-", round(waic))) |>
  mutate(across(
    c(kld, crps, cdf_mad, cdf_ci_coverage, cdf_ci_width),
    ~ ifelse(
      is.infinite(.) | is.nan(.) | is.na(.),
      "-",
      round(., 3) |> format(nsmall = 3)
    )
  )) |>
  kbl(
    booktabs = TRUE,
    linesep = c(""),
    col.names = c(
      "Model",
      "KLD $\\downarrow$",
      "CRPS $\\downarrow$",
      "WAIC $\\downarrow$",
      "MAD $\\downarrow$",
      "Coverage",
      "Width"
    ),
    format = "latex",
    align = c("rccccc"),
    escape = FALSE
  ) |>
  kable_styling(full_width = TRUE) |>
  add_header_above(c(
    " " = 5,
    "CDF CI" = 2
  )) |>
  identity()


# ..............................................................................
# ---- Detailed comparison to Gaussian model ----
# ..............................................................................

gauss_tab <- distsu |>
  filter(data_type == "Gaussian") |>
  filter(model %in% c("Gaussian", "PTM")) |>
  group_by(model, ntrain) |>
  summarise(
    kld = mean(kld),
    crps = mean(crps),
    waic = mean(waic),
    cdf_mad = mean(cdf_mad),
    cdf_ci_coverage = mean(cdf_ci_coverage, na.rm = TRUE),
    cdf_ci_width = mean(cdf_ci_width, na.rm = TRUE),
  ) |>
  ungroup() |>
  arrange(kld) |>
  mutate(waic = ifelse(is.na(waic), "-", round(waic))) |>
  mutate(across(
    c(kld, crps, cdf_mad, cdf_ci_coverage, cdf_ci_width),
    ~ ifelse(
      is.infinite(.) | is.nan(.) | is.na(.),
      "-",
      round(., 3) |> format(nsmall = 3)
    )
  )) |>
  kbl(
    booktabs = TRUE,
    linesep = c("", "\\addlinespace"),
    col.names = c(
      "Model",
      "$N_{\\text{train}}$",
      "KLD $\\downarrow$",
      "CRPS $\\downarrow$",
      "WAIC $\\downarrow$",
      "MAD $\\downarrow$",
      "Coverage",
      "Width"
    ),
    format = "latex",
    align = c("rcccccc"),
    escape = FALSE
  ) |>
  kable_styling(full_width = TRUE) |>
  add_header_above(c(
    " " = 6,
    "CDF CI" = 2
  )) |>
  identity()

cat(gauss_tab, file = path(out_dir, "sim2_dist_gauss.tex"))

# ..............................................................................
# ---- Disaggregated Plots ----
# ..............................................................................

dist |>
  filter(model != "SBGP") |>
  filter(model != "PTM-jitter") |>
  rename(measure = waic) |>
  filter(!is.na(measure)) |>
  mutate(model = fct_reorder(model, measure, .desc = TRUE, .fun = mean)) |>
  group_by(model, data_type, ntrain) |>
  mutate(sd = sd(measure)) |>
  mutate(mean = mean(measure)) |>
  ungroup() |>
  group_by(data_type, ntrain) |>
  mutate(is_minimum = row_number() == which.min(mean)) |>
  mutate(measure_minimum = ifelse(is_minimum, mean, NA)) |>
  ungroup() |>
  ggplot() +
  aes(model, measure) +
  stat_summary(
    fun = mean,
    geom = "point",
    aes(color = data_type, shape = data_type, size = sd),
    alpha = 0.7,
  ) +
  stat_summary(
    fun = mean,
    geom = "line",
    aes(color = data_type, group = data_type),
    alpha = 0.7,
  ) +
  geom_point(aes(model, measure_minimum), shape = 8, color = "black") +
  coord_flip() +
  facet_wrap(~N_train, scales = "free_x", labeller = label_both) +
  labs(
    y = "WAIC",
    x = "Model",
    title = TeX("WAIC by Data Type and $N_{train}$"),
    size = "SD",
    color = "Data Type",
    fill = "",
    shape = "Data Type"
  ) +
  theme_light() +
  guides(size = "none") +
  theme(
    legend.position = "bottom",
    panel.grid = element_blank(),
    axis.title.y = element_blank()
  ) +
  NULL

ggsave(path(out_dir, "waic.pdf"), width = 6.5, height = 4)


dist |>
  filter(model != "SBGP") |>
  filter(model != "PTM-jitter") |>
  rename(measure = kld) |>
  filter(!is.na(measure)) |>
  mutate(model = fct_reorder(model, measure, .desc = TRUE, .fun = mean)) |>
  group_by(model, data_type, ntrain) |>
  mutate(sd = sd(measure)) |>
  mutate(mean = mean(measure)) |>
  ungroup() |>
  group_by(data_type, ntrain) |>
  mutate(is_minimum = row_number() == which.min(mean)) |>
  mutate(measure_minimum = ifelse(is_minimum, mean, NA)) |>
  ungroup() |>
  ggplot() +
  aes(model, measure) +
  stat_summary(
    fun = mean,
    geom = "point",
    aes(color = data_type, shape = data_type, size = sd),
    alpha = 0.7,
  ) +
  stat_summary(
    fun = mean,
    geom = "line",
    aes(color = data_type, group = data_type),
    alpha = 0.7,
  ) +
  geom_point(aes(model, measure_minimum), shape = 8, color = "black") +
  coord_flip() +
  facet_wrap(~N_train, scales = "free_x", labeller = label_both) +
  labs(
    y = "KLD",
    x = "Model",
    title = TeX("KLD by Data Type and $N_{train}$"),
    size = "SD",
    color = "Data Type",
    fill = "",
    shape = "Data Type"
  ) +
  theme_light() +
  guides(size = "none") +
  theme(
    legend.position = "bottom",
    panel.grid = element_blank(),
    axis.title.y = element_blank()
  ) +
  NULL

ggsave(path(out_dir, "kld.pdf"), width = 6.5, height = 4)


dist |>
  filter(model != "SBGP") |>
  filter(model != "PTM-jitter") |>
  rename(measure = crps) |>
  filter(!is.na(measure)) |>
  mutate(model = fct_reorder(model, measure, .desc = TRUE, .fun = mean)) |>
  group_by(model, data_type, ntrain) |>
  mutate(sd = sd(measure)) |>
  mutate(mean = mean(measure)) |>
  ungroup() |>
  group_by(data_type, ntrain) |>
  mutate(is_minimum = row_number() == which.min(mean)) |>
  mutate(measure_minimum = ifelse(is_minimum, mean, NA)) |>
  ungroup() |>
  ggplot() +
  aes(model, measure) +
  stat_summary(
    fun = mean,
    geom = "point",
    aes(color = data_type, shape = data_type, size = sd),
    alpha = 0.7,
  ) +
  stat_summary(
    fun = mean,
    geom = "line",
    aes(color = data_type, group = data_type),
    alpha = 0.7,
  ) +
  geom_point(aes(model, measure_minimum), shape = 8, color = "black") +
  coord_flip() +
  facet_wrap(~N_train, scales = "free_x", labeller = label_both) +
  labs(
    y = "CRPS",
    x = "Model",
    title = TeX("CRPS by Data Type and $N_{train}$"),
    size = "SD",
    color = "Data Type",
    fill = "",
    shape = "Data Type"
  ) +
  theme_light() +
  guides(size = "none") +
  theme(
    legend.position = "bottom",
    panel.grid = element_blank(),
    axis.title.y = element_blank()
  ) +
  NULL

ggsave(path(out_dir, "crps.pdf"), width = 6.5, height = 4)

dist |>
  filter(model != "PTM-jitter") |>
  # filter(model != "SBGP") |>
  rename(measure = crps) |>
  filter(!is.na(measure)) |>
  mutate(model = fct_reorder(model, measure, .desc = TRUE, .fun = mean)) |>
  group_by(model, data_type, ntrain) |>
  mutate(sd = sd(measure)) |>
  mutate(mean = mean(measure)) |>
  ungroup() |>
  group_by(data_type, ntrain) |>
  mutate(is_minimum = row_number() == which.min(mean)) |>
  mutate(measure_minimum = ifelse(is_minimum, mean, NA)) |>
  ungroup() |>
  ggplot() +
  aes(model, measure) +
  stat_summary(
    fun = mean,
    geom = "point",
    aes(color = data_type, shape = data_type, size = sd),
    alpha = 0.7,
  ) +
  stat_summary(
    fun = mean,
    geom = "line",
    aes(color = data_type, group = data_type),
    alpha = 0.7,
  ) +
  geom_point(aes(model, measure_minimum), shape = 8, color = "black") +
  coord_flip() +
  facet_wrap(~N_train, scales = "free_x", labeller = label_both) +
  labs(
    y = "CRPS",
    x = "Model",
    title = TeX("CRPS by Data Type and $N_{train}$"),
    size = "SD",
    color = "Data Type",
    fill = "",
    shape = "Data Type"
  ) +
  theme_light() +
  guides(size = "none") +
  theme(
    legend.position = "bottom",
    panel.grid = element_blank(),
    axis.title.y = element_blank()
  ) +
  NULL

ggsave(path(out_dir, "crps_sbgp.pdf"), width = 6.5, height = 4)

dist |>
  filter(model != "PTM-jitter") |>
  filter(model != "SBGP") |>
  rename(measure = cdf_mad) |>
  filter(!is.na(measure)) |>
  mutate(model = fct_reorder(model, measure, .desc = TRUE, .fun = mean)) |>
  group_by(model, data_type, ntrain) |>
  mutate(sd = sd(measure)) |>
  mutate(mean = mean(measure)) |>
  ungroup() |>
  group_by(data_type, ntrain) |>
  mutate(is_minimum = row_number() == which.min(mean)) |>
  mutate(measure_minimum = ifelse(is_minimum, mean, NA)) |>
  ungroup() |>
  ggplot() +
  aes(model, measure) +
  stat_summary(
    fun = mean,
    geom = "point",
    aes(color = data_type, shape = data_type, size = sd),
    alpha = 0.7,
  ) +
  stat_summary(
    fun = mean,
    geom = "line",
    aes(color = data_type, group = data_type),
    alpha = 0.7,
  ) +
  geom_point(aes(model, measure_minimum), shape = 8, color = "black") +
  coord_flip() +
  facet_wrap(~N_train, scales = "free_x", labeller = label_both) +
  labs(
    y = "MAD",
    x = "Model",
    title = TeX("MAD by Data Type and $N_{train}$"),
    size = "SD",
    color = "Data Type",
    fill = "",
    shape = "Data Type"
  ) +
  guides(size = "none") +
  theme_light() +
  theme(
    legend.position = "bottom",
    panel.grid = element_blank(),
    axis.title.y = element_blank()
  ) +
  NULL

ggsave(path(out_dir, "mad.pdf"), width = 6.5, height = 4)


dist |>
  filter(model != "PTM-jitter") |>
  # filter(model != "SBGP") |>
  rename(measure = cdf_mad) |>
  filter(!is.na(measure)) |>
  mutate(model = fct_reorder(model, measure, .desc = TRUE, .fun = mean)) |>
  group_by(model, data_type, ntrain) |>
  mutate(sd = sd(measure)) |>
  mutate(mean = mean(measure)) |>
  ungroup() |>
  group_by(data_type, ntrain) |>
  mutate(is_minimum = row_number() == which.min(mean)) |>
  mutate(measure_minimum = ifelse(is_minimum, mean, NA)) |>
  ungroup() |>
  ggplot() +
  aes(model, measure) +
  stat_summary(
    fun = mean,
    geom = "point",
    aes(color = data_type, shape = data_type, size = sd),
    alpha = 0.7,
  ) +
  stat_summary(
    fun = mean,
    geom = "line",
    aes(color = data_type, group = data_type),
    alpha = 0.7,
  ) +
  geom_point(aes(model, measure_minimum), shape = 8, color = "black") +
  coord_flip() +
  facet_wrap(~N_train, scales = "free_x", labeller = label_both) +
  labs(
    y = "MAD",
    x = "Model",
    title = TeX("MAD by Data Type and $N_{train}$"),
    size = "SD",
    color = "Data Type",
    fill = "",
    shape = "Data Type"
  ) +
  guides(size = "none") +
  theme_light() +
  theme(
    legend.position = "bottom",
    panel.grid = element_blank(),
    axis.title.y = element_blank()
  ) +
  NULL

ggsave(path(out_dir, "mad_sbgp.pdf"), width = 6.5, height = 4)


dist |>
  filter(model != "PTM-jitter") |>
  # filter(model != "SBGP") |>
  rename(measure = cdf_ci_coverage) |>
  filter(!is.na(measure)) |>
  mutate(model = fct_reorder(model, measure, .fun = mean)) |>
  group_by(model, data_type, ntrain) |>
  mutate(sd = sd(measure)) |>
  ungroup() |>
  ggplot() +
  aes(model, measure) +
  stat_summary(
    fun = mean,
    geom = "point",
    aes(color = data_type, shape = data_type, size = sd),
    alpha = 0.7,
  ) +
  stat_summary(
    fun = mean,
    geom = "line",
    aes(color = data_type, group = data_type),
    alpha = 0.7,
  ) +
  coord_flip() +
  geom_hline(yintercept = 0.9) +
  facet_wrap(~N_train, scales = "free_x", labeller = label_both) +
  labs(
    y = "Coverage",
    x = "Model",
    title = TeX("CDF Coverage by Data Type and $N_{train}$"),
    size = "SD",
    color = "Data Type",
    fill = "",
    shape = "Data Type"
  ) +
  guides(size = "none") +
  theme_light() +
  theme(
    legend.position = "bottom",
    panel.grid = element_blank(),
    axis.title.y = element_blank()
  ) +
  NULL

ggsave(path(out_dir, "cdf_coverage.pdf"), width = 6.5, height = 4)
