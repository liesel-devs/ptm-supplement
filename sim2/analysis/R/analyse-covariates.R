library(tidyverse)
library(fs)
library(kableExtra)

# Get the directory of the currently active file
current_file <- rstudioapi::getActiveDocumentContext()$path
current_dir <- dirname(current_file)

data_dir <- path(current_dir, "..", "data")
out_dir <- path(current_dir, "..", "out")
dir_create(out_dir)

covars <- read_csv(path(data_dir, "covariates.csv"))


# ..............................................................................
# ---- First look ----
# ..............................................................................

covars |> distinct(job)
covars |> distinct(model)

covars |>
  group_by(job, model, parameter) |>
  summarise(
    mse = mean(mse),
    coverage = mean(ci_coverage),
    width = mean(ci_width)
  ) |>
  pivot_wider(
    names_from = parameter,
    values_from = c("mse", "coverage", "width")
  ) |>
  relocate(model, mse_loc, coverage_loc, width_loc) |>
  arrange(mse_loc) |>
  ungroup()

# ..............................................................................
# ---- Data preparation ----
# ..............................................................................

covars <- covars |>
  mutate(
    model = case_when(
      str_detect(job, "ddpstar") ~ "DDPstar",
      str_detect(job, "gaussian") ~ "Gaussian",
      str_detect(job, "ptm") ~ "PTM"
    )
  ) |>
  mutate(
    data_type = case_when(
      data_type == "gaussian" ~ "Gaussian",
      data_type == "mixture" ~ "Mixture",
      data_type == "skewnorm" ~ "Skewnorm",
      data_type == "ptm" ~ "PTM"
    )
  ) |>
  mutate(
    parameter = case_when(
      parameter == "loc" ~ "Loc",
      parameter == "scale" ~ "Scale"
    )
  ) |>
  mutate(Predictor = parameter) |>
  mutate(N_train = ntrain)

# ..............................................................................
# ---- Highly aggregated table ----
# ..............................................................................

covars_summary <- covars |>
  group_by(model, parameter) |>
  summarise(
    mse = mean(mse),
    coverage = mean(ci_coverage),
    width = mean(ci_width)
  ) |>
  pivot_wider(
    names_from = parameter,
    values_from = c("mse", "coverage", "width")
  ) |>
  relocate(model, mse_Loc, coverage_Loc, width_Loc) |>
  arrange(mse_Loc) |>
  ungroup()

covars_tab <- covars_summary |>
  mutate(across(
    -model,
    ~ ifelse(is.na(.), "-", round(., 3) |> format(big.mark = " "))
  )) |>
  kbl(
    booktabs = TRUE,
    linesep = c(""),
    col.names = c(
      "Model",
      "MSE $\\downarrow$",
      "Coverage",
      "Width",
      "MSE $\\downarrow$",
      "Coverage",
      "Width"
    ),
    format = "latex",
    align = c("rcccccc"),
    escape = FALSE
  ) |>
  kable_styling(full_width = TRUE) |>
  add_header_above(c(
    " " = 1,
    "Location Terms" = 3,
    "Scale Terms" = 3
  )) |>
  identity()

cat(covars_tab, file = path(out_dir, "sim2_covars.tex"))


# ..............................................................................
# ---- Highly aggregated table (N 250 and N 500) ----
# ..............................................................................
# For comparison to other IG prior specifications

covars_summary.subset <- covars |>
  filter(N_train <= 500) |>
  group_by(model, parameter) |>
  summarise(
    mse = mean(mse),
    coverage = mean(ci_coverage),
    width = mean(ci_width)
  ) |>
  pivot_wider(
    names_from = parameter,
    values_from = c("mse", "coverage", "width")
  ) |>
  relocate(model, mse_Loc, coverage_Loc, width_Loc) |>
  arrange(mse_Loc) |>
  ungroup()

covars_tab.subset <- covars_summary.subset |>
  mutate(across(
    -model,
    ~ ifelse(is.na(.), "-", round(., 3) |> format(big.mark = " "))
  )) |>
  kbl(
    booktabs = TRUE,
    linesep = c(""),
    col.names = c(
      "Model",
      "MSE $\\downarrow$",
      "Coverage",
      "Width",
      "MSE $\\downarrow$",
      "Coverage",
      "Width"
    ),
    format = "latex",
    align = c("rcccccc"),
    escape = FALSE
  ) |>
  kable_styling(full_width = TRUE) |>
  add_header_above(c(
    " " = 1,
    "Location Terms" = 3,
    "Scale Terms" = 3
  )) |>
  identity()


# ..............................................................................
# ---- Color scale ----
# ..............................................................................

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
# ---- Disaggregated Plots: Bias ----
# ..............................................................................

covars |>
  ggplot() +
  aes(model, bias) +
  geom_boxplot(aes(fill = data_type), outliers = FALSE) +
  ylim(c(-0.0001, 0.0001)) +
  coord_flip() +
  facet_wrap(Predictor ~ N_train, labeller = label_both) +
  labs(
    x = "Model",
    y = "Bias",
    title = "Bias of Covariate Effects by Data Type",
    fill = "Data Type"
  ) +
  theme_light() +
  theme(
    legend.position = "bottom",
    panel.grid = element_blank(),
    legend.text = element_text(size = 8)
  )

ggsave(path(out_dir, "bias_by_data.pdf"), width = 6.5, height = 6)

covars |>
  mutate(xnum = xnum + 1) |>
  ggplot() +
  aes(model, bias) +
  geom_boxplot(aes(fill = factor(xnum)), outliers = FALSE) +
  ylim(c(-0.0001, 0.0001)) +
  coord_flip() +
  facet_wrap(Predictor ~ N_train, labeller = label_both) +
  labs(
    x = "Model",
    y = "Bias",
    title = "Bias of Covariate Effects by Covariate Function",
    fill = "Covariate index"
  ) +
  theme_light() +
  theme(
    legend.position = "bottom",
    panel.grid = element_blank(),
    legend.text = element_text(size = 8)
  )

ggsave(path(out_dir, "bias_by_cov.pdf"), width = 6.5, height = 6)


# ..............................................................................
# ---- Disaggregated Plots: Variance ----
# ..............................................................................

covars |>
  ggplot() +
  aes(model, var) +
  geom_boxplot(aes(fill = data_type), outliers = FALSE) +
  coord_flip() +
  facet_wrap(Predictor ~ N_train, scales = "free_x", labeller = label_both) +
  labs(
    x = "Model",
    y = "Variance",
    title = "Variance of Covariate Effects by Data Type",
    fill = "Data Type"
  ) +
  theme_light() +
  theme(
    legend.position = "bottom",
    panel.grid = element_blank(),
    legend.text = element_text(size = 8)
  )

ggsave(path(out_dir, "var_by_data.pdf"), width = 6.5, height = 6)

covars |>
  mutate(xnum = xnum + 1) |>
  ggplot() +
  aes(model, var) +
  geom_boxplot(aes(fill = factor(xnum)), outliers = FALSE) +
  # ylim(c(-0.0001, 0.0001)) +
  coord_flip() +
  facet_wrap(Predictor ~ N_train, scales = "free_x", labeller = label_both) +
  labs(
    x = "Model",
    y = "Variance",
    title = "Variance of Covariate Effects by Covariate Function",
    fill = "Covariate index"
  ) +
  theme_light() +
  theme(
    legend.position = "bottom",
    panel.grid = element_blank(),
    legend.text = element_text(size = 8)
  )

ggsave(path(out_dir, "var_by_cov.pdf"), width = 6.5, height = 6)


# ..............................................................................
# ---- Disaggregated Plots: MSE ----
# ..............................................................................

covars |>
  ggplot() +
  aes(model, mse) +
  geom_boxplot(aes(fill = data_type), outliers = FALSE) +
  coord_flip() +
  facet_wrap(Predictor ~ N_train, scales = "free_x", labeller = label_both) +
  labs(
    x = "Model",
    y = "MSE",
    title = "MSE of Covariate Effects by Data Type",
    fill = "Data Type"
  ) +
  theme_light() +
  theme(
    legend.position = "bottom",
    panel.grid = element_blank(),
    legend.text = element_text(size = 8)
  )

ggsave(path(out_dir, "mse_by_data.pdf"), width = 6.5, height = 6)

covars |>
  mutate(xnum = xnum + 1) |>
  ggplot() +
  aes(model, mse) +
  geom_boxplot(aes(fill = factor(xnum)), outliers = FALSE) +
  coord_flip() +
  facet_wrap(Predictor ~ N_train, scales = "free_x", labeller = label_both) +
  labs(
    x = "Model",
    y = "MSE",
    title = "MSE of Covariate Effects by Covariate Function",
    fill = "Covariate index"
  ) +
  theme_light() +
  theme(
    legend.position = "bottom",
    panel.grid = element_blank(),
    legend.text = element_text(size = 8)
  )

ggsave(path(out_dir, "mse_by_cov.pdf"), width = 6.5, height = 6)

# ..............................................................................
# ---- Disaggregated Plots: Coverage ----
# ..............................................................................

covars |>
  ggplot() +
  aes(model, ci_coverage) +
  geom_boxplot(aes(fill = data_type), outliers = FALSE) +
  geom_hline(yintercept = 0.9) +
  coord_flip() +
  facet_wrap(Predictor ~ N_train, scales = "free_x", labeller = label_both) +
  labs(
    x = "Model",
    y = "Coverage",
    title = "Coverage of Covariate Effects by Data Type",
    fill = "Data Type"
  ) +
  theme_light() +
  theme(
    legend.position = "bottom",
    panel.grid = element_blank(),
    legend.text = element_text(size = 8)
  )

ggsave(path(out_dir, "cov_coverage_by_data.pdf"), width = 6.5, height = 6)

covars |>
  mutate(xnum = xnum + 1) |>
  ggplot() +
  aes(model, ci_coverage) +
  geom_boxplot(aes(fill = factor(xnum)), outliers = FALSE) +
  geom_hline(yintercept = 0.9) +
  coord_flip() +
  facet_wrap(Predictor ~ N_train, scales = "free_x", labeller = label_both) +
  labs(
    x = "Model",
    y = "Coverage",
    title = "Coverage of Covariate Effects by Covariate Function",
    fill = "Covariate index"
  ) +
  theme_light() +
  theme(
    legend.position = "bottom",
    panel.grid = element_blank(),
    legend.text = element_text(size = 8)
  )

ggsave(path(out_dir, "cov_coverage_by_cov.pdf"), width = 6.5, height = 6)
