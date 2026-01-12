# ..............................................................................
# ---- Libraries ----
# ..............................................................................
library(tidyverse)
library(latex2exp)
library(kableExtra)
library(ggview)
library(patchwork)


# ..............................................................................
# ---- ggplot options ----
# ..............................................................................

cbPalette <- c(
  "#E69F00", # orangy yellow
  "#56B4E9", # light blue
  "#009E73", # green
  "#CC79A7", # saturated pink
  "#D55E00", # orangy red
  "#0072B2", # blue
  "#999999", # grey
  "#F0E442" # bright yellow
)


theme_set(
  theme_bw()
)

theme_update(
  legend.position = "bottom",
  legend.title = element_blank(),
  legend.key.size = unit(1, 'cm'),
  legend.key.height = unit(0.5, 'cm'),
  legend.key.width = unit(2, 'cm')
)
options(ggplot2.discrete.colour = cbPalette)

# ..............................................................................
# ---- Combine ----
# ..............................................................................

analysis <- read_csv("sim1/data/hyperparameters/analysis.csv")
config <- read_csv("sim1/data/hyperparameters/config.csv")
analysis <- analysis |>
  left_join(config, by = c("run")) |>
  mutate(
    data = factor(
      data,
      levels = c("gaussian", "ptm", "mixture", "skewnorm", "u-shaped", "unif"),
      labels = c("Gaussian", "PTM", "Mixture", "Skewnorm", "U-shaped", "Unif")
    )
  ) |>
  filter(condition != "identity_extrap_error")


analysis_g <- read_csv("sim1/data/ptm_and_gaussian/analysis.csv")
config_g <- read_csv("sim1/data/ptm_and_gaussian/config.csv")
analysis_g <- analysis_g |> left_join(config_g, by = c("model", "run"))

analysis_g <- analysis_g |>
  mutate(
    model = factor(
      model,
      levels = c("kde", "point-gaussian", "gaussian", "ptm", "ptm-J30"),
      labels = c("kde", "gaussian-pt", "gaussian", "ptm-15", "ptm-30")
    )
  ) |>
  mutate(
    data = factor(
      data,
      levels = c("gaussian", "ptm", "mixture", "skewnorm", "u-shaped", "unif"),
      labels = c("Gaussian", "PTM", "Mixture", "Skewnorm", "U-shaped", "Unif")
    )
  ) |>
  filter(model != "gaussian-pt") |>
  identity()

analysis_g <- analysis_g |>
  filter(model == "gaussian") |>
  filter(data %in% c("Gaussian", "PTM")) |>
  mutate(condition = "Gaussian")


(plt.kld <- analysis |>
  filter(condition %in% c("IG", "ap90")) |>
  mutate(
    condition = case_when(
      condition == "ap90" ~ "Weibull(0.5, 0.5)",
      condition == "IG" ~ "IG(0.01, 0.01)"
    )
  ) |>
  bind_rows(analysis_g) |>
  mutate(nobs = factor(nobs)) |>
  ggplot() +
  aes(nobs, kld_test, color = condition, group = condition) +
  stat_summary(
    geom = "line",
    fun = "mean",
    aes(group = condition, linetype = condition)
  ) +
  stat_summary(
    geom = "point",
    fun = "mean",
    aes(group = condition),
    color = "white",
    size = 3,
  ) +
  stat_summary(
    aes(shape = condition),
    geom = "point",
    fun = "mean",
    size = 2,
  ) +
  # scale_color_discrete(cbPalette) +
  # scale_shape_manual(values = shapes) +
  facet_wrap(~data, labeller = label_both, scales = "free_y") +
  labs(
    x = TeX("$N_{train}$"),
    y = "KLD"
  ) +
  theme(
    panel.grid = element_blank(),
    legend.position = "inside",
    legend.position.inside = c(0.3, 0.7),
    legend.background = element_blank(),
    legend.key.width = unit(0.5, 'cm'),
    strip.background = element_blank(),
    text = element_text(size = 8)
  ))

ggsave("sim1/out/kld_2case.pdf", width = 6.5, height = 2)
