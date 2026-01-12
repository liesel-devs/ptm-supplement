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
  "#999999", # grey
  "#0072B2", # blue
  "#F0E442" # bright yellow
)


theme_set(
  theme_bw()
)

theme_update(
  legend.position = "bottom",
  legend.title = element_blank(),
  legend.key.size = unit(0.5, 'cm'),
  legend.key.height = unit(0.5, 'cm'),
  legend.key.width = unit(1, 'cm')
)

# ..............................................................................
# ---- Data import ----
# ..............................................................................

analysis_kde <- read_csv("sim1/data/kde/analysis.csv")

analysis_point_gaussian <- analysis_kde |>
  rename(mad_test = mad_gaussian) |>
  rename(kld_test = kld_gaussian) |>
  mutate(model = "point-gaussian")

analysis_kde <- analysis_kde |>
  mutate(model = "kde") |>
  rename(mad_test = mad_kde) |>
  rename(kld_test = kld_kde_floor)


analysis <- read_csv("sim1/data/ptm_and_gaussian/analysis.csv")
config <- read_csv("sim1/data/ptm_and_gaussian/config.csv")
analysis <- analysis |> left_join(config, by = c("model", "run"))

analysis <- analysis |>
  bind_rows(analysis_kde, analysis_point_gaussian) |>
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


# ..............................................................................
# ---- Consistent colors ----
# ..............................................................................

model_colors <- cbPalette
names(model_colors) <- analysis |> pull(model) |> unique()

options(ggplot2.discrete.colour = model_colors)
options(ggplot2.discrete.fill = model_colors)


# ..............................................................................
# ---- KLD ----
# ..............................................................................

(plt.kld <- analysis |>
  mutate(nobs = factor(nobs)) |>
  ggplot() +
  aes(nobs, kld_test, color = model, group = model) +
  # stat_summary(
  #     geom = "linerange",
  #     fun.data = function(y) {
  #         q <- quantile(y, probs = c(0.05, 0.95), na.rm = TRUE)
  #         data.frame(ymin = q[1], ymax = q[2], y = mean(y, na.rm = TRUE))
  #     },
  #     alpha = 0.6,
  #     position = position_dodge(width = 0.2)
  # ) +
  stat_summary(
    geom = "line",
    fun = "mean",
    aes(group = model, linetype = model)
  ) +
  stat_summary(
    geom = "point",
    fun = "mean",
    aes(group = model),
    color = "white",
    size = 2,
    # position = position_dodge2(width = 0.2)
  ) +
  stat_summary(
    aes(shape = model),
    geom = "point",
    fun = "mean",
    # position = position_dodge2(width = 0.2)
  ) +
  facet_wrap(~data, labeller = label_both, scales = "free_y") +
  labs(x = TeX("$N_{train}$"), y = "KLD"))

(plt.kld + canvas(height = 100, width = 165, units = "mm")) |>
  save_ggplot(file = "sim1/out/kld.pdf")

# ..............................................................................
# ---- Coverage (HDI) ----
# ..............................................................................

(plt <- analysis |>
  mutate(nobs = factor(nobs)) |>
  ggplot() +
  aes(nobs, in_hdi_cdf, color = model, group = model) +
  geom_hline(yintercept = 0.9) +
  stat_summary(
    geom = "line",
    fun = "mean",
    aes(group = model, linetype = model)
  ) +
  stat_summary(
    geom = "point",
    fun = "mean",
    aes(group = model),
    color = "white",
    size = 2,
    # position = position_dodge2(width = 0.2)
  ) +
  stat_summary(
    aes(shape = model),
    geom = "point",
    fun = "mean",
    # position = position_dodge2(width = 0.2)
  ) +
  facet_wrap(~data, labeller = label_both, scales = "free_y") +
  labs(x = TeX("$N_{train}$"), y = "Coverage (CDF)") +
  ylim(c(0, 1)) +
  canvas(height = 100, width = 165, units = "mm"))

save_ggplot(plt, file = "sim1/out/coverage_cdf.pdf")
