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

# ..............................................................................
# ---- Data import ----
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


# ..............................................................................
# ---- Consistent colors ----
# ..............................................................................

condition_colors <- cbPalette
names(condition_colors) <- analysis |> pull(condition) |> unique()

options(ggplot2.discrete.colour = condition_colors)
options(ggplot2.discrete.fill = condition_colors)

# ..............................................................................
# ---- Consistent shapes ----
# ..............................................................................

shapes <- c(1, 2, 3, 4, 5, 8, 13)
names(shapes) <- analysis |> pull(condition) |> unique()


# ..............................................................................
# ---- KLD ----
# ..............................................................................

(plt.kld <- analysis |>
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
    size = 2,
  ) +
  stat_summary(
    aes(shape = condition),
    geom = "point",
    fun = "mean",
  ) +
  scale_shape_manual(values = shapes) +
  facet_wrap(~data, labeller = label_both, scales = "free_y") +
  labs(
    x = TeX("$N_{train}$"),
    y = "KLD",
    title = "Kullback-Leibler Divergence"
  ))

(plt.kld + canvas(height = 100, width = 165, units = "mm")) |>
  save_ggplot(file = "sim1/out/kld-hyperparameters.pdf")
