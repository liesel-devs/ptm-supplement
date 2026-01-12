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
  legend.position = "top",
  legend.title = element_blank(),
  legend.key.size = unit(1, 'cm'),
  legend.key.height = unit(0.5, 'cm'),
  legend.key.width = unit(2, 'cm')
)

# ..............................................................................
# ---- Data import ----
# ..............................................................................

diagnostics <- read_csv("sim1/data/hyperparameters/diagnostics.csv")
errors <- read_csv("sim1/data/hyperparameters/errors.csv")
config <- read_csv("sim1/data/hyperparameters/config.csv")

config <- config |>
  mutate(
    data = case_when(
      data == "gaussian" ~ "Gaussian data",
      data == "ptm" ~ "PTM data"
    )
  )

errors <- errors |>
  left_join(
    config |> select(run, condition_number, condition, data, nobs),
    by = "run"
  ) |>
  ungroup() |>
  filter(condition != "identity_extrap_error")

diagnostics <- diagnostics |>
  left_join(
    config |>
      select(run, condition_number, condition, nparam, data, nobs, seed),
    by = c("run")
  ) |>
  filter(condition != "identity_extrap_error")

diag_long <- diagnostics |>
  select(
    -c(
      run,
      tau_position_moved,
      rhat_tau_clipped,
      ess_bulk_per_minute_tau_clipped,
      ess_tail_per_minute_tau_clipped,
      ess_bulk_tau_clipped,
      ess_tail_tau_clipped
    )
  ) |>
  pivot_longer(
    -c(nparam, data, nobs, seed, condition, condition_number),
    names_to = "criterium",
    values_to = "value"
  ) |>
  # mutate(nobs = factor(nobs)) |>
  identity()


# ..............................................................................
# ---- Consistent colors ----
# ..............................................................................

condition_colors <- cbPalette
names(condition_colors) <- diagnostics |> pull(condition) |> unique()

options(ggplot2.discrete.colour = condition_colors)
options(ggplot2.discrete.fill = condition_colors)

# ..............................................................................
# ---- Consistent shapes ----
# ..............................................................................

shapes <- c(1, 2, 3, 4, 5, 8, 13)
names(shapes) <- diagnostics |> pull(condition) |> unique()


# --------------------------------------------------------------------------- #
# Effective Sample Size per Minute: Delta
# --------------------------------------------------------------------------- #

# Notable disadvantage J30
# Notable advantage with tail ESS for IG with Gaussian data
# Pattern persists with larger sample sizes

(plt.ess.delta <- diag_long |>
  # filter(nobs == 25) |>
  mutate(nobs = factor(nobs)) |>
  filter(
    str_detect(criterium, "ess"),
    str_detect(criterium, "per_minute"),
    !str_detect(criterium, "median"),
    str_detect(criterium, "delta")
  ) |>
  mutate(
    criterium = case_when(
      str_detect(criterium, "bulk") ~ "bulk",
      str_detect(criterium, "tail") ~ "tail"
    )
  ) |>
  ggplot() +
  aes(nobs, value, color = condition) +
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
  labs(x = TeX("$N_{train}$"), y = TeX("Min. (ESS / m) for $\\delta$")) +
  labs(title = TeX("Effective Samples Drawn per Minute for $\\delta$")) +
  facet_wrap(criterium ~ data, ncol = 4))

(plt.ess.delta + canvas(height = 110, width = 165, units = "mm")) |>
  save_ggplot(file = "sim1/out/ess_per_min_delta.pdf")

# ..............................................................................
# ---- Effective sample size for tau ----
# ..............................................................................

(plt.ess.tau <- diag_long |>
  # filter(nobs == 25) |>
  mutate(nobs = factor(nobs)) |>
  filter(
    str_detect(criterium, "ess"),
    str_detect(criterium, "per_minute"),
    !str_detect(criterium, "median"),
    str_detect(criterium, "tau")
  ) |>
  mutate(
    criterium = case_when(
      str_detect(criterium, "bulk") ~ "bulk",
      str_detect(criterium, "tail") ~ "tail"
    )
  ) |>
  ggplot() +
  aes(nobs, value, color = condition) +
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
  labs(x = TeX("$N_{train}$"), y = TeX("(ESS / m) for $\\tau^2$")) +
  facet_wrap(criterium ~ data, ncol = 4) +
  ylim(c(0, NA)) +
  labs(title = TeX("Effective Samples Drawn per Minute for $\\tau^2$")))


(plt.ess.tau + canvas(height = 110, width = 165, units = "mm")) |>
  save_ggplot(file = "sim1/out/ess_per_min_tau.pdf")

# plt.ess.tau <- plt.ess.tau +
#   theme(legend.position = "none") +
#   canvas(height = 70, width = 165, units = "mm")
#
# save_ggplot(plt.ess.tau, file = "sim1/out/ess_per_min_tau_no_legend.pdf")

# ..............................................................................
# ---- Rhat ----
# ..............................................................................

(plt.rhat <- diag_long |>
  # filter(nobs == 25) |>
  mutate(nobs = factor(nobs)) |>
  filter(
    str_detect(criterium, "rhat"),
    !str_detect(criterium, "median")
  ) |>
  mutate(
    criterium = case_when(
      str_detect(criterium, "delta") ~
        TeX("$\\delta$", bold = TRUE, output = "character"),
      str_detect(criterium, "tau") ~ TeX("$\\tau^2$", output = "character")
    )
  ) |>
  mutate(
    data = case_when(
      str_detect(data, "Gaussian") ~ "Gaussian~data",
      str_detect(data, "PTM") ~ "PTM~data",
    )
  ) |>
  ggplot() +
  aes(nobs, value, color = condition) +
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
  labs(x = TeX("$N_{train}$"), y = TeX("Avg. max. $\\hat{R}$")) +
  facet_wrap(criterium ~ data, ncol = 4, labeller = label_parsed) +
  geom_hline(yintercept = c(1.01), color = "grey", linetype = "dotted") +
  # ylim(c(0, NA)) +
  labs(title = TeX("Convergence, measured by $\\hat{R}$")))

(plt.rhat +
  canvas(height = 100, width = 165, units = "mm")) |>
  save_ggplot(file = "sim1/out/max_rhat.pdf")

# ..............................................................................
# ---- Combine ----
# ..............................................................................

plt <- plt.ess.delta /
  plt.ess.tau /
  plt.rhat +
  plot_layout(guides = "collect") &
  theme(text = element_text(size = 8)) &
  theme(legend.position = "bottom") &
  NULL


(plt + canvas(height = 200, width = 165, units = "mm")) |>
  save_ggplot(file = "sim1/out/diag-ess_rhat.pdf")

# ..............................................................................
# ---- Acceptance prob. ----
# ..............................................................................

(plt.aprob <- diag_long |>
  # filter(nobs == 25) |>
  mutate(nobs = factor(nobs)) |>
  filter(
    str_detect(criterium, "prob"),
    str_detect(criterium, "delta")
  ) |>
  mutate(
    criterium = case_when(
      str_detect(criterium, "delta") ~
        TeX("$\\delta$", bold = TRUE, output = "character"),
      str_detect(criterium, "tau") ~ TeX("$\\tau^2$", output = "character")
    )
  ) |>
  mutate(
    data = case_when(
      str_detect(data, "Gaussian") ~ "Gaussian~data",
      str_detect(data, "PTM") ~ "PTM~data",
    )
  ) |>
  ggplot() +
  aes(nobs, value, color = condition) +
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
  labs(x = TeX("$N_{train}$"), y = TeX("Avg. acceptance probability")) +
  facet_wrap(~data, ncol = 4, labeller = label_parsed) +
  # ylim(c(0, NA)) +
  labs(title = "Acceptance Probability"))

(plt.aprob + canvas(height = 110, width = 165, units = "mm")) |>
  save_ggplot(file = "sim1/out/acceptance_prob.pdf")

# ..............................................................................
# ---- Error codes ----
# ..............................................................................

(plt.errors <- errors |>
  filter(phase == "posterior") |>
  filter(error_msg == "divergent transition") |>
  filter(str_detect(variable, "latent")) |>
  mutate(nobs = factor(nobs)) |>
  mutate(
    variable = case_when(
      str_detect(variable, "latent") ~
        TeX("$\\delta$", bold = TRUE, output = "character"),
      str_detect(variable, "scale") ~ TeX("$\\tau^2$", output = "character")
    )
  ) |>
  mutate(
    data = case_when(
      str_detect(data, "Gaussian") ~ "Gaussian~data",
      str_detect(data, "PTM") ~ "PTM~data",
    )
  ) |>
  ggplot() +
  aes(nobs, relative, color = condition) +
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
  facet_wrap(~data, ncol = 4, labeller = label_parsed) +
  scale_shape_manual(values = shapes) +
  labs(x = TeX("$N_{train}$"), y = TeX("Avg. share of divergent transitions")) +
  theme(legend.position = "top") +
  labs(title = "Share of Divergent Transitions") +
  NULL)

(plt.errors + canvas(height = 110, width = 165, units = "mm")) |>
  save_ggplot(file = "sim1/out/errors.pdf")


# ..............................................................................
# ---- Combine ----
# ..............................................................................

plt <- plt.aprob +
  plt.errors +
  plot_layout(guides = "collect") &
  theme(text = element_text(size = 8)) &
  theme(legend.position = "bottom") &
  NULL

(plt + canvas(height = 90, width = 165, units = "mm")) |>
  save_ggplot(file = "sim1/out/diag-aprob_errors.pdf")
