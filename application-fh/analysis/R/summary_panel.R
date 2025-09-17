library(tidyverse)
library(fs)
library(patchwork)
library(latex2exp)

data_dir <- path(path_wd(), "application-fh", "analysis", "data")
out_dir <- path(path_wd(), "application-fh", "analysis", "out")
dir_create(out_dir)

loc <- read_csv(path(data_dir, "loc_summary.csv"))
scale <- read_csv(path(data_dir, "scale_summary.csv"))
pred <- read_csv(path(data_dir, "predictions.csv")) |>
  filter(
    !(((alpha > 0.2) & (alpha < 0.4)) | ((alpha > 0.6) & (alpha < 0.8)))
  )


pdfsu <- read_csv(path(data_dir, "prob_summary.csv")) |>
  filter(model == "ptm-nonlinear-nori-77-id") |>
  filter(fold == -1)

grid <- read_csv(path(data_dir, "grid.csv"))[, -1] |>
  rename(obs = index)
rgrid <- read_csv(path(data_dir, "rgrid.csv")) |>
  rename(obs = ...1)

loc_samples <- read_csv(path(data_dir, "loc_samples_summary.csv"))
scale_samples <- read_csv(path(data_dir, "scale_samples_summary.csv"))
log_prob_samples <- read_csv(path(data_dir, "log_prob_samples_summary.csv"))

loc_samples <- loc_samples |> left_join(grid, by = "obs")
scale_samples <- scale_samples |> left_join(grid, by = "obs")
log_prob_samples <- log_prob_samples |>
  left_join(rgrid, by = "obs") |>
  filter(sample <= 50) |>
  filter(fold == -1)

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

fh <- read_csv(path(path_wd(), "data", "framingham.csv")) |>
  mutate(age = age + year)

cholst_mean <- fh$cholst |> mean()
cholst_sd <- fh$cholst |> sd()
age_mean <- fh$age |> mean()
age_sd <- fh$age |> sd()

pred <- pred |>
  mutate(age = age * age_sd + age_mean) |>
  mutate(predicted_quantile = predicted_quantile * cholst_sd + cholst_mean)


loc <- loc |>
  filter(model == "ptm-nonlinear-nori-77-id") |>
  filter(fold == -1) |>
  mutate(age = age_std * age_sd + age_mean) |>
  mutate(mean = mean * cholst_sd) |>
  mutate(`q_0.05` = `q_0.05` * cholst_sd) |>
  mutate(`q_0.95` = `q_0.95` * cholst_sd)

scale <- scale |>
  filter(model == "ptm-nonlinear-nori-77-id") |>
  filter(fold == -1) |>
  mutate(age = age_std * age_sd + age_mean) |>
  mutate(mean = exp(mean)) |>
  mutate(`q_0.05` = exp(`q_0.05`)) |>
  mutate(`q_0.95` = exp(`q_0.95`))

scale_samples <- scale_samples |>
  mutate(age = age * age_sd + age_mean) |>
  mutate(scale = exp(scale)) |>
  filter(fold == -1) |>
  filter(sex == 0) |>
  filter(sample <= 50)

loc_samples <- loc_samples |>
  mutate(age = age * age_sd + age_mean) |>
  mutate(loc = loc * cholst_sd) |>
  filter(fold == -1) |>
  filter(sex == 0) |>
  filter(sample <= 50)

(plot.loc <- ggplot() +
  geom_hline(yintercept = 0, color = "grey") +

  geom_ribbon(
    aes(age, ymin = `q_0.05`, ymax = `q_0.95`),
    data = loc,
    alpha = 0.2,
    fill = cbPalette[1]
  ) +
  geom_line(
    aes(age, loc, group = sample),
    data = loc_samples,
    color = cbPalette[1],
    alpha = 0.2
  ) +
  geom_line(
    aes(age, mean),
    data = loc,
    # color = cbPalette[1]
  ) +
  labs(x = "Age", y = "s(Age)", title = "c) Location Effect of Age") +
  theme_light() +
  theme(
    legend.position = "inside",
    legend.position.inside = c(0.3, 0.8),
    legend.background = element_blank(),
    legend.title = element_blank(),
    panel.grid = element_blank(),
    legend.text = element_text(size = 7)
  ) +
  NULL)

(plot.scale <- ggplot() +
  geom_hline(yintercept = 1, color = "grey") +
  geom_ribbon(
    aes(age, ymin = `q_0.05`, ymax = `q_0.95`),
    data = scale,
    alpha = 0.2,
    fill = cbPalette[2]
  ) +
  geom_line(
    aes(age, scale, group = sample),
    data = scale_samples,
    color = cbPalette[2],
    alpha = 0.2
  ) +
  geom_line(
    aes(age, mean),
    data = scale,
    # color = cbPalette[2]
  ) +
  labs(x = "Age", y = TeX("$exp(g(Age))$"), title = "d) Scale Effect of Age") +
  theme_light() +
  theme(
    legend.position = "inside",
    legend.position.inside = c(0.3, 0.8),
    legend.background = element_blank(),
    legend.title = element_blank(),
    panel.grid = element_blank(),
    legend.text = element_text(size = 7)
  ) +
  NULL)

(plot.dens <- ggplot() +
  stat_function(fun = dnorm, color = "grey") +
  geom_ribbon(
    aes(r, ymin = `q_0.05`, ymax = `q_0.95`),
    data = pdfsu,
    alpha = 0.2,
    fill = cbPalette[3]
  ) +
  geom_line(
    aes(r, exp(log_prob), group = sample),
    data = log_prob_samples,
    color = cbPalette[3],
    alpha = 0.2
  ) +
  geom_line(
    aes(r, mean),
    data = pdfsu,
    # color = cbPalette[3]
  ) +
  labs(
    x = "r",
    y = TeX("$\\hat{f}_R(r)$"),
    title = "a) Standardized Conditional Density"
  ) +
  xlim(c(-3.5, 5)) +
  theme_light() +
  theme(
    legend.position = "inside",
    legend.position.inside = c(0.3, 0.8),
    legend.background = element_blank(),
    legend.title = element_blank(),
    panel.grid = element_blank(),
    legend.text = element_text(size = 7)
  ) +
  NULL)


(plot.qcurve <- pred |>
  filter(model == "ptm-nonlinear-nori-77-id") |>
  filter(sex == 0) |>
  ggplot() +
  aes(age, predicted_quantile) +
  geom_point(
    aes(age, cholst),
    data = fh,
    alpha = 0.2,
    size = 1,
    color = cbPalette[5]
  ) +
  geom_line(
    aes(group = factor(alpha)),
    # color = cbPalette[5],
    linewidth = 0.6
  ) +
  labs(
    x = "Age",
    y = "Cholesterol",
    title = "b) Quantile Curves"
  ) +
  theme_light() +
  theme(
    legend.position = "inside",
    legend.position.inside = c(0.3, 0.8),
    legend.background = element_blank(),
    legend.title = element_blank(),
    panel.grid = element_blank(),
    legend.text = element_text(size = 7)
  ) +
  NULL)

plot.dens +
  plot.qcurve +
  plot.loc +
  plot.scale &
  theme(text = element_text(size = 7))

ggsave(path(out_dir, "fh_summary.pdf"), width = 6.5, height = 4.5)
