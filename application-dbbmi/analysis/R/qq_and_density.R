library(tidyverse)
library(fs)
library(latex2exp)
library(patchwork)

data_dir <- path(path_wd(), "application-dbbmi", "analysis", "data")
out_dir <- path(path_wd(), "application-dbbmi", "analysis", "out")
dir_create(out_dir)

zsu <- read_csv(path(data_dir, "z_summary.csv"))
pdfsu <- read_csv(path(data_dir, "prob_summary.csv"))

zsu |>
  filter(str_detect(model, "id") | model == "gaussian") |>
  filter(str_detect(model, "44") | str_detect(model, "47")) |>
  distinct(model)


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


(qq.plot <- zsu |>
  # filter(str_detect(model, "id")) |>
  filter(str_detect(model, "44") | str_detect(model, "47")) |>
  filter(!str_detect(model, "47-co")) |>
  filter(variable == "z") |>
  select(model, variable, mean) |>
  ggplot() +
  geom_abline() +
  geom_qq(aes(sample = mean, color = model), alpha = 0.5, size = 0.6) +
  scale_color_manual(
    values = cbPalette[1:3],
    labels = c(
      "ptm-44-co" = TeX("$a=-4, b=4, \\lambda\\to \\infty$"),
      "ptm-47-id" = TeX("$a=-4, b=7, \\lambda=0.1(b-a)$"),
      "ptm-44-id" = TeX("$a=-4, b=4, \\lambda=0.1(b-a)$")
    )
  ) +
  labs(
    x = "Theoretical Quantile",
    y = "Sample Quantile",
    title = TeX("a) Quantile-Quantile Plot")
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
  guides(
    color = guide_legend(
      override.aes = list(size = 3, alpha = 1)
    )
  ) +

  NULL)

ggsave(path(out_dir, "qq.png"))


(pdf.plot <- pdfsu |>
  # filter(str_detect(model, "id")) |>
  filter(str_detect(model, "44") | str_detect(model, "47")) |>
  filter(!str_detect(model, "47-co")) |>
  select(model, variable, mean, r, `q_0.05`, `q_0.95`) |>
  ggplot() +
  stat_function(fun = dnorm, color = "grey") +
  geom_ribbon(
    aes(r, ymin = `q_0.05`, ymax = `q_0.95`, fill = model),
    alpha = 0.2
  ) +
  geom_line(aes(r, mean, color = model)) +
  labs(
    x = "r",
    y = TeX("$\\hat{f}_R(r)$"),
    title = TeX("b) Posterior Density")
  ) +
  theme_light() +
  theme(
    legend.position = "none",
    legend.position.inside = c(0.8, 0.8),
    legend.background = element_blank(),
    legend.title = element_blank(),
    panel.grid = element_blank()
  ) +
  guides(
    color = guide_legend(
      override.aes = list(size = 3, alpha = 1)
    )
  ) +
  scale_color_manual(
    values = cbPalette[1:3],
    labels = c(
      "ptm-44-co" = TeX("$a=-4, b=4, \\lambda\\to \\infty$"),
      "ptm-47-id" = TeX("$a=-4, b=7, \\lambda=0.1(b-a)$"),
      "ptm-44-id" = TeX("$a=-4, b=4, \\lambda=0.1(b-a)$")
    )
  ) +
  scale_fill_manual(
    values = cbPalette[1:3],
    labels = c(
      "ptm-44-co" = TeX("$a=-4, b=4, \\lambda\\to \\infty$"),
      "ptm-47-id" = TeX("$a=-4, b=7, \\lambda=0.1(b-a)$"),
      "ptm-44-id" = TeX("$a=-4, b=4, \\lambda=0.1(b-a)$")
    )
  ) +
  NULL)


(pdf.plot.tail <- pdfsu |>
  # filter(str_detect(model, "id")) |>
  filter(str_detect(model, "44") | str_detect(model, "47")) |>
  filter(!str_detect(model, "47-co")) |>
  filter(r > 2, r < 6) |>
  select(model, variable, mean, r, `q_0.05`, `q_0.95`) |>
  ggplot() +
  geom_ribbon(
    aes(r, ymin = `q_0.05`, ymax = `q_0.95`, fill = model),
    alpha = 0.2
  ) +
  geom_line(aes(r, mean, color = model)) +
  labs(
    x = "r",
    y = TeX("$\\hat{f}_R(r)$"),
  ) +
  theme_light() +
  theme(
    legend.position = "none",
    legend.position.inside = c(0.8, 0.8),
    legend.background = element_blank(),
    legend.title = element_blank(),
    panel.grid = element_blank(),
    axis.title = element_blank(),
    plot.background = element_blank(),
    axis.text = element_text(size = 6)
  ) +
  guides(
    color = guide_legend(
      override.aes = list(size = 3, alpha = 1)
    )
  ) +
  scale_color_manual(
    values = cbPalette[1:3],
    labels = c(
      "ptm-44-co" = TeX("$a=-4, b=4, \\lambda\\to \\infty$"),
      "ptm-47-id" = TeX("$a=-4, b=7, \\lambda=0.1(b-a)$"),
      "ptm-44-id" = TeX("$a=-4, b=4, \\lambda=0.1(b-a)$")
    )
  ) +
  scale_fill_manual(
    values = cbPalette[1:3],
    labels = c(
      "ptm-44-co" = TeX("$a=-4, b=4, \\lambda\\to \\infty$"),
      "ptm-47-id" = TeX("$a=-4, b=7, \\lambda=0.8$"),
      "ptm-44-id" = TeX("$a=-4, b=4, \\lambda=0.8$")
    )
  ) +
  NULL)


pdf.plot <- pdf.plot + inset_element(pdf.plot.tail, 0.5, 0.4, 0.99, 0.99)

qq.plot + pdf.plot & theme(text = element_text(size = 8))

ggsave(path(out_dir, "qq_density.pdf"), width = 6.5, height = 3)
