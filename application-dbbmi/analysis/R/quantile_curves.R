library(tidyverse)
library(fs)
library(latex2exp)

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

data_dir <- path(path_wd(), "application-dbbmi", "analysis", "data")
out_dir <- path(path_wd(), "application-dbbmi", "analysis", "out")
dir_create(out_dir)

pred <- read_csv(path(data_dir, "predictions.csv"))

db <- read_csv(path(path_wd(), "data", "dbbmi.csv"))

bmi_mean <- db$bmi |> mean()
bmi_sd <- db$bmi |> sd()
age_mean <- db$age |> mean()
age_sd <- db$age |> sd()

pred <- pred |>
  mutate(age = age * age_sd + age_mean) |>
  mutate(predicted_quantile = predicted_quantile * bmi_sd + bmi_mean) |>
  filter(
    !(((alpha > 0.2) & (alpha < 0.4)) | ((alpha > 0.6) & (alpha < 0.8)))
  ) |>
  identity()


# ..............................................................................
# ---- PTM, QGAM, DDPSTAR ----
# ..............................................................................

pred |>
  filter(
    model %in% c("ddpstar", "qgam", "ptm-47-id")
  ) |>
  mutate(
    model = case_when(
      model == "ddpstar" ~ "DDPstar",
      model == "ptm-47-id" ~ "PTM",
      model == "qgam" ~ "QGAM"
    )
  ) |>
  ggplot() +
  aes(age, predicted_quantile) +
  geom_point(
    aes(age, bmi),
    data = db,
    alpha = 0.1,
    color = cbPalette[5],
    size = 1
  ) +
  geom_line(
    aes(
      group = factor(alpha)
    ),
    # color = "#F8766D",
    # color = "#00BFC4",
    # color = "#619CFF",
    linewidth = 0.6
  ) +
  labs(x = "Age", y = "Body Mass Index", title = "c) Quantile Curves") +
  theme_light() +
  theme(
    legend.position = "inside",
    legend.position.inside = c(0.3, 0.8),
    legend.background = element_blank(),
    legend.title = element_blank(),
    panel.grid = element_blank(),
    strip.background = element_blank(),
    strip.text = element_text(color = "black"),
    text = element_text(size = 8)
  ) +
  facet_wrap(~model)

ggsave(path(out_dir, "qcurves.pdf"), width = 6.5, height = 2.5)

# ..............................................................................
# ---- Models in separate panels ----
# ..............................................................................

pred <- pred |>
  mutate(
    model_label = case_when(
      model == "bctm" ~ "BCTM",
      model == "ddpstar" ~ "DDPstar",
      model == "gaussian" ~ "Gaussian",
      model == "kowal" ~ "SBGP",
      model == "ptm-47-id" ~
        TeX(
          "$PTM^*, a = -4, b = 7, \\lambda = 0.1(b-a)$",
          output = "character"
        ),
      model == "ptm-47-co" ~
        TeX(
          "$PTM, a = -4, b = 7, \\lambda \\to \\infty$",
          output = "character"
        ),
      model == "ptm-44-id" ~
        TeX(
          "$PTM, a = -4, b = 4, \\lambda = 0.1(b-a)$",
          output = "character"
        ),
      model == "ptm-44-co" ~
        TeX(
          "$PTM, a = -4, b = 4, \\lambda \\to \\infty$",
          output = "character"
        ),
      model == "ptm-77-co" ~
        TeX(
          "$PTM, a = -7, b = 7, \\lambda = \\to \\infty$",
          output = "character"
        ),
      model == "ptm-77-id" ~
        TeX(
          "$PTM, a = -7, b = 7, \\lambda = 0.1(b-a)$",
          output = "character"
        ),
      model == "tamls" ~ "TAMLS",
      model == "qgam" ~ "QGAM",
      TRUE ~ model
    )
  ) |>
  mutate(
    model = case_when(
      model == "bctm" ~ "BCTM",
      model == "ddpstar" ~ "DDPstar",
      model == "gaussian" ~ "Gaussian",
      model == "kowal" ~ "SBGP",
      model == "ptm-47-id" ~ "PTM*",
      model == "tamls" ~ "TAMLS",
      model == "qgam" ~ "QGAM",
      TRUE ~ model
    )
  )


# ..............................................................................
# ---- All models ----
# ..............................................................................

pred_all <- pred |>
  filter(!str_detect(model, "ptm"))

pred_all |>
  ggplot() +
  aes(age, predicted_quantile) +
  geom_point(
    aes(age, bmi),
    data = db,
    alpha = 0.1,
    color = cbPalette[5],
    size = 1
  ) +
  geom_line(
    aes(
      group = factor(alpha),
    ),
    data = pred_all |> filter(model != "SBGP"),
    # color = cbPalette[5],
    linewidth = 0.7,
  ) +
  geom_line(
    aes(group = factor(alpha)),
    linewidth = 0.2,
    data = pred_all |> filter(model == "SBGP"),
    # color = cbPalette[5]
  ) +
  geom_smooth(
    aes(group = factor(alpha)),
    se = FALSE,
    data = pred_all |> filter(model == "SBGP"),
    linewidth = 0.7,
    color = "black"
    # color = cbPalette[5]
  ) +
  labs(
    x = "Age",
    y = "Body Mass Index",
    title = "Quantile curves with different models"
  ) +
  theme_light() +
  theme(
    legend.position = "inside",
    legend.position.inside = c(0.2, 0.8),
    legend.background = element_blank(),
    legend.title = element_blank(),
    panel.grid = element_blank(),
    legend.text = element_text(size = 8)
  ) +
  facet_wrap(~model, ncol = 4)

ggsave(path(out_dir, "qcurves_all.pdf"), width = 6.5, height = 4)

# ..............................................................................
# ---- PTM Variants ----
# ..............................................................................

pred_ptm <- pred |>
  filter(str_detect(model_label, "PTM")) |>
  mutate(model_label = factor(model_label))

ptm_labels <- parse(text = levels(pred_ptm$model_label))

pred_ptm |>
  ggplot() +
  aes(age, predicted_quantile) +
  geom_point(aes(age, bmi), data = db, alpha = 0.05) +
  geom_line(
    aes(
      color = model,
      linetype = model,
      group = interaction(model, factor(alpha)),
    ),
    linewidth = 0.8,
  ) +
  labs(
    x = "Age",
    y = "Body Mass Index",
    title = "Quantile curves with different PTM specifications"
  ) +
  scale_color_discrete(
    labels = ptm_labels
  ) +
  scale_linetype_discrete(
    labels = ptm_labels
  ) +
  theme_light() +
  theme(
    legend.position = "inside",
    legend.position.inside = c(0.2, 0.8),
    legend.background = element_blank(),
    legend.title = element_blank(),
    panel.grid = element_blank(),
    legend.text = element_text(size = 8)
  ) +
  # facet_wrap(~model) +
  NULL

ggsave(path(out_dir, "qcurves_ptm.pdf"), width = 6.5, height = 5)
