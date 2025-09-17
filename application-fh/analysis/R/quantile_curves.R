library(tidyverse)
library(fs)
library(slider)

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

data_dir <- path(path_wd(), "application-fh", "analysis", "data")
out_dir <- path(path_wd(), "application-fh", "analysis", "out")
dir_create(out_dir)

pred <- read_csv(path(data_dir, "predictions.csv"))

fh <- read_csv(path(path_wd(), "data", "framingham.csv"))

fh <- fh |>
  mutate(age = age + year)

cholst_mean <- fh$cholst |> mean()
cholst_sd <- fh$cholst |> sd()
age_mean <- fh$age |> mean()
age_sd <- fh$age |> sd()

pred <- pred |>
  mutate(age = age * age_sd + age_mean) |>
  mutate(predicted_quantile = predicted_quantile * cholst_sd + cholst_mean)

pred <- pred |>
  filter(!(job %in% c("015b-kowal", "015c-kowal")))


pred <- pred |>
  mutate(
    model_label = case_when(
      model == "bctm" ~ "BCTM",
      model == "bctm-ri" ~ "BCTM +RI",
      model == "ddpstar" ~ "DDPstar",
      model == "gaussian-linear" ~ "Gaussian (lin) +RI",
      model == "gaussian-linear-nori" ~ "Gaussian (lin)",
      model == "gaussian-nonlinear-nori" ~ "Gaussian (nonlin)",
      model == "gaussian-nonlinear" ~ "Gaussian (nonlin) +RI",
      model == "ptm-linear-77-id" ~ "PTM (lin) +RI",
      model == "ptm-linear-nori-77-id" ~ "PTM (lin)",
      model == "ptm-nonlinear-nori-77-id" ~ "PTM (nonlin)",
      model == "ptm-nonlinear-77-id" ~ "PTM (nonlin) +RI",
      model == "qgam" ~ "QGAM",
      model == "tamls" ~ "TAMLS",
      model == "kowal" ~ "SBGP",
      model == "tamls-ri" ~ "TAMLS +RI"
    )
  ) |>
  mutate(
    model_label = factor(
      model_label,
      levels = c(
        "PTM (lin)",
        "PTM (lin) +RI",
        "PTM (nonlin)",
        "PTM (nonlin) +RI",
        "Gaussian (lin)",
        "Gaussian (lin) +RI",
        "Gaussian (nonlin)",
        "Gaussian (nonlin) +RI",
        "BCTM",
        "BCTM +RI",
        "TAMLS",
        "TAMLS +RI",
        "DDPstar",
        "QGAM",
        "SBGP"
      )
    )
  )


# ..............................................................................
# ---- Models in separate panels ----
# ..............................................................................

pred0 <- pred |>
  filter(sex == 0 | sex < 0) |>
  filter(fold == -1) |>
  filter(!str_detect(model, "ptm") | str_detect(model, "77-id"))

sampled_quantile_models <- c(
  "kowal",
  "tamls-ri",
  "bctm",
  "bctm-ri",
  "gaussian-nonlinear",
  "gaussian-linear",
  "ptm-linear-44-id",
  "ptm-linear-77-id",
  "ptm-nonlinear-44-id",
  "ptm-nonlinear-77-id"
)


pred0 |>
  ggplot() +
  aes(age, predicted_quantile) +
  geom_point(
    aes(age, cholst),
    data = fh,
    alpha = 0.1,
    color = cbPalette[5],
    size = 1
  ) +
  geom_line(
    aes(group = factor(alpha)),
    data = pred0 |> filter(!(model %in% sampled_quantile_models))
  ) +
  geom_line(
    aes(group = factor(alpha)),
    linewidth = 0.3,
    data = pred0 |> filter(model %in% sampled_quantile_models)
  ) +
  geom_smooth(
    aes(group = factor(alpha)),
    se = F,
    data = pred0 |> filter(model %in% sampled_quantile_models),
    linewidth = 0.4,
    color = "black"
  ) +
  labs(
    x = "Age",
    y = "Cholesterol",
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
  facet_wrap(~model_label)

ggsave(path(out_dir, "qcurves_all.pdf"), width = 6.5, height = 6)
