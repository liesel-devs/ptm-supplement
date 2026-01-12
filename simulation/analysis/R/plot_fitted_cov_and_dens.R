library(tidyverse)
library(fs)
library(latex2exp)
library(patchwork)

# Get the directory of the currently active file
current_file <- rstudioapi::getActiveDocumentContext()$path
current_dir <- dirname(current_file)

data_dir <- path(current_dir, "..", "data")
out_dir <- path(current_dir, "..", "out")
dir_create(out_dir)


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

loc <- read_csv(path(data_dir, "loc_summary.csv"))
scale <- read_csv(path(data_dir, "scale_summary.csv"))
r_dens <- read_csv(path(data_dir, "r_dens_summary.csv"))

# the _samples datasets are quite large (>500mb in total), so they are available
# as a .zip file: `simulation/analysis/data/samples_datasets.zip`.
# Please unpack this .zip file and place the files in the
# `simulation/analysis/data` directory
# to import these datasets.
loc_samples <- read_csv(path(data_dir, "loc_samples_summary.csv"))
scale_samples <- read_csv(path(data_dir, "scale_samples_summary.csv"))
r_dens_samples <- read_csv(path(data_dir, "r_dens_summary_samples.csv"))

# ..............................................................................
# ---- Data cleaning ----
# ..............................................................................

su <- bind_rows(loc, scale) |>
  mutate(xnum = xnum + 1) |>
  mutate(Index = xnum) |>
  rename(Predictor = parameter)

sa <- bind_rows(loc_samples, scale_samples) |>
  mutate(xnum = xnum + 1) |>
  mutate(Index = xnum) |>
  rename(Predictor = parameter)

su |> distinct(data_seed, ntrain)

seed <- 6
n <- 500
su.loc <- su |>
  filter(data_type == "ptm") |>
  filter(data_seed == seed) |>
  filter(ntrain == n) |>
  filter(Predictor == "loc") |>
  ggplot() +
  geom_ribbon(
    aes(x, ymin = low, ymax = high),
    fill = cbPalette[2],
    alpha = 0.2
  ) +
  geom_line(
    data = sa |>
      filter(data_type == "ptm", data_seed == seed, Predictor == "loc") |>
      filter(ntrain == n),
    aes(x, fx_hat, group = sample),
    color = cbPalette[2],
    alpha = 0.1
  ) +
  geom_line(aes(x, fx_hat), color = cbPalette[5], linewidth = 1) +
  geom_line(aes(x, fx_true), linetype = "dotted") +
  facet_grid(
    Predictor ~ Index,
    labeller = labeller(
      Index = c(
        "1" = "Loc, s1",
        "2" = "Loc, s2",
        "3" = "Loc, s3",
        "4" = "Loc, s4"
      )
    )
  ) +
  theme_light() +
  theme(
    legend.title = element_blank(),
    panel.grid = element_blank(),
    strip.background = element_rect(fill = "white"),
    strip.text = element_text(color = "black"),
    strip.text.y = element_blank(),
    axis.text.x = element_blank(),
    axis.title.x = element_blank()
  ) +
  labs(y = TeX("$s(x)$")) +
  NULL

su.scale <- su |>
  filter(data_type == "ptm") |>
  filter(data_seed == seed) |>
  filter(ntrain == n) |>
  filter(Predictor == "scale") |>
  ggplot() +
  geom_ribbon(
    aes(x, ymin = low, ymax = high),
    fill = cbPalette[2],
    alpha = 0.2
  ) +
  geom_line(
    data = sa |>
      filter(data_type == "ptm", data_seed == seed, Predictor == "scale") |>
      filter(ntrain == n),
    aes(x, fx_hat, group = sample),
    color = cbPalette[2],
    alpha = 0.1
  ) +
  geom_line(aes(x, fx_hat), color = cbPalette[5], linewidth = 1) +
  geom_line(aes(x, fx_true), linetype = "dotted") +
  facet_grid(
    Predictor ~ Index,
    labeller = labeller(
      Index = c(
        "1" = "Scale, s1",
        "2" = "Scale, s2",
        "3" = "Scale, s3",
        "4" = "Scale, s4"
      )
    )
  ) +
  theme_light() +
  theme(
    legend.title = element_blank(),
    panel.grid = element_blank(),
    strip.background = element_rect(fill = "white"),
    strip.text = element_text(color = "black"),
    strip.text.y = element_blank(),
    # axis.title.y = element_blank()
  ) +
  labs(y = TeX("$s(x)$")) +
  NULL

su.loc / su.scale
ggsave(path(out_dir, "fitted_cov.pdf"), width = 6.5, height = 4)


r_dens |>
  distinct(data_type, data_seed, ntrain) |>
  filter(ntrain == 500, data_type == "ptm")

ptm_seeds <- c(13, 20)
input_data_path <- path("sim/sim-conditional/data")
data_type <- "ptm"
data_files <- path(input_data_path, data_type, "train") |> dir_ls()
sel <- map_lgl(data_files, function(x) {
  any(str_detect(x, sprintf("%03d", ptm_seeds)))
})
true_ptm <- map_dfr(which(sel), function(i) {
  read_csv(data_files[i]) |>
    mutate(data_seed = i) |>
    mutate(data_type = data_type)
})


data_type <- "mixture"
data_files <- path(input_data_path, data_type, "train") |> dir_ls()
sel <- map_lgl(data_files, function(x) {
  any(str_detect(x, sprintf("%03d", c(1, 2))))
})
true_mixture <- map_dfr(which(sel), function(i) {
  read_csv(data_files[i]) |>
    mutate(data_seed = i) |>
    mutate(data_type = data_type)
})

data_type <- "skewnorm"
data_files <- path(input_data_path, data_type, "train") |> dir_ls()
sel <- map_lgl(data_files, function(x) {
  any(str_detect(x, sprintf("%03d", c(1, 2))))
})
true_skew <- map_dfr(which(sel), function(i) {
  read_csv(data_files[i]) |>
    mutate(data_seed = i) |>
    mutate(data_type = data_type)
})

mixture_seed <- 1
skew_seed <- 1


n <- 500

true_pdf <- bind_rows(true_ptm, true_mixture, true_skew) |>
  filter(
    (data_type == "ptm" & data_seed %in% ptm_seeds) |
      (data_type == "mixture" & data_seed == mixture_seed) |
      (data_type == "skewnorm" & data_seed == skew_seed)
  ) |>
  mutate(data = paste0(data_type, "-", data_seed)) |>
  mutate(
    data = factor(
      data,
      levels = c("mixture-1", "skewnorm-1", "ptm-13", "ptm-20"),
      labels = c("Mixture-1", "Skewnorm-1", "PTM-13", "PTM-20")
    )
  )


r_dens_plot <- r_dens |>
  filter(ntrain == n) |>
  filter(
    (data_type == "ptm" & data_seed %in% ptm_seeds) |
      (data_type == "mixture" & data_seed == mixture_seed) |
      (data_type == "skewnorm" & data_seed == skew_seed)
  ) |>
  mutate(data = paste0(data_type, "-", data_seed)) |>
  mutate(
    data = factor(
      data,
      levels = c("mixture-1", "skewnorm-1", "ptm-13", "ptm-20"),
      labels = c("Mixture-1", "Skewnorm-1", "PTM-13", "PTM-20")
    )
  )


r_dens_samples_plot <- r_dens_samples |>
  filter(ntrain == n) |>
  filter(
    (data_type == "ptm" & data_seed %in% ptm_seeds) |
      (data_type == "mixture" & data_seed == mixture_seed) |
      (data_type == "skewnorm" & data_seed == skew_seed)
  ) |>
  mutate(data = paste0(data_type, "-", data_seed)) |>
  mutate(
    data = factor(
      data,
      levels = c("mixture-1", "skewnorm-1", "ptm-13", "ptm-20"),
      labels = c("Mixture-1", "Skewnorm-1", "PTM-13", "PTM-20")
    )
  )

r_dens_plot |>
  ggplot() +
  geom_ribbon(
    aes(r, ymin = low, ymax = high),
    fill = cbPalette[2],
    alpha = 0.2
  ) +
  geom_line(
    data = r_dens_samples_plot,
    aes(r, pdf, group = sample),
    color = cbPalette[2],
    alpha = 0.1
  ) +
  geom_line(aes(r, pdf_hat), color = cbPalette[5], linewidth = 0.8) +
  geom_line(
    aes(r, pdf_r),
    linetype = "dotted",
    data = true_pdf
  ) +
  facet_wrap(
    ~data,
    ncol = 4,
    # scales = "free_y",
    # labeller = label_both
  ) +
  theme_light() +
  theme(
    legend.title = element_blank(),
    panel.grid = element_blank(),
    strip.background = element_rect(fill = "white"),
    strip.text = element_text(color = "black")
    # axis.title.y = element_blank()
  ) +
  xlim(c(-5, 5)) +
  labs(y = TeX("$f_R(r)$"), x = "r") +
  theme(axis.text.x = element_text(size = 7)) +
  NULL

ggsave(path(out_dir, "fitted_dens.pdf"), width = 6.5, height = 2)
