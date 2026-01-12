library(tidyverse)
library(latex2exp)
library(patchwork)
library(fs)

# Get the directory of the currently active file
current_file <- rstudioapi::getActiveDocumentContext()$path
current_dir <- dirname(current_file)

data_dir <- path(current_dir, "..", "data")
out_dir <- path(current_dir, "..", "out")
dir_create(out_dir)

# Please download the data first via the instructions in
# simulation/data/README.md
df <- read_csv(path(
  current_dir,
  "..",
  "..",
  "/data/gaussian/train/gaussian-001.csv"
))

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

df <- df |>
  select(
    (starts_with("x") | starts_with("f")) &
      (!ends_with("mean") &
        !ends_with("scale_mean") &
        !ends_with("scale_scale") &
        !ends_with("loc_scale"))
  )


plt.loc <- df |>
  ggplot() +
  geom_line(aes(x0, f0_loc, color = "x1"), linewidth = 1.5) +
  geom_line(aes(x1, f1_loc, color = "x2"), linewidth = 1.5) +
  geom_line(aes(x2, f2_loc, color = "x3"), linewidth = 1.5) +
  geom_line(aes(x3, f3_loc, color = "x4"), linewidth = 1.5) +
  labs(x = "x", y = "s(x)", title = TeX("Covariate functions on $\\mu$")) +
  theme_light() +
  guides(size = "none") +
  theme(
    legend.title = element_blank(),
    # panel.grid = element_blank(),
    # axis.title.y = element_blank()
  ) +
  NULL

plt.scale <- df |>
  ggplot() +
  geom_line(aes(x0, f0_scale, color = "x1"), linewidth = 1.5) +
  geom_line(aes(x1, f1_scale, color = "x2"), linewidth = 1.5) +
  geom_line(aes(x2, f2_scale, color = "x3"), linewidth = 1.5) +
  geom_line(aes(x3, f3_scale, color = "x4"), linewidth = 1.5) +
  labs(
    x = "x",
    y = "s(x)",
    title = TeX("Covariate functions on $\\ln(\\sigma)$")
  ) +
  theme_light() +
  guides(size = "none") +
  theme(
    legend.title = element_blank(),
    # panel.grid = element_blank(),
    # axis.title.y = element_blank()
  ) +
  NULL


plt.loc +
  plt.scale +
  patchwork::plot_layout(guides = "collect") &
  theme(legend.position = "bottom")

ggsave(path(out_dir, "covariates.pdf"), width = 6.5, height = 4)
