# ..............................................................................
# ---- Libraries ----
# ..............................................................................
library(tidyverse)
library(LaplacesDemon)
library(ggview)


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
  theme_minimal()
)

theme_update(
  legend.position = "top",
  legend.title = element_blank(),
  legend.key.size = unit(1, 'cm'),
  legend.key.height = unit(0.5, 'cm'),
  legend.key.width = unit(2, 'cm'),
  panel.grid = element_blank()
)

options(ggplot2.discrete.colour = cbPalette)
options(ggplot2.discrete.fill = cbPalette)

# ..............................................................................
# ---- Data import ----
# ..............................................................................
ridge15 <- read_csv(
  "sim1/data/showcase_ridge-vs-rw-skewnorm/ridge_grid_J15_summary.csv"
) |>
  mutate(nparam = 15) |>
  mutate(prior = "ridge")
ridge30 <- read_csv(
  "sim1/data/showcase_ridge-vs-rw-skewnorm/ridge_grid_J30_summary.csv"
) |>
  mutate(nparam = 30) |>
  mutate(prior = "ridge")
rw15 <- read_csv(
  "sim1/data/showcase_ridge-vs-rw-skewnorm/rw_grid_J15_summary.csv"
) |>
  mutate(nparam = 15) |>
  mutate(prior = "rw")
rw30 <- read_csv(
  "sim1/data/showcase_ridge-vs-rw-skewnorm/rw_grid_J30_summary.csv"
) |>
  mutate(nparam = 30) |>
  mutate(prior = "rw")

df <- bind_rows(ridge15, ridge30, rw15, rw30)

# ..............................................................................
# ---- True input distribution ----
# ..............................................................................

mean_of_sn <- function(xi, omega, alpha) {
  delta <- alpha / (sqrt(1 + alpha^2))

  xi + omega * delta * sqrt(2 / pi)
}

var_of_sn <- function(omega, alpha) {
  delta <- alpha / (sqrt(1 + alpha^2))

  omega^2 * (1 - (2 * delta^2) / pi)
}


std_dsn <- function(x, xi, omega, alpha) {
  m <- mean_of_sn(xi, omega, alpha)
  s2 <- var_of_sn(omega, alpha)
  s <- sqrt(s2)

  sn::dsn(s * x + m, xi, omega, alpha) * s
}

xi <- 0
omega <- 1
alpha <- 5
df <- df |>
  mutate(
    pdf_true = std_dsn(y, xi, omega, alpha)
  )

# ..............................................................................
# ---- Plot ----
# ..............................................................................

(plt <- df |>
  filter(y > -3) |>
  mutate(
    prior = case_when(
      prior == "ridge" ~ "Ridge Prior",
      prior == "rw" ~ "Random Walk Prior"
    ) |>
      factor() |>
      fct_rev()
  ) |>
  mutate(
    nparam = case_when(
      nparam == 15 ~ "J-1 = 15",
      nparam == 30 ~ "J-1 = 30"
    )
  ) |>
  ggplot() +
  geom_ribbon(
    aes(y, ymin = `q_0.05`, ymax = `q_0.95`, fill = prior),
    alpha = 0.2
  ) +
  geom_line(aes(y, pdf_true), linewidth = 0.3) +
  geom_line(aes(y, pdf, color = prior), linewidth = 0.7) +
  facet_wrap(~nparam, ncol = 2) +
  labs(y = "PDF", x = "R") +
  theme(
    legend.position = "right",
    axis.text = element_blank(),
    axis.title = element_blank()
  ))

(plt + canvas(height = 45, width = 165, units = "mm")) |>
  save_ggplot(file = "sim1/out/ridge_vs_rw.pdf")
