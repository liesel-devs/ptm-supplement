library(tidyverse)
library(ggview)


sdpriorj30 <- read_csv("sdprior/data/sdprior_simJ30.csv") |>
  mutate(nparam = 30)
sdpriorj15 <- read_csv("sdprior/data/sdprior_simJ15.csv") |>
  mutate(nparam = 15)

sdprior <- bind_rows(sdpriorj15, sdpriorj30)

sdprior <- sdprior |>
  rename(psi = theta) |>
  mutate(psi_exponent = log2(psi))

# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #

sdprior$converged |> table()
sdprior$var_std |> hist()
sdprior$mean_std |> hist()

sdprior |> filter(!converged)
# I'll exclude the simulation runs in which standardization did not converge

sdprior <- sdprior |> filter(converged)

# --------------------------------------------------------------------------- #
# Total variation distance by psi and lambda
# --------------------------------------------------------------------------- #

# pattern looks very similar for all lambda
sdprior |>
  ggplot() +
  geom_boxplot(
    aes(x = factor(psi_exponent), y = tv),
    outlier.alpha = 0.1
  ) +
  facet_wrap(nparam ~ lambda, labeller = label_both) +
  coord_flip()


sdprior |>
  group_by(psi_exponent, lambda, nparam) |>
  reframe(alpha = quantile(tv, c(0.5))) |>
  ggplot() +
  geom_point(aes(psi_exponent, alpha, color = factor(lambda))) +
  geom_line(aes(psi_exponent, alpha, group = lambda)) +
  facet_wrap(~nparam, labeller = label_both) +
  NULL

# --------------------------------------------------------------------------- #
# Quantile curves for J=30
# --------------------------------------------------------------------------- #

qs <- c(0.01, 0.05, seq(0.1, 0.9, by = 0.1), 0.95, 0.99)
qs <- c(0.9, 0.95, 0.99)

sdprior_summary <- sdprior |>
  mutate(nparam = factor(nparam)) |>
  group_by(psi_exponent, nparam) |>
  reframe(quantile = qs, tv_quantile = quantile(tv, qs)) |>
  mutate(
    label = paste0("tilde(alpha)==", format(round(1 - quantile, 2), nsmall = 2))
  )

(sdprior.plot <- sdprior_summary |>
  mutate(quantile = factor(quantile)) |>
  filter(nparam == 30) |>
  ggplot() +
  aes(psi_exponent, tv_quantile) +
  geom_line(aes(color = quantile, group = quantile, linetype = nparam)) +
  geom_point(size = 2, color = "white") +
  geom_point(size = 1) +
  geom_text(
    aes(label = label),
    data = sdprior_summary |>
      filter(nparam == 30) |>
      filter(psi_exponent == max(psi_exponent)),
    nudge_y = -0.05,
    angle = 10,
    nudge_x = 0,
    parse = TRUE
  ) +
  scale_color_grey() +
  scale_x_continuous(
    breaks = seq(-10, 5, by = 1),
    labels = as.character(rbind(seq(-10, 5, by = 2), "")),
    # labels = seq(-10, 5, by = 2),
    limits = c(-10, 5),
    minor_breaks = NULL
  ) +
  theme_light() +
  theme(legend.position = "none") +
  theme(panel.grid.major.x = element_blank()) +
  labs(y = "Total Variation Distance", x = latex2exp::TeX("$\\log_2(psi)$")) +
  canvas(width = 100, height = 100, units = "mm") +
  # xlim(c(-10, 5)) +
  NULL)

save_ggplot(sdprior.plot, "sdprior/out/sdprior.pdf")

# --------------------------------------------------------------------------- #
# Compare J=15 and J=30
# --------------------------------------------------------------------------- #

sdprior_summary <- sdprior |>
  group_by(psi_exponent, nparam) |>
  reframe(quantile = qs, tv_quantile = quantile(tv, qs)) |>
  mutate(label = paste0("alpha==", format(round(1 - quantile, 2), nsmall = 2)))

sdprior_summary |>
  ggplot() +
  aes(psi_exponent, tv_quantile) +
  geom_line(aes(color = quantile, group = quantile)) +
  geom_point() +
  geom_text(
    aes(label = label),
    data = sdprior_summary |>
      filter(psi_exponent == max(psi_exponent)),
    nudge_y = -0.05,
    angle = 10,
    nudge_x = 0,
    parse = TRUE
  ) +
  scale_x_continuous(
    breaks = seq(-10, 5, by = 1),
    labels = as.character(rbind(seq(-10, 5, by = 2), "")),
    # labels = seq(-10, 5, by = 2),
    limits = c(-10, 5),
    minor_breaks = NULL
  ) +
  theme_light() +
  theme(legend.position = "none") +
  theme(panel.grid.major.x = element_blank()) +
  labs(
    y = "Total Variation Distance Quantile",
    x = latex2exp::TeX("$\\log_2(psi)$")
  ) +
  # canvas(width = 100, height = 100, units = "mm") +
  facet_wrap(~nparam, labeller = label_both) +
  NULL

(sdprior.j15.vs.30 <- sdprior_summary |>
  mutate(nparam = factor(nparam)) |>
  ggplot() +
  aes(psi_exponent, tv_quantile) +
  geom_line(aes(color = nparam, group = nparam, linetype = nparam)) +
  geom_point(size = 2, color = "white") +
  geom_point(size = 1, aes(shape = nparam)) +
  # geom_text(
  #     aes(label = label),
  #     data = sdprior_summary |>
  #         filter(psi_exponent == max(psi_exponent)),
  #     nudge_y = -0.05,
  #     angle = 10,
  #     nudge_x = 0,
  #     parse = TRUE
  # ) +
  scale_x_continuous(
    breaks = seq(-10, 5, by = 1),
    labels = as.character(rbind(seq(-10, 5, by = 2), "")),
    # labels = seq(-10, 5, by = 2),
    limits = c(-10, 5),
    minor_breaks = NULL
  ) +
  labs(color = "J-1", linetype = "J-1", shape = "J-1") +
  theme_light() +
  # theme(legend.position = "none") +
  theme(panel.grid.major.x = element_blank()) +
  labs(y = "Total Variation Distance", x = latex2exp::TeX("$\\log_2(psi)$")) +
  # canvas(width = 100, height = 100, units = "mm") +
  facet_wrap(~quantile, labeller = label_both) +
  NULL)

(sdprior.j15.vs.30 + canvas(width = 165, height = 70, units = "mm")) |>
  save_ggplot(file = "sdprior/out/sdprior1530.pdf")
