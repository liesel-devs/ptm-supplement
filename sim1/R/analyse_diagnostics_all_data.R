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
    "#999999", # grey
    "#0072B2", # blue
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

config <- read_csv("sim1/data/ptm_and_gaussian/config.csv")
diagnostics <- read_csv("sim1/data/ptm_and_gaussian/diagnostics.csv")
errors <- read_csv("sim1/data/ptm_and_gaussian/errors.csv")

errors <- errors |>
    left_join(
        config |> select(run, data, nobs),
        by = "run"
    ) |>
    filter(model != "gaussian") |>
    mutate(condition = model) |>
    mutate(
        model = factor(
            model,
            levels = c("ptm", "ptm-J30"),
            labels = c("ptm-15", "ptm-30")
        )
    ) |>
    mutate(
        data = factor(
            data,
            levels = c(
                "gaussian",
                "ptm",
                "mixture",
                "skewnorm",
                "u-shaped",
                "unif"
            ),
            labels = c(
                "Gaussian",
                "PTM",
                "Mixture",
                "Skewnorm",
                "U-shaped",
                "Unif"
            )
        )
    ) |>
    ungroup()

diagnostics <- diagnostics |>
    left_join(
        config |>
            select(run, model, data, nobs, seed),
        by = c("run")
    ) |>
    filter(model != "gaussian") |>
    mutate(condition = model) |>
    mutate(
        model = factor(
            model,
            levels = c("ptm", "ptm-J30"),
            labels = c("ptm-15", "ptm-30")
        )
    ) |>
    mutate(
        data = factor(
            data,
            levels = c(
                "gaussian",
                "ptm",
                "mixture",
                "skewnorm",
                "u-shaped",
                "unif"
            ),
            labels = c(
                "Gaussian",
                "PTM",
                "Mixture",
                "Skewnorm",
                "U-shaped",
                "Unif"
            )
        )
    )

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
        -c(data, nobs, seed, model, condition),
        names_to = "criterium",
        values_to = "value"
    ) |>
    # mutate(nobs = factor(nobs)) |>
    identity()

# ..............................................................................
# ---- Consistent colors ----
# ..............................................................................

data_colors <- cbPalette
names(data_colors) <- diagnostics |> pull(data) |> unique()

options(ggplot2.discrete.colour = data_colors)
options(ggplot2.discrete.fill = data_colors)

# ..............................................................................
# ---- Consistent shapes ----
# ..............................................................................

shapes <- c(1, 2, 3, 4, 5, 8, 13)
names(shapes) <- diagnostics |> pull(data) |> unique()

# --------------------------------------------------------------------------- #
# Effective Sample Size per Minute: Delta
# --------------------------------------------------------------------------- #

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
    filter(condition != "gaussian") |>
    mutate(model = as.character(model)) |>
    ggplot() +
    aes(nobs, value, color = data) +
    stat_summary(
        geom = "line",
        fun = "mean",
        aes(group = data, linetype = data)
    ) +
    stat_summary(
        geom = "point",
        fun = "mean",
        aes(group = data),
        color = "white",
        size = 2,
    ) +
    stat_summary(
        aes(shape = data),
        geom = "point",
        fun = "mean",
    ) +
    scale_shape_manual(values = shapes) +
    labs(x = TeX("$N_{train}$"), y = TeX("Min. (ESS / m) for $\\delta$")) +
    labs(title = TeX("Effective Samples Drawn per Minute for $\\delta$")) +
    facet_wrap(criterium ~ model, ncol = 4) +
    ylim(c(0, NA)))


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
    filter(condition != "gaussian") |>
    ggplot() +
    aes(nobs, value, color = data) +
    stat_summary(
        geom = "line",
        fun = "mean",
        aes(group = data, linetype = data)
    ) +
    stat_summary(
        geom = "point",
        fun = "mean",
        aes(group = data),
        color = "white",
        size = 2,
    ) +
    stat_summary(
        aes(shape = data),
        geom = "point",
        fun = "mean",
    ) +
    scale_shape_manual(values = shapes) +
    labs(x = TeX("$N_{train}$"), y = TeX("(ESS / m) for $\\tau^2$")) +
    facet_wrap(criterium ~ condition, ncol = 4) +
    ylim(c(0, NA)) +
    labs(title = TeX("Effective Samples Drawn per Minute for $\\tau^2$")))


(plt.rhat <- diag_long |>
    # filter(nobs == 25) |>
    filter(model != "gaussian") |>
    mutate(nobs = factor(nobs)) |>
    filter(
        str_detect(criterium, "rhat"),
        !str_detect(criterium, "median")
    ) |>
    filter(condition != "gaussian") |>
    drop_na() |>
    # mutate(
    #     criterium = case_when(
    #         str_detect(criterium, "delta") ~
    #             TeX("$\\delta$", bold = TRUE, output = "character"),
    #         str_detect(criterium, "tau") ~ TeX("$\\tau^2$", output = "character")
    #     )
    # ) |>
    # mutate(
    #     data = case_when(
    #         str_detect(data, "Gaussian") ~ "Gaussian~data",
    #         str_detect(data, "PTM") ~ "PTM~data",
    #     )
    # ) |>
    ggplot() +
    aes(nobs, value, color = data) +
    stat_summary(
        geom = "line",
        fun = "mean",
        aes(group = data, linetype = data)
    ) +
    stat_summary(
        geom = "point",
        fun = "mean",
        aes(group = data),
        color = "white",
        size = 2,
    ) +
    stat_summary(
        aes(shape = data),
        geom = "point",
        fun = "mean",
    ) +
    scale_shape_manual(values = shapes) +
    labs(x = TeX("$N_{train}$"), y = TeX("Avg. max. $\\hat{R}$")) +
    facet_wrap(criterium ~ condition, ncol = 6, labeller = label_parsed) +
    geom_hline(yintercept = c(1.01), color = "grey", linetype = "dotted") +
    # ylim(c(0, NA)) +
    labs(title = TeX("Convergence, measured by $\\hat{R}$")))


(plt.aprob <- diag_long |>
    # filter(nobs == 25) |>
    filter(model != "gaussian") |>
    mutate(nobs = factor(nobs)) |>
    filter(
        str_detect(criterium, "prob"),
        str_detect(criterium, "delta")
    ) |>
    # mutate(
    #     criterium = case_when(
    #         str_detect(criterium, "delta") ~
    #             TeX("$\\delta$", bold = TRUE, output = "character"),
    #         str_detect(criterium, "tau") ~ TeX("$\\tau^2$", output = "character")
    #     )
    # ) |>
    # mutate(
    #     data = case_when(
    #         str_detect(data, "Gaussian") ~ "Gaussian~data",
    #         str_detect(data, "PTM") ~ "PTM~data",
    #     )
    # ) |>
    ggplot() +
    aes(nobs, value, color = data) +
    stat_summary(
        geom = "line",
        fun = "mean",
        aes(group = data, linetype = data)
    ) +
    stat_summary(
        geom = "point",
        fun = "mean",
        aes(group = data),
        color = "white",
        size = 2,
    ) +
    stat_summary(
        aes(shape = data),
        geom = "point",
        fun = "mean",
    ) +
    scale_shape_manual(values = shapes) +
    labs(x = TeX("$N_{train}$"), y = TeX("Avg. acceptance probability")) +
    facet_wrap(~condition, ncol = 2) +
    # ylim(c(0, NA)) +
    labs(title = "Acceptance Probability"))


(plt.errors <- errors |>
    filter(phase == "posterior") |>
    filter(error_msg == "divergent transition") |>
    filter(str_detect(variable, "latent")) |>
    mutate(nobs = factor(nobs)) |>
    # mutate(
    #     variable = case_when(
    #         str_detect(variable, "latent") ~
    #             TeX("$\\delta$", bold = TRUE, output = "character"),
    #         str_detect(variable, "scale") ~ TeX("$\\tau^2$", output = "character")
    #     )
    # ) |>
    # mutate(
    #     data = case_when(
    #         str_detect(data, "Gaussian") ~ "Gaussian~data",
    #         str_detect(data, "PTM") ~ "PTM~data",
    #     )
    # ) |>
    ggplot() +
    aes(nobs, relative, color = data) +
    stat_summary(
        geom = "line",
        fun = "mean",
        aes(group = data, linetype = data)
    ) +
    stat_summary(
        geom = "point",
        fun = "mean",
        aes(group = data),
        color = "white",
        size = 2,
    ) +
    stat_summary(
        aes(shape = data),
        geom = "point",
        fun = "mean",
    ) +
    facet_wrap(~condition, ncol = 2) +
    scale_shape_manual(values = shapes) +
    labs(
        x = TeX("$N_{train}$"),
        y = TeX("Avg. share of divergent transitions")
    ) +
    theme(legend.position = "top") +
    labs(title = "Share of Divergent Transitions") +
    NULL)


# ..............................................................................
# ---- Combined plots ----
# ..............................................................................

plt <- plt.ess.delta /
    plt.ess.tau /
    plt.rhat +
    plot_layout(guides = "collect") &
    theme(text = element_text(size = 8)) &
    theme(legend.position = "bottom") &
    NULL


(plt + canvas(height = 200, width = 165, units = "mm")) |>
    save_ggplot(file = "sim1/out/diag-ess_rhat-all_data.pdf")

plt <- plt.aprob +
    plt.errors +
    plot_layout(guides = "collect") &
    theme(text = element_text(size = 8)) &
    theme(legend.position = "bottom") &
    NULL

(plt + canvas(height = 90, width = 165, units = "mm")) |>
    save_ggplot(file = "sim1/out/diag-aprob_errors-all_data.pdf")
