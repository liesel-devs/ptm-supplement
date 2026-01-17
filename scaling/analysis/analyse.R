library(tidyverse)

out_path <- fs::path("scaling/analysis/out")

dist <- read_csv("scaling/analysis/data/dist.csv")

dist |>
    select(model, ntrain, fit_seconds) |>
    arrange(ntrain) |>
    pivot_wider(names_from = c("ntrain"), values_from = "fit_seconds")


dist |>
    filter(ntrain == 2000) |>
    mutate(
        log_score = case_when(
            # this is necessary because, for performance reasons, we evaluated the log score
            # for ddpstar using only 1000 test observations (compared to 5000 for the
            # other models)
            model == "ddpstar" ~ log_score * 5,
            TRUE ~ log_score
        )
    ) |>
    select(model, ntrain, fit_seconds, waic, log_score, crps, kld)


dist |>
    filter(ntrain == 10000) |>
    mutate(
        log_score = case_when(
            # this is necessary because, for performance reasons, we evaluated the log score
            # for ddpstar using only 1000 test observations (compared to 5000 for the
            # other models)
            model == "ddpstar" ~ log_score * 5,
            TRUE ~ log_score
        )
    ) |>
    select(model, ntrain, fit_seconds, waic, log_score, crps, kld)

dist |>
    filter(ntrain == 20000) |>
    mutate(
        log_score = case_when(
            # this is necessary because, for performance reasons, we evaluated the log score
            # for ddpstar using only 1000 test observations (compared to 5000 for the
            # other models)
            model == "ddpstar" ~ log_score * 5,
            TRUE ~ log_score
        )
    ) |>
    select(model, ntrain, fit_seconds, waic, log_score, crps, kld)


dist |>
    filter(ntrain > 1000) |>
    select(model, ntrain, fit_seconds) |>
    pivot_wider(names_from = model, values_from = fit_seconds)


dist |>
    filter(ntrain > 1000) |>
    select(model, ntrain, fit_seconds) |>
    pivot_wider(names_from = ntrain, values_from = fit_seconds) |>
    arrange(`20000`)

dist |>
    filter(model == "ptm-iwls-nuts") |>
    arrange(ntrain)


dist |>
    filter(ntrain > 1000) |>
    select(model, ntrain, fit_seconds) |>
    ggplot() +
    aes(fct_reorder(model, fit_seconds, .desc = TRUE), fit_seconds) +
    geom_bar(
        stat = "identity",
        aes(fill = model),
        position = position_dodge2()
    ) +
    facet_wrap(~ntrain, scales = "free_y", labeller = label_both, ncol = 1) +
    coord_flip()

library(ggview)

dist |>
    filter(ntrain > 1000) |>
    select(model, ntrain, fit_seconds) |>
    mutate(ntrain = factor(ntrain)) |>
    ggplot() +
    aes(fct_reorder(model, fit_seconds, .desc = TRUE), fit_seconds) +
    geom_bar(
        stat = "identity",
        aes(fill = ntrain),
        # position = position_dodge2()
    ) +
    facet_wrap(~ntrain, labeller = label_both, ncol = 3, scales = "free_x") +
    coord_flip() +
    canvas(8, 3)

p <- dist |>
    filter(ntrain > 1000) |>
    select(model, ntrain, fit_seconds) |>
    # mutate(ntrain = factor(ntrain)) |>
    mutate(
        model = case_when(
            model == "gaussian-iwls" ~ "Gaussian",
            model == "tamls" ~ "TAMLS",
            model == "kowal" ~ "SBGP",
            model == "ddpstar" ~ "DDPStar",
            model == "bctm-locshift" ~ "BCTM-LS",
            model == "bctm-te" ~ "BCTM-TE",
            model == "ptm-iwls-nuts" ~ "PTM",
            model == "qgam" ~ "QGAM",
            TRUE ~ model
        )
    ) |>
    mutate(
        model = factor(
            model,
            levels = c(
                "Gaussian",
                "SBGP",
                "DDPStar",
                "PTM",
                "QGAM",
                "TAMLS",
                "BCTM-LS",
                "BCTM-TE"
            ) |>
                rev()
        )
    ) |>
    ggplot() +
    aes(model, fit_seconds) +
    geom_bar(
        stat = "identity",
        aes(fill = model),
        # position = position_dodge2()
    ) +
    # facet_wrap(~ntrain, labeller = label_both, ncol = 1) +
    facet_wrap(
        ~ntrain,
        labeller = label_bquote(N ~ "=" ~ .(ntrain)),
        ncol = 1
    ) +
    coord_flip() +
    theme_minimal() +
    theme(legend.position = "none") +
    theme(axis.title.y = element_blank()) +
    theme(axis.text.y = element_text(size = 8)) +
    labs(y = "Runtime in Seconds") +
    khroma::scale_fill_okabeito(black_position = "last") +
    canvas(9, 4)

ggview::save_ggplot(p, fs::path(out_path, "scaling.pdf"))


dist |>
    filter(ntrain > 1000) |>
    select(model, ntrain, fit_seconds) |>
    mutate(ntrain = factor(ntrain)) |>
    mutate(model = fct_reorder(model, fit_seconds, .desc = TRUE)) |>
    ggplot() +
    aes(ntrain, fit_seconds) +
    geom_bar(
        stat = "identity",
        aes(fill = model),
        position = position_dodge2()
    ) +
    # facet_wrap(~ntrain, scales = "free_y", labeller = label_both, ncol = 1) +
    coord_flip()


dist |>
    filter(ntrain > 1000) |>
    select(model, ntrain, fit_seconds) |>
    mutate(ntrain = factor(ntrain)) |>
    mutate(model = fct_reorder(model, fit_seconds)) |>
    ggplot() +
    aes(model, fit_seconds) +
    geom_bar(
        stat = "identity",
        aes(fill = model),
        position = position_dodge2()
    ) +
    facet_wrap(~ntrain, labeller = label_both) +
    theme(axis.text.x = element_text(angle = 40, hjust = 1)) +
    # coord_flip() +
    NULL

diagnostics <- read_csv("scaling/analysis/data/diagnostics.csv")

diagnostics |>
    filter(model == "ptm-iwls-nuts") |>
    group_by(ntrain) |>
    summarise(
        min_ess_bulk = min(ess_bulk_min),
        avg_min_ess_bulk = mean(ess_bulk_min),
        min_ess_tail = min(ess_tail_min),
        avg_min_ess_tail = mean(ess_tail_min)
    )
