run_one <- function(
  data_seed,
  data_type,
  ntrain,
  ntest,
  nsave,
  out_path_dist,
  out_path_log
) {
  model <- "kowal"

  identifier <- paste0(
    model,
    "-",
    data_type,
    "-",
    sprintf("%03d", data_seed),
    "-",
    "n",
    ntrain,
    ".log"
  )

  logfile <- fs::path(out_path_log, identifier)
  log_appender(appender_file(logfile))
  log_info("Run started.")

  train <- load_data(data_seed, data_type, "train")
  test <- load_data(data_seed, data_type, "test")

  train <- train[1:ntrain, ]
  test <- test[1:ntest, ]

  # ..............................................................................
  # ---- Model ----
  # ..............................................................................

  log_info("Starting to fit model.")
  tic()
  m <- sbgp(
    y = train$y,
    locs = train[, c("x0", "x1", "x2", "x3")],
    X = cbind(1, train[, c("x0", "x1", "x2", "x3")]),
    nsave = nsave,
    approx_g = FALSE,
    locs_test = test[, c("x0", "x1", "x2", "x3")],
    X_test = cbind(1, test[, c("x0", "x1", "x2", "x3")])
  )
  timing <- toc(quiet = TRUE)
  log_info("Model fit complete.")

  # ..............................................................................
  # ---- Log Score and KLD on test data ----
  # ..............................................................................

  log_info("Starting log score and KLD computation.")
  dkde <- function(x, xnew) {
    density_values <- map_dbl(
      xnew,
      function(one_element_of_xnew) {
        density(
          x = x,
          from = one_element_of_xnew,
          to = one_element_of_xnew,
          n = 1,
          kernel = "gaussian",
          bw = "SJ"
        )$y
      }
    )

    density_values
  }

  i <- 1:ncol(m$post_ypred)
  # applying kernel density estimate to each set of conditional samples
  # evaluating at test observations
  pdf_estimated <- map_dbl(i, function(i) {
    dkde(xnew = test$y[i], x = m$post_ypred[, i])
  })
  log_pdf_estimated <- pmax(pdf_estimated, 1e-23) |> # safeguard against -Inf
    log()

  log_score_approx <- -sum(log_pdf_estimated)
  kld_approx = mean(test$log_pdf - log_pdf_estimated)

  log_info("Log score and KLD computation finished.")

  log_score_contributions <- scoringRules::logs_sample(test$y, t(m$post_ypred))
  kld_approx2 <- mean(test$log_pdf + log_score_contributions)
  log_score_approx2 <- sum(log_score_contributions)

  # ..............................................................................
  # ---- Mean absoluten deviation of CDF on test data ----
  # ..............................................................................
  log_info("Starting CDF computation.")
  cdf_estimated <- map_dbl(i, function(i) {
    skewsamp::pemp(q = test$y[i], sample = m$post_ypred[, i])
  })
  cdf_mad <- mean(abs(test$cdf - cdf_estimated))

  cdf_samples <- map_dfr(i, function(i) {
    cdf_evals <- skewsamp::pemp(
      q = m$post_ypred[, i],
      sample = m$post_ypred[, i]
    )
    tibble(
      n = i,
      cdf_sample = cdf_evals,
      cdf_true = test$cdf[i],
      draw = 1:length(cdf_evals)
    )
  })

  cdf_calibration <- cdf_samples |>
    group_by(n) |>
    summarise(
      q05 = quantile(cdf_sample, 0.05),
      q95 = quantile(cdf_sample, 0.95)
    ) |>
    mutate(cdf_true = test$cdf) |>
    mutate(in_ci = q05 <= cdf_true & cdf_true <= q95) |>
    summarise(coverage = mean(in_ci), width = mean(q95 - q05)) |>
    identity()
  log_info("CDF computation finished.")

  # ..............................................................................
  # ---- CRPS on test data ----
  # ..............................................................................
  log_info("Starting CRPS computation.")
  crps <- mean(crps_sample(test$y, t(m$post_ypred)))
  log_info("CRPS computation finished.")

  # ..............................................................................
  # ---- Summary of distribution analysis ----
  # ..............................................................................

  dist_summary <- tibble(
    kld = kld_approx2,
    log_score = log_score_approx2,
    kld_manual = kld_approx,
    log_score_manual = log_score_approx,
    cdf_mad,
    crps
  ) |>
    mutate(cdf_ci_coverage = cdf_calibration$coverage) |>
    mutate(cdf_ci_width = cdf_calibration$width)

  # ..............................................................................
  # ---- Location Function ----
  # ..............................................................................

  # visual plausibility checks
  # j <- 5

  # par(mfrow = c(1, 1))
  # plot(test$y, t(m$post_ypred)[,j])

  # par(mfrow = c(2, 2))
  # plot(test$x0, t(m$post_ypred)[,j])
  # plot(test$x1, t(m$post_ypred)[,j])
  # plot(test$x2, t(m$post_ypred)[,j])
  # plot(test$x3, t(m$post_ypred)[,j])

  # ..............................................................................
  # ---- Save run information ----
  # ..............................................................................

  tid <- format(Sys.time(), "%Y%m%d-%H%M%S")
  job <- fs::path(out_path_dist, "..", "..") |>
    fs::path_real() |>
    fs::path_file()

  dist_summary <- dist_summary |>
    mutate(
      data_type = data_type,
      data_seed = data_seed,
      model = model,
      ntrain = ntrain,
      ntest = ntest,
      fit_seconds = timing$toc - timing$tic,
      run = tid,
      job = job
    )

  # ..............................................................................
  # ---- Write results to disk ----
  # ..............................................................................
  log_info("Writing results.")

  identifier <- paste0(
    model,
    "-",
    data_type,
    "-",
    sprintf("%03d", data_seed),
    "-",
    "n",
    ntrain,
    ".csv"
  )

  fp_dist <- fs::path(out_path_dist, paste0("dist-", identifier))

  write_csv(dist_summary, fp_dist)
  log_info("Run finished.")
}
