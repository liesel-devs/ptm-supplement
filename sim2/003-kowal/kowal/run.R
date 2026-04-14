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
    X_test = cbind(1, test[, c("x0", "x1", "x2", "x3")]),
    y_test = test$y
  )
  timing <- toc(quiet = TRUE)
  log_info("Model fit complete.")

  apply(m$post_LogPDFytrain, 2, var) |> hist()

  var(m$post_LogPDFytrain) |> dim()

  cdf_mad <- mean(abs(test$cdf - colMeans(m$post_CDFy)))

  log_lik_contributions <- apply(m$post_LogPDFy, 2, function(x) {
    m = max(x)
    m + log(mean(exp(x - m)))
  })

  kld <- mean(test$log_pdf - log_lik_contributions)

  waic <- loo::waic(m$post_LogPDFytrain)$estimates[3]

  # ..............................................................................
  # ---- CRPS on test data ----
  # ..............................................................................
  log_info("Starting CRPS computation.")
  crps <- mean(crps_sample(test$y, t(m$post_ypred)))
  log_info("CRPS computation finished.")

  # ..............................................................................
  # ---- CDF calibration ----
  # ..............................................................................

  cdf_samples <- as.data.frame(t(m$post_CDFy)) |>
    as_tibble() |>
    mutate(n = row_number()) |>
    mutate(cdf_true = test$cdf) |>
    pivot_longer(
      starts_with("V"),
      values_to = "cdf_sample",
      names_to = "draw",
      names_prefix = "V"
    )

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

  # ..............................................................................
  # ---- Summary of distribution analysis ----
  # ..............................................................................

  dist_summary <- tibble(
    kld,
    waic,
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
