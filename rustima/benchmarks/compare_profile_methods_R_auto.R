#!/usr/bin/env Rscript
# forecast::auto.arima on Korean power demand for 1yr/2yr/3yr horizons.
# xreg = [hm, ta], seasonal period = 24, stepwise = TRUE (Hyndman-Khandakar).
# Writes per-horizon summary to docs/auto_arima_R.csv and a verbose log to
# docs/auto_arima_R.log via stdout redirection from the caller.

suppressPackageStartupMessages({
  library(forecast)
})

DATA_PATH <- "/Users/ijongseung/Documents/GitHub/arima-type/Rust-python-arima/power_demand_final.csv"
OUT_CSV   <- "/Users/ijongseung/Documents/GitHub/arima-type/Rust-python-arima/rustima/docs/auto_arima_R.csv"

cat(sprintf("R %s, forecast %s\n", R.version.string,
            as.character(packageVersion("forecast"))))
cat(sprintf("started: %s\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S")))

df <- read.csv(DATA_PATH, stringsAsFactors = FALSE)
df$일시 <- as.POSIXct(df$일시, tz = "UTC")
cat(sprintf("loaded %d rows from %s\n", nrow(df), DATA_PATH))

horizons <- list(
  list(label = "1yr", start = "2019-01-01", end = "2020-01-01"),
  list(label = "2yr", start = "2019-01-01", end = "2021-01-01"),
  list(label = "3yr", start = "2019-01-01", end = "2022-01-01")
)

results <- list()
csv_rows <- list()

for (h in horizons) {
  cat("\n================================================================\n")
  cat(sprintf("[%s] %s .. %s\n", h$label, h$start, h$end))
  cat("================================================================\n")

  mask <- df$일시 >= as.POSIXct(h$start, tz = "UTC") &
          df$일시 <  as.POSIXct(h$end,   tz = "UTC")
  sub  <- df[mask, ]
  y    <- sub$"power.demand.MW."
  xreg <- cbind(hm = sub$hm, ta = sub$ta)

  cat(sprintf("n_obs = %d, n_exog = %d\n", length(y), ncol(xreg)))

  # ts with frequency=24 makes auto.arima treat it as seasonal hourly.
  yts <- ts(y, frequency = 24)

  flush.console()

  t0 <- Sys.time()
  fit <- tryCatch({
    auto.arima(
      yts,
      xreg          = xreg,
      seasonal      = TRUE,
      stepwise      = TRUE,
      approximation = TRUE,
      trace         = TRUE,
      ic            = "aic",
      allowmean     = FALSE,
      allowdrift    = TRUE
    )
  }, error = function(e) {
    cat(sprintf("ERROR for %s: %s\n", h$label, conditionMessage(e)))
    NULL
  })
  dt <- as.numeric(Sys.time() - t0, units = "secs")

  if (is.null(fit)) {
    csv_rows[[length(csv_rows) + 1L]] <- data.frame(
      horizon = h$label, n_obs = length(y), order = NA, seasonal = NA,
      loglik = NA, aic = NA, aicc = NA, bic = NA, sigma2 = NA,
      runtime_s = dt, status = "failed", stringsAsFactors = FALSE
    )
    next
  }

  ord  <- arimaorder(fit)  # named vector: p,d,q (and P,D,Q,s if seasonal)
  ord_nonseas <- sprintf("(%d,%d,%d)", ord["p"], ord["d"], ord["q"])
  ord_seas    <- if ("P" %in% names(ord)) {
    sprintf("(%d,%d,%d)[%d]", ord["P"], ord["D"], ord["Q"], ord["Frequency"])
  } else { "(0,0,0)[0]" }

  cat(sprintf("\n[%s] selected: %s%s\n", h$label, ord_nonseas, ord_seas))
  cat(sprintf("[%s] loglik=%.4f  AIC=%.4f  AICc=%.4f  BIC=%.4f  sigma2=%.6f\n",
              h$label, fit$loglik, AIC(fit), fit$aicc, BIC(fit), fit$sigma2))
  cat(sprintf("[%s] runtime = %.2fs\n", h$label, dt))
  cat("[coef]\n"); print(round(coef(fit), 6))

  csv_rows[[length(csv_rows) + 1L]] <- data.frame(
    horizon = h$label, n_obs = length(y),
    order = ord_nonseas, seasonal = ord_seas,
    loglik = fit$loglik, aic = AIC(fit), aicc = fit$aicc, bic = BIC(fit),
    sigma2 = fit$sigma2, runtime_s = dt, status = "ok",
    stringsAsFactors = FALSE
  )

  # Write incrementally so we keep partial results if the next horizon dies.
  out <- do.call(rbind, csv_rows)
  dir.create(dirname(OUT_CSV), showWarnings = FALSE, recursive = TRUE)
  write.csv(out, OUT_CSV, row.names = FALSE)
  cat(sprintf("[%s] wrote partial %s\n", h$label, OUT_CSV))
}

cat("\n================================================================\n")
cat("done.\n")
cat(sprintf("finished: %s\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S")))
if (length(csv_rows) > 0L) {
  out <- do.call(rbind, csv_rows)
  print(out)
}
