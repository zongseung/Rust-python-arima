#!/usr/bin/env Rscript
# Compare statsmodels vs R's stats::arima on the same SARIMAX fit
# Goal: is sm's LL=-69,292 the true MLE (R reaches same) or sm-specific?

suppressPackageStartupMessages({
  library(stats)
})

# load 1y power demand
data <- read.csv("../power_demand_final.csv", stringsAsFactors = FALSE)
data$일시 <- as.POSIXct(data$일시)
mask <- data$일시 >= as.POSIXct("2019-01-01") & data$일시 < as.POSIXct("2020-01-01")
data <- data[mask, ]

y <- as.numeric(data[["power.demand.MW."]])
# impute NaN with mean (matches Python)
if (any(is.na(y))) y[is.na(y)] <- mean(y, na.rm = TRUE)
ex <- as.matrix(data[, c("ta", "hm")])
for (j in 1:ncol(ex)) {
  col <- ex[, j]
  if (any(is.na(col))) ex[is.na(col), j] <- mean(col, na.rm = TRUE)
}

cat(sprintf("Data: n=%d, exog shape=(%d,%d)\n", length(y), nrow(ex), ncol(ex)))

ORDER    <- c(3, 0, 3)
SEASONAL <- list(order = c(1, 1, 1), period = 24)

cat("\n--- R stats::arima(method='CSS-ML') ---\n")
t0 <- Sys.time()
fit_cssml <- tryCatch(
  arima(y, order = ORDER, seasonal = SEASONAL, xreg = ex,
        method = "CSS-ML", optim.method = "BFGS",
        optim.control = list(maxit = 500)),
  error = function(e) { cat("CSS-ML failed:", conditionMessage(e), "\n"); NULL }
)
t_cssml <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
if (!is.null(fit_cssml)) {
  ll  <- fit_cssml$loglik
  k   <- length(fit_cssml$coef) + 1   # +1 for sigma2
  aic <- -2 * ll + 2 * k
  cat(sprintf("  LL = %.4f   AIC = %.4f   time = %.1fs\n", ll, aic, t_cssml))
  cat("  coefs:\n")
  for (nm in names(fit_cssml$coef)) {
    cat(sprintf("    %-12s = %+.6f\n", nm, fit_cssml$coef[[nm]]))
  }
  cat(sprintf("    %-12s = %+.4f\n", "sigma^2", fit_cssml$sigma2))
}

cat("\n--- R stats::arima(method='ML') ---\n")
t0 <- Sys.time()
fit_ml <- tryCatch(
  arima(y, order = ORDER, seasonal = SEASONAL, xreg = ex,
        method = "ML", optim.method = "BFGS",
        optim.control = list(maxit = 500)),
  error = function(e) { cat("ML failed:", conditionMessage(e), "\n"); NULL }
)
t_ml <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
if (!is.null(fit_ml)) {
  ll  <- fit_ml$loglik
  k   <- length(fit_ml$coef) + 1
  aic <- -2 * ll + 2 * k
  cat(sprintf("  LL = %.4f   AIC = %.4f   time = %.1fs\n", ll, aic, t_ml))
  cat("  coefs:\n")
  for (nm in names(fit_ml$coef)) {
    cat(sprintf("    %-12s = %+.6f\n", nm, fit_ml$coef[[nm]]))
  }
  cat(sprintf("    %-12s = %+.4f\n", "sigma^2", fit_ml$sigma2))
}

cat("\n--- Reference from earlier runs ---\n")
cat("  statsmodels SARIMAX: LL = -69292.0788   AIC = 138606.16   ta=-268.05  hm=-4.74\n")
cat("  rustima             : LL = -71216.34    AIC = 142454.68   ta=  +2.02  hm=+28.23\n")
