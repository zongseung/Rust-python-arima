#!/usr/bin/env Rscript
# Single-window forecast::auto.arima worker for the e12 scaling-exog table.
# Mirrors bench_sarima_worker_exog.py exactly:
#   - cumulative window anchored at 2021-01-01 (1yr/2yr/3yr -> end 2022/2023/2024)
#   - exog = [ta, hm], NaN -> column mean
#   - constrained search (3,3,1,1,1,1), stepwise, ic="aic"
# Run ONE horizon per process so /usr/bin/time -l captures per-horizon peak RSS.
#
# usage: Rscript bench_auto_R_exog_worker.R <years>
# stdout last line: __RESULT__{json}

suppressPackageStartupMessages(library(forecast))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) {
  stop("usage: bench_auto_R_exog_worker.R <years>")
}
years <- as.integer(args[[1]])

DATA_PATH <- "/Users/ijongseung/Documents/GitHub/arima-type/Rust-python-arima/power_demand_final.csv"

df <- read.csv(DATA_PATH, stringsAsFactors = FALSE, check.names = FALSE)
df[["일시"]] <- as.POSIXct(df[["일시"]], tz = "UTC")

end_year <- 2021L + years
mask <- df[["일시"]] >= as.POSIXct("2021-01-01", tz = "UTC") &
        df[["일시"]] <  as.POSIXct(sprintf("%d-01-01", end_year), tz = "UTC")
sub <- df[mask, ]

y  <- as.numeric(sub[["power demand(MW)"]])
ta <- as.numeric(sub[["ta"]])
hm <- as.numeric(sub[["hm"]])
fill_mean <- function(v) { v[is.na(v)] <- mean(v, na.rm = TRUE); v }
y  <- fill_mean(y)
ta <- fill_mean(ta)
hm <- fill_mean(hm)
xreg <- cbind(ta = ta, hm = hm)

n_obs <- length(y)
cat(sprintf("years=%d n_obs=%d\n", years, n_obs))
flush.console()

yts <- ts(y, frequency = 24)

t0 <- Sys.time()
fit <- tryCatch(
  auto.arima(
    yts, xreg = xreg, seasonal = TRUE,
    max.p = 3, max.q = 3, max.P = 1, max.Q = 1, max.d = 1, max.D = 1,
    stepwise = TRUE, approximation = TRUE, ic = "aic",
    allowmean = FALSE, allowdrift = TRUE, trace = FALSE
  ),
  error = function(e) { cat(sprintf("ERROR: %s\n", conditionMessage(e))); NULL }
)
dt <- as.numeric(Sys.time() - t0, units = "secs")

if (is.null(fit)) {
  cat(sprintf("__RESULT__{\"years\":%d,\"n_obs\":%d,\"time_s\":%.3f,\"status\":\"failed\"}\n",
              years, n_obs, dt))
  quit(status = 0)
}

ord <- arimaorder(fit)
order_str    <- sprintf("(%d, %d, %d)", ord["p"], ord["d"], ord["q"])
seasonal_str <- if ("P" %in% names(ord)) {
  sprintf("(%d, %d, %d, %d)", ord["P"], ord["D"], ord["Q"], ord["Frequency"])
} else { "(0, 0, 0, 0)" }
aic_val <- AIC(fit)

cat(sprintf("selected %s%s  AIC=%.4f  time=%.2fs\n",
            order_str, seasonal_str, aic_val, dt))
cat(sprintf("__RESULT__{\"years\":%d,\"n_obs\":%d,\"time_s\":%.3f,\"order\":\"%s\",\"seasonal\":\"%s\",\"aic\":%.4f,\"status\":\"ok\"}\n",
            years, n_obs, dt, order_str, seasonal_str, aic_val))
