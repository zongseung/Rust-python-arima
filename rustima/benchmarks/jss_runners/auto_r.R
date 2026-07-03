# JSS §4.2 single-engine auto_arima runner — R::forecast::auto.arima.
# Called as a subprocess from exp_jss_auto_arima.py.
#
# Usage:
#   Rscript auto_r.R --out result.json
suppressPackageStartupMessages({
  library(jsonlite)
  library(forecast)
})

# ---- arg parsing -----------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(name) {
  i <- match(name, args)
  if (is.na(i) || i + 1 > length(args)) {
    stop(paste0("missing arg: ", name))
  }
  args[i + 1]
}
out_path <- get_arg("--out")

# ---- data ------------------------------------------------------------------
DATA_PATH <- "/Users/ijongseung/Documents/GitHub/arima-type/Rust-python-arima/power_demand_final.csv"
df <- read.csv(DATA_PATH, stringsAsFactors = FALSE)
df$일시 <- as.POSIXct(df$일시, tz = "UTC")
mask <- df$일시 >= as.POSIXct("2021-01-01", tz = "UTC") &
        df$일시 <  as.POSIXct("2024-01-01", tz = "UTC")
df <- df[mask, ]
stopifnot(nrow(df) == 26280)

y <- ts(df$"power.demand.MW.", frequency = 24)
xreg <- cbind(ta = df$ta, hm = df$hm)

cat(sprintf("[runner:auto:r_forecast] n=%d s=24\n", length(y)),
    file = stderr())

# ---- fit -------------------------------------------------------------------
# Search space mirrors auto_py.py: max_p/q=3, max_P/Q=1, max_d/D=1, stepwise.
t0 <- Sys.time()
fit <- forecast::auto.arima(
  y,
  xreg          = xreg,
  max.p         = 3,
  max.q         = 3,
  max.P         = 1,
  max.Q         = 1,
  max.d         = 1,
  max.D         = 1,
  seasonal      = TRUE,
  stepwise      = TRUE,
  ic            = "aic",
  approximation = FALSE,
  trace         = FALSE,
  allowdrift    = FALSE,
  allowmean     = FALSE
)
dt <- as.numeric(Sys.time() - t0, units = "secs")

# arimaorder() returns c(p, d, q, P, D, Q, period) when seasonal.
ord <- forecast::arimaorder(fit)
if (length(ord) == 7) {
  order  <- as.integer(ord[1:3])
  sorder <- as.integer(c(ord[4:6], ord[7]))
} else {
  order  <- as.integer(ord[1:3])
  sorder <- c(0L, 0L, 0L, 24L)
}

out <- list(
  engine          = "r_forecast",
  order           = as.list(order),
  seasonal_order  = as.list(sorder),
  aic             = fit$aic,
  n_models        = NULL,  # not exposed
  runtime_inner_s = dt,
  search = list(
    s = 24, max_p = 3, max_q = 3, max_P = 1, max_Q = 1,
    max_d = 1, max_D = 1, stepwise = TRUE, criterion = "aic"
  )
)

write_json(out, out_path, auto_unbox = TRUE, pretty = TRUE, digits = NA, null = "null")

cat(sprintf(
  "[runner:auto:r_forecast] OK order=(%d,%d,%d) sorder=(%d,%d,%d)[%d] aic=%.3f inner=%.2fs\n",
  order[1], order[2], order[3],
  sorder[1], sorder[2], sorder[3], sorder[4],
  fit$aic, dt),
  file = stderr())
