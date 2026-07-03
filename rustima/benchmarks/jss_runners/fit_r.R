# JSS §4 single-engine fit runner — R::stats::arima (CSS-ML).
# Called as a subprocess from exp_jss_fixed_order.py.
#
# Usage:
#   Rscript fit_r.R --order 2,0,0 --seasonal-order 1,0,1,24 --out result.json
suppressPackageStartupMessages({
  library(jsonlite)
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

order  <- as.integer(strsplit(get_arg("--order"),          ",")[[1]])
sorder <- as.integer(strsplit(get_arg("--seasonal-order"), ",")[[1]])
out_path <- get_arg("--out")

stopifnot(length(order) == 3, length(sorder) == 4)

# ---- data ------------------------------------------------------------------
DATA_PATH <- "/Users/ijongseung/Documents/GitHub/arima-type/Rust-python-arima/power_demand_final.csv"
df <- read.csv(DATA_PATH, stringsAsFactors = FALSE)
df$일시 <- as.POSIXct(df$일시, tz = "UTC")
mask <- df$일시 >= as.POSIXct("2021-01-01", tz = "UTC") &
        df$일시 <  as.POSIXct("2024-01-01", tz = "UTC")
df <- df[mask, ]
stopifnot(nrow(df) == 26280)

y <- df$"power.demand.MW."
xreg <- cbind(ta = df$ta, hm = df$hm)

cat(sprintf("[runner:r_stats_arima] n=%d order=(%d,%d,%d) sorder=(%d,%d,%d)[%d]\n",
            length(y), order[1], order[2], order[3],
            sorder[1], sorder[2], sorder[3], sorder[4]),
    file = stderr())

# ---- fit -------------------------------------------------------------------
t0 <- Sys.time()
fit <- stats::arima(
  y,
  order        = order,
  seasonal     = list(order = sorder[1:3], period = sorder[4]),
  xreg         = xreg,
  method       = "CSS-ML",
  include.mean = FALSE,
  optim.control = list(maxit = 500)
)
dt <- as.numeric(Sys.time() - t0, units = "secs")

# R::arima counts sigma² in npar (k = length(coef) + 1).
# Match that convention so AIC/BIC are internally consistent.
n   <- fit$nobs
k   <- length(fit$coef) + 1
aic <- -2 * fit$loglik + 2 * k
bic <- -2 * fit$loglik + log(n) * k

co <- as.list(coef(fit))

out <- list(
  engine          = "r_stats_arima",
  order           = as.list(order),
  seasonal_order  = as.list(sorder),
  trend           = "n",
  params          = co,
  loglike         = fit$loglik,
  aic             = aic,
  bic             = bic,
  scale           = fit$sigma2,
  n_obs           = n,
  runtime_inner_s = dt
)

write_json(out, out_path, auto_unbox = TRUE, pretty = TRUE, digits = NA)

cat(sprintf("[runner:r_stats_arima] OK ll=%.3f aic=%.3f bic=%.3f inner=%.2fs\n",
            fit$loglik, aic, bic, dt),
    file = stderr())
