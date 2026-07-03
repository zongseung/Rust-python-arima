# R stats::arima(method="CSS-ML") on 2019 power demand
# Order: (3,0,3)(1,1,1)[24], xreg = [hm, ta]
suppressPackageStartupMessages({})

df <- read.csv("/Users/ijongseung/Documents/GitHub/arima-type/Rust-python-arima/power_demand_final.csv",
               stringsAsFactors = FALSE)
df$일시 <- as.POSIXct(df$일시, tz = "UTC")
mask <- df$일시 >= as.POSIXct("2019-01-01", tz = "UTC") &
        df$일시 <  as.POSIXct("2020-01-01", tz = "UTC")
df <- df[mask, ]
stopifnot(nrow(df) == 8760)

y <- df$"power.demand.MW."
xreg <- cbind(hm = df$hm, ta = df$ta)

t0 <- Sys.time()
fit <- stats::arima(
  y,
  order        = c(3, 0, 3),
  seasonal     = list(order = c(1, 1, 1), period = 24),
  xreg         = xreg,
  method       = "CSS-ML",
  include.mean = FALSE,
  optim.control = list(maxit = 500)
)
dt <- as.numeric(Sys.time() - t0, units = "secs")

cat("---R stats::arima result---\n")
cat(sprintf("loglik = %.6f\n", fit$loglik))
cat(sprintf("aic    = %.6f\n", fit$aic))
cat(sprintf("sigma2 = %.6f\n", fit$sigma2))
cat(sprintf("nobs   = %d\n", fit$nobs))
cat(sprintf("runtime_s = %.2f\n", dt))
cat("coef:\n")
print(round(coef(fit), 6))

# Save as one-row CSV with explicit names
out <- data.frame(
  method  = "r_stats_arima_css_ml",
  loglike = fit$loglik,
  aic     = fit$aic,
  sigma2  = fit$sigma2,
  n_obs   = fit$nobs,
  runtime_s = dt,
  stringsAsFactors = FALSE
)
co <- as.list(coef(fit))
for (nm in names(co)) out[[nm]] <- co[[nm]]
write.csv(
  out,
  "/Users/ijongseung/Documents/GitHub/arima-type/Rust-python-arima/rustima/docs/param_compare_2019_R.csv",
  row.names = FALSE
)
cat("\nwrote param_compare_2019_R.csv\n")
