suppressMessages({
  library(jsonlite)
  library(MASS)
})

series <- read.csv("trend_series.csv", check.names = FALSE)
cfg <- fromJSON("trend_config.json")
future <- as.integer(cfg$future)

apc <- NA_real_
lo <- NA_real_
hi <- NA_real_
if (nrow(series) >= 2 && all(series$rate > 0, na.rm = TRUE)) {
  fit <- lm(log(rate) ~ year, data = series)
  b <- coef(summary(fit))["year", "Estimate"]
  se <- coef(summary(fit))["year", "Std. Error"]
  apc <- (exp(b) - 1) * 100
  lo <- (exp(b - 1.96 * se) - 1) * 100
  hi <- (exp(b + 1.96 * se) - 1) * 100
}

write.csv(
  data.frame(APC_pct = round(apc, 2), lo95 = round(lo, 2), hi95 = round(hi, 2)),
  "trend_apc.csv",
  row.names = FALSE
)

nb_out <- data.frame(year = future, forecast = NA_real_, lo95 = NA_real_, hi95 = NA_real_)
if (nrow(series) >= 2 && all(series$N > 0, na.rm = TRUE)) {
  fit <- tryCatch(
    glm.nb(count ~ year + offset(log(N)), data = series),
    error = function(e) glm(count ~ year + offset(log(N)), data = series, family = poisson())
  )
  nd <- data.frame(year = future, N = round(mean(series$N, na.rm = TRUE)))
  pred <- predict(fit, newdata = nd, type = "link", se.fit = TRUE)
  rate <- exp(pred$fit) * 100
  nb_out <- data.frame(
    year = future,
    forecast = round(rate, 2),
    lo95 = round(exp(pred$fit - 1.96 * pred$se.fit) * 100, 2),
    hi95 = round(exp(pred$fit + 1.96 * pred$se.fit) * 100, 2)
  )
}

write.csv(nb_out, "trend_nb.csv", row.names = FALSE)
cat("trend.R OK\n")
