suppressMessages({
  library(survey)
  library(jsonlite)
})

options(survey.lonely.psu = "adjust")

cfg <- fromJSON("engine_config.json")
d <- read.csv("engine_analytic.csv", check.names = FALSE)

des <- svydesign(ids = ~psu, strata = ~kstrata, weights = ~wt_pool, data = d, nest = TRUE)

fmt_num <- function(x) {
  if (is.na(x)) return("")
  sprintf("%.2f", x)
}

svy_mean_se <- function(var, design) {
  f <- as.formula(paste0("~", var))
  out <- tryCatch(svymean(f, design, na.rm = TRUE), error = function(e) NULL)
  if (is.null(out)) return(c(NA_real_, NA_real_))
  c(as.numeric(coef(out)[1]), as.numeric(SE(out)[1]))
}

fmt_cont <- function(var, design) {
  z <- svy_mean_se(var, design)
  if (is.na(z[1])) return("")
  paste0(fmt_num(z[1]), " (", fmt_num(z[2]), ")")
}

fmt_bin <- function(var, design) {
  z <- svy_mean_se(var, design)
  if (is.na(z[1])) return("")
  paste0(sprintf("%.1f", 100 * z[1]), "%")
}

label_for <- function(var) {
  labs <- cfg$labels
  if (!is.null(labs[[var]])) return(labs[[var]])
  var
}

rows <- list()
add_row <- function(variable, all_value, g0 = NULL, g1 = NULL) {
  row <- data.frame(Variable = variable, All = all_value, check.names = FALSE)
  if (!is.null(g0)) row[["0"]] <- g0
  if (!is.null(g1)) row[["1"]] <- g1
  rows[[length(rows) + 1]] <<- row
}

group <- cfg$group
has_group <- !is.null(group) && nzchar(group) && group %in% names(d)

for (v in cfg$cont) {
  if (!v %in% names(d)) next
  if (has_group) {
    add_row(
      label_for(v),
      fmt_cont(v, des),
      fmt_cont(v, subset(des, get(group) == 0)),
      fmt_cont(v, subset(des, get(group) == 1))
    )
  } else {
    add_row(label_for(v), fmt_cont(v, des))
  }
}

for (v in cfg$bin) {
  if (!v %in% names(d)) next
  if (has_group) {
    add_row(
      label_for(v),
      fmt_bin(v, des),
      fmt_bin(v, subset(des, get(group) == 0)),
      fmt_bin(v, subset(des, get(group) == 1))
    )
  } else {
    add_row(label_for(v), fmt_bin(v, des))
  }
}

table1 <- if (length(rows)) do.call(rbind, rows) else data.frame(Variable = character(), All = character())
write.csv(table1, "engine_table1.csv", row.names = FALSE, fileEncoding = "UTF-8")

coef_row <- function(fit, term, measure) {
  s <- summary(fit)$coefficients
  if (!term %in% rownames(s)) return(NULL)
  b <- s[term, 1]
  se <- s[term, 2]
  p <- s[term, ncol(s)]
  if (measure == "OR") {
    data.frame(est = exp(b), ci_low = exp(b - 1.96 * se), ci_high = exp(b + 1.96 * se), p = p)
  } else {
    data.frame(est = b, ci_low = b - 1.96 * se, ci_high = b + 1.96 * se, p = p)
  }
}

results <- data.frame()
pairs <- cfg$pairs
if (!is.null(pairs) && nrow(pairs) > 0) {
  for (i in seq_len(nrow(pairs))) {
    y <- pairs$y[i]
    x <- pairs$x[i]
    otype <- pairs$otype[i]
    if (!all(c(y, x) %in% names(d))) next

    cov <- cfg$cov[cfg$cov %in% names(d)]
    terms <- c(x, cov)
    needed <- c(y, terms, "psu", "kstrata", "wt_pool")
    dd <- d[complete.cases(d[, needed, drop = FALSE]), , drop = FALSE]
    if (nrow(dd) < 20) next

    des2 <- svydesign(ids = ~psu, strata = ~kstrata, weights = ~wt_pool, data = dd, nest = TRUE)
    rhs <- paste(terms, collapse = " + ")
    form <- as.formula(paste(y, "~", rhs))
    measure <- ifelse(otype == "b", "OR", "beta")
    fit <- tryCatch({
      if (otype == "b") svyglm(form, design = des2, family = quasibinomial()) else svyglm(form, design = des2)
    }, error = function(e) NULL)
    if (is.null(fit)) next

    cr <- coef_row(fit, x, measure)
    if (is.null(cr)) next
    results <- rbind(
      results,
      data.frame(
        exposure = x,
        outcome = y,
        measure = measure,
        est = cr$est,
        ci_low = cr$ci_low,
        ci_high = cr$ci_high,
        p = cr$p
      )
    )
  }
}

write.csv(results, "engine_results.csv", row.names = FALSE, fileEncoding = "UTF-8")
cat("engine.R OK\n")
