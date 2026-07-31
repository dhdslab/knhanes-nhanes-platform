suppressMessages({
  library(survey)
  library(jsonlite)
})

options(survey.lonely.psu = "adjust")

cfg <- fromJSON("epi_config.json")
d <- read.csv("epi_analytic.csv", check.names = FALSE)

y <- cfg$outcome
x <- cfg$exposure
cov <- cfg$cov[cfg$cov %in% names(d)]
subgroups <- cfg$subgroups[cfg$subgroups %in% names(d)]

needed <- c(y, x, cov, "psu", "kstrata", "wt_pool")
d0 <- d[complete.cases(d[, needed, drop = FALSE]), , drop = FALSE]
des <- svydesign(ids = ~psu, strata = ~kstrata, weights = ~wt_pool, data = d0, nest = TRUE)

fit_logit <- function(terms) {
  rhs <- paste(terms, collapse = " + ")
  svyglm(as.formula(paste(y, "~", rhs)), design = des, family = quasibinomial())
}

terms_to_table <- function(fit) {
  s <- summary(fit)$coefficients
  out <- data.frame()
  for (term in setdiff(rownames(s), "(Intercept)")) {
    b <- s[term, 1]
    se <- s[term, 2]
    p <- s[term, ncol(s)]
    out <- rbind(
      out,
      data.frame(
        term = term,
        OR = exp(b),
        lo = exp(b - 1.96 * se),
        hi = exp(b + 1.96 * se),
        p = p
      )
    )
  }
  out
}

crude_fit <- fit_logit(c(x))
adj_fit <- fit_logit(c(x, cov))

write.csv(terms_to_table(crude_fit), "epi_crude.csv", row.names = FALSE, fileEncoding = "UTF-8")
write.csv(terms_to_table(adj_fit), "epi_adj.csv", row.names = FALSE, fileEncoding = "UTF-8")

risk <- tryCatch(as.numeric(coef(svymean(as.formula(paste0("~", y)), des, na.rm = TRUE))[1]) * 100, error = function(e) NA_real_)
write.csv(data.frame(weighted_risk_pct = risk), "epi_risk.csv", row.names = FALSE)

or_for_x <- function(fit, term) {
  s <- summary(fit)$coefficients
  if (!term %in% rownames(s)) return(c(NA_real_, NA_real_, NA_real_))
  b <- s[term, 1]
  se <- s[term, 2]
  c(exp(b), exp(b - 1.96 * se), exp(b + 1.96 * se))
}

subres <- data.frame()
for (sg in subgroups) {
  vals <- sort(unique(d0[[sg]][!is.na(d0[[sg]])]))
  pint <- NA_real_
  if (length(vals) > 1) {
    int_fit <- tryCatch(
      svyglm(as.formula(paste(y, "~", x, "*", sg, "+", paste(cov, collapse = " + "))), design = des, family = quasibinomial()),
      error = function(e) NULL
    )
    if (!is.null(int_fit)) {
      ss <- summary(int_fit)$coefficients
      interaction_rows <- grep(paste0("^", x, ":"), rownames(ss))
      if (!length(interaction_rows)) interaction_rows <- grep(paste0(":", x, "$"), rownames(ss))
      if (length(interaction_rows)) pint <- ss[interaction_rows[1], ncol(ss)]
    }
  }
  for (lv in vals) {
    dsub <- d0[d0[[sg]] == lv, , drop = FALSE]
    if (nrow(dsub) < 20 || length(unique(dsub[[y]])) < 2) next
    dess <- svydesign(ids = ~psu, strata = ~kstrata, weights = ~wt_pool, data = dsub, nest = TRUE)
    rhs_terms <- c(x, cov)
    fit <- tryCatch(
      svyglm(as.formula(paste(y, "~", paste(rhs_terms, collapse = " + "))), design = dess, family = quasibinomial()),
      error = function(e) NULL
    )
    if (is.null(fit)) next
    z <- or_for_x(fit, x)
    subres <- rbind(subres, data.frame(subgroup = sg, level = lv, OR = z[1], lo = z[2], hi = z[3], p_int = pint))
  }
}

write.csv(subres, "epi_sub.csv", row.names = FALSE, fileEncoding = "UTF-8")
cat("epi.R OK\n")
