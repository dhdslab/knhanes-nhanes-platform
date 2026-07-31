pkgs <- c("survey", "jsonlite", "rms", "MASS", "sandwich")
repos <- "https://cloud.r-project.org"

user_lib <- Sys.getenv("R_LIBS_USER")
if (!nzchar(user_lib)) {
  user_lib <- file.path(getwd(), ".Rlibs")
}

dir.create(user_lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(user_lib, .libPaths()))

cat("Using R library:\n")
cat(user_lib, "\n")

missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing) > 0) {
  install.packages(missing, lib = user_lib, repos = repos)
}

cat("R package check complete.\n")
