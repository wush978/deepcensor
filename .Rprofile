local({
  .path <- file.path(".lib", sprintf("%s.%s", getRversion()$major, getRversion()$minor))
  .path <- normalizePath(.path)
  if (!dir.exists(.path)) {
    dir.create(.path, recursive = TRUE)
  }
  .libPaths(new = .path)
})
options(repos = c(CRAN = "https://cloud.r-project.org/"))
