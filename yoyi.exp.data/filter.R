# filter feature whose positive instance is lower than given threshould
argv <- commandArgs(TRUE)
threshold <- as.numeric(argv[1])
stopifnot(!is.na(threshold))
library(methods)
library(Matrix)
library(magrittr)
library(parallel)
path <- dir(".", "train.obj.*.Rds")
cl <- makeCluster(length(path))
parLapply(cl, seq_along(path), function(.i) {
  assign(".i", .i, envir = globalenv())
}) %>%
  invisible()
clusterEvalQ(cl, {
  X <- readRDS(sprintf("train.obj.%02d.Rds", .i))
  NULL
}) %>%
  invisible()
nrow.total <- clusterEvalQ(cl, nrow(X$X)) %>%
  unlist() %>%
  sum()
colsums.total <- clusterEvalQ(cl, {
  library(methods)
  library(Matrix)
  colSums(X$X)
})
ncol.total <- sapply(colsums.total, length) %>%
  max()

colsums.total %<>% lapply(function(x) {
  length(x) <- ncol.total
  x[is.na(x)] <- 0
  x
}) %>%
  Reduce(f = `+`)

# alive columns
i <- which(colsums.total >= nrow.total * threshold)
clusterExport(cl, c("i", "ncol.total"))
clusterEvalQ(cl, {
  .X <- X$X
  .X@Dim[2] <- as.integer(ncol.total)
  . <- .X[,i]
  X$X <- .
  rm(.X, .)
  gc()
  NULL
}) %>% invisible()

# save to hdf5

dir.create(file.path(argv))
clusterExport(cl, "argv")
clusterEvalQ(cl, {
  dir.create(file.path(argv, .i))
}) %>% invisible()
clusterEvalQ(cl, {
  library(hdf5r)
  fname <- file.path(argv, .i, "exp.data.h5")
  if (file.exists(fname)) file.remove(fname)
  file.h5 <- H5File$new(fname, mode = "w")
  # create lv1
  lv1.grp <- file.h5$create_group("lv1")
  .X <- as(X$X, "RsparseMatrix")
  .g <- lv1.grp$create_group("hashed")
  .g[[".value"]] <- .X@j
  .g[["class"]] <- "compressed.list"
  .g[["element.size"]] <- diff(.X@p)
  # create response
  responses.grp <- file.h5$create_group("responses")
  .g <- responses.grp$create_group("clk")
  .g[[".value"]] <- X$click
  .g[["class"]] <- class(X$click)
  .g <- responses.grp$create_group("wp")
  .g[[".value"]] <- X$price
  .g[["class"]] <- class(X$price)
  file.h5$close_all()
  NULL
}) %>% invisible()


stopCluster(cl)

