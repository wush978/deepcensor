Rcpp::sourceCpp("parseYOYI.cpp")
# read data
x <- readLines(gzfile("../yoyi/train.yzx.txt.gz"))
# construct splitting indexes
i <- rep(1:10, floor(length(x) / 10))
length(i) <- length(x)
i <- sort(i)
for(.i in 1:10) {
  .x <- x[i == .i]
  path <- sprintf("train.obj.%02d.Rds", .i)
  cat(sprintf("Processing %s...\n", path))
  saveRDS(parseYOYI(.x), path)
}
