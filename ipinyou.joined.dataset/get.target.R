#! /usr/bin/env Rscript
f <- file("stdin")
open(f)
while(length(line <- readLines(f, n = 1)) > 0) {
  tokens <- strsplit(line, ".", fixed = TRUE)[[1]]
  cat(grep("^\\d+$", tokens, value = TRUE))
  cat(".Rds\n")
}
