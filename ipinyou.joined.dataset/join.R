if (interactive()) argv <- "20131024.Rds" else argv <- commandArgs(TRUE)
log.path <- sprintf("%s.log", argv)

suppressPackageStartupMessages({
  library(magrittr)
  library(data.table)
  library(dplyr)
  library(dtplyr)
})

source("../colClass.R")

update.col.names <- function(x, name) {
  setnames(x, colnames(x), name)
}

parse_timestamp <- function(str) {
  stopifnot(nchar(str) == 17)
  base <- strptime(substring(str, 1, 14), "%Y%m%d%H%M%S")
  ms <- as.numeric(substring(str, 15, 17)) * 1e-3
  base + ms
}

target.date <- as.Date(argv, "%Y%m%d.Rds")
options(warn = 2)
src <- sprintf("find ../ipinyou.contest.dataset/* | grep %s", format(target.date, "%Y%m%d")) %>%
  system(intern = TRUE)
src <- sapply(c("bid", "imp", "clk", "conv"), grep, x = src, value = TRUE)
if (!is.character(src)) {
  src <- src[sapply(src, nchar) != 0] %>%
    unlist()
}
capture.output(print(src)) %>%
  cat(file = log.path, append = TRUE)
file.exists(src) %>% stopifnot()

season.src <- strsplit(src, split = "/", fixed = TRUE) %>%
  sapply("[", 3)
stopifnot(season.src == season.src[1])
season <- season.src[1]

get.dt <- function(name, colClassFunction) {
  col.class <- colClassFunction(season)
  src.path <- src[name]
  if (!is.na(src.path)) {
    sprintf("bzcat %s > %s", src[name], tmp.path <- tempfile(".txt")) %>%
      system(intern = TRUE) %>%
      invisible()
    fread(
      tmp.path, 
      sep = "\t", colClasses = as.vector(col.class), 
      header = F, showProgress = interactive(), data.table = TRUE) %>%
      update.col.names(names(col.class))
  } else {
    lapply(col.class, get, envir = globalenv()) %>% 
      lapply(function(x) x()) %>%
      do.call(what = data.table)
  }
}

# read bid
bid <- get.dt("bid", bidColClass)

# read imp
imp <- get.dt("imp", impColClass)

# read clk
clk <- get.dt("clk", impColClass)

# read conv
conv <- get.dt("conv", impColClass)

join.imp.clk.conv <- function(imp, clk, conv) {
  ts <- parse_timestamp(imp$Timestamp)
  imp2 <- dtplyr::tbl_dt(imp) %>% dplyr::mutate(imp_t = ts)
  ts <- parse_timestamp(clk$Timestamp)
  clk2 <- dtplyr::tbl_dt(clk) %>% dplyr::mutate(clk_t = ts, is_click = TRUE) %>%
    dplyr::select(BidID, clk_t, is_click)
  result <- dplyr::left_join(imp2, clk2, by = "BidID") %>%
    arrange(imp_t)
  ts <- parse_timestamp(conv$Timestamp)
  conv2 <- dtplyr::tbl_dt(conv) %>% dplyr::mutate(conv_t = ts, is_conversion = TRUE) %>%
    dplyr::select(BidID, conv_t, is_conversion)
  dplyr::left_join(result, conv2, by = "BidID") %>%
    arrange(imp_t)
}

imp.clk.conv <- join.imp.clk.conv(imp, clk, conv) %>%
  data.table()

join.bid.imp.clk.conv <- function(bid, imp.clk.conv) {
  ts <- parse_timestamp(bid$Timestamp)
  bid2 <- dtplyr::tbl_dt(bid) %>%
    mutate(weekday = format(ts, "%w"), hour = format(ts, "%H"), bid_t = ts) %>%
    dplyr::select(BidID, BiddingPrice, IP, Region, City, AdExchange, Domain, URL, AdSlotId,
                  AdSlotWidth, AdSlotHeight, AdSlotVisibility, AdSlotFormat, CreativeID, weekday, hour, bid_t)
  imp.clk.conv2 <- dtplyr::tbl_dt(imp.clk.conv) %>%
    dplyr::select(BidID, PayingPrice, adid, usertag, imp_t, is_click, clk_t, is_conversion, conv_t)
  left_join(bid2, imp.clk.conv2, by = "BidID") %>%
    arrange(bid_t)
}

result <- join.bid.imp.clk.conv(bid, imp.clk.conv)
saveRDS(result, argv.tmp <- sprintf("%s.tmp", argv))
file.rename(argv.tmp, argv)

