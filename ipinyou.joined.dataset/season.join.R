library(hdf5r)
library(magrittr)
library(data.table)
library(dplyr)
library(dtplyr)
library(multidplyr)
library(parallel)
library(foreach)
if (getwd() %>% basename() != "ipinyou.joined.dataset") setwd("ipinyou.joined.dataset")
if (interactive()) argv <- "201310" else argv <- commandArgs(TRUE)
flist <- dir("../ipinyou.contest.dataset", pattern = sprintf("bid.%s.*txt.bz2", argv), full.names = TRUE, recursive = TRUE)
stopifnot(length(flist) > 0)
cl <- makeCluster(length(flist))
parLapply(cl, seq_along(cl), function(i) {
  assign(".id", i, envir = globalenv())
}) %>% 
  invisible()
clusterExport(cl, "flist")
# library
clusterEvalQ(cl, {
  library(magrittr)
  library(data.table)
  library(dplyr)
  library(dtplyr)
  library(multidplyr)
  library(text2vec)
  library(foreach)
}) %>%
  invisible()
# loading dependencies
clusterEvalQ(cl, {
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
  
  target.date <- 
    regexpr("(\\d{8})", flist[.id]) %>%
    regmatches(x = flist[.id]) %>%
    unlist() %>%
    as.Date("%Y%m%d")
  src <- sprintf("find ../ipinyou.contest.dataset/* | grep %s", format(target.date, "%Y%m%d")) %>%
    system(intern = TRUE)
  src <- sapply(c("bid", "imp", "clk", "conv"), grep, x = src, value = TRUE, simplify = FALSE)
  season.src <- strsplit(src$bid, split = "/", fixed = TRUE) %>%
    sapply("[", 3)
  # stopifnot(season.src == season.src[1])
  season <- season.src[1]
  get.dt <- function(name, colClassFunction) {
    if (length(src[[name]]) == 0) return(NULL)
    if (!file.exists(src[[name]])) return(NULL)
    col.class <- colClassFunction(season)
    sprintf("bzcat %s > %s", src[name], tmp.path <- tempfile(".txt", "/dev/shm")) %>%
      system(intern = TRUE) %>%
      invisible()
    on.exit(unlink(tmp.path))
    fread(
      tmp.path, 
      sep = "\t", colClasses = as.vector(col.class), 
      header = F, showProgress = interactive(), data.table = TRUE) %>%
      update.col.names(names(col.class))
  }
}) %>%
  invisible()
# loading datasets
bid.classes <- clusterEvalQ(cl, {
  bid <- get.dt("bid", bidColClass)
  class(bid)
}) %>%
  lapply(`==`, c("data.table", "data.frame")) %>%
  sapply(all) %>%
  all() %>%
  stopifnot()
clusterEvalQ(cl, {
  is.na(bid$BiddingPrice) %>% any()
}) %>%
  unlist() %>%
  `!`() %>%
  all() %>%
  stopifnot()
imp.classes <- clusterEvalQ(cl, {
  imp <- get.dt("imp", impColClass)
  if (is.null(imp)) {
    imp <- structure(
      list(
        BidID = character(0), Timestamp = character(0),
        LogType = integer(0), iPinyouID = character(0), UserAgent = character(0),
        IP = character(0), Region = integer(0), City = integer(0),
        AdExchange = integer(0), Domain = character(0), URL = character(0),
        AnonymousURLId = character(0), AdSlotId = character(0), AdSlotWidth = character(0),
        AdSlotHeight = character(0), AdSlotVisibility = character(0),
        AdSlotFormat = character(0), AdSlotFloorPrice = numeric(0),
        CreativeID = character(0), BiddingPrice = numeric(0), PayingPrice = numeric(0),
        KeyPageURL = character(0), adid = character(0), usertag = character(0)), 
      .Names = c(
        "BidID",
        "Timestamp", "LogType", "iPinyouID", "UserAgent", "IP", "Region",
        "City", "AdExchange", "Domain", "URL", "AnonymousURLId", "AdSlotId",
        "AdSlotWidth", "AdSlotHeight", "AdSlotVisibility", "AdSlotFormat",
        "AdSlotFloorPrice", "CreativeID", "BiddingPrice", "PayingPrice",
        "KeyPageURL", "adid", "usertag"), 
      class = "data.frame", row.names = integer(0)) %>%
      data.table()
  }
  class(imp)
}) %>%
  lapply(`==`, c("data.table", "data.frame")) %>%
  sapply(all) %>%
  all() %>%
  stopifnot()
clusterEvalQ(cl, {
  is.na(imp$PayingPrice) %>% any()
}) %>%
  unlist() %>%
  `!`() %>%
  all() %>%
  stopifnot()
clusterEvalQ(cl, {
  clk <- get.dt("clk", impColClass)
  if (is.null(clk)) {
    clk <- structure(
      list(
        BidID = character(0), Timestamp = character(0), 
        LogType = integer(0), iPinyouID = character(0), UserAgent = character(0), 
        IP = character(0), Region = integer(0), City = integer(0), 
        AdExchange = integer(0), Domain = character(0), URL = character(0), 
        AnonymousURLId = character(0), AdSlotId = character(0), AdSlotWidth = character(0), 
        AdSlotHeight = character(0), AdSlotVisibility = character(0), 
        AdSlotFormat = character(0), AdSlotFloorPrice = numeric(0), 
        CreativeID = character(0), BiddingPrice = numeric(0), PayingPrice = numeric(0), 
        KeyPageURL = character(0), adid = character(0), usertag = character(0)), 
      .Names = c(
        "BidID", 
        "Timestamp", "LogType", "iPinyouID", "UserAgent", "IP", "Region", 
        "City", "AdExchange", "Domain", "URL", "AnonymousURLId", "AdSlotId", 
        "AdSlotWidth", "AdSlotHeight", "AdSlotVisibility", "AdSlotFormat", 
        "AdSlotFloorPrice", "CreativeID", "BiddingPrice", "PayingPrice", 
        "KeyPageURL", "adid", "usertag"), 
      class = "data.frame", row.names = integer(0)) %>%
      data.table()
  }
  class(clk)
}) %>%
  lapply(`==`, c("data.table", "data.frame")) %>%
  sapply(all) %>%
  all() %>%
  stopifnot()
clusterEvalQ(cl, {
  conv <- get.dt("conv", impColClass)
  if (is.null(conv)) {
    conv <- structure(
      list(
        BidID = character(0), Timestamp = character(0), 
        LogType = integer(0), iPinyouID = character(0), UserAgent = character(0), 
        IP = character(0), Region = integer(0), City = integer(0), 
        AdExchange = integer(0), Domain = character(0), URL = character(0), 
        AnonymousURLId = character(0), AdSlotId = character(0), AdSlotWidth = character(0), 
        AdSlotHeight = character(0), AdSlotVisibility = character(0), 
        AdSlotFormat = character(0), AdSlotFloorPrice = numeric(0), 
        CreativeID = character(0), BiddingPrice = numeric(0), PayingPrice = numeric(0), 
        KeyPageURL = character(0), adid = character(0), usertag = character(0)), 
      .Names = c("BidID", 
                 "Timestamp", "LogType", "iPinyouID", "UserAgent", "IP", "Region", 
                 "City", "AdExchange", "Domain", "URL", "AnonymousURLId", "AdSlotId", 
                 "AdSlotWidth", "AdSlotHeight", "AdSlotVisibility", "AdSlotFormat", 
                 "AdSlotFloorPrice", "CreativeID", "BiddingPrice", "PayingPrice", 
                 "KeyPageURL", "adid", "usertag"), 
      class = "data.frame", row.names = integer(0)) %>%
      data.table()
  }
  class(conv)
}) %>%
  lapply(`==`, c("data.table", "data.frame")) %>%
  sapply(all) %>%
  all() %>%
  stopifnot()
# parsing timestamp
clusterEvalQ(cl, {
  parse_Timestamp <- function(x) {
    t1 <- substring(x, 1, 14) %>%
      strptime("%Y%m%d%H%M%S") %>% 
      as.POSIXct(tz = "UTC") %>%
      as.numeric()
    t2 <- substring(x, 15, nchar(x)) %>%
      as.numeric()
    t <- t1 + t2 * 1e-3
    class(t) <- c("POSIXct", "POSIXt")
    t
  }
}) %>%
  invisible()
clusterEvalQ(cl, {
  bid %<>% mutate(bid_t = parse_Timestamp(Timestamp))
  imp %<>% mutate(imp_t = parse_Timestamp(Timestamp))
  clk %<>% mutate(clk_t = parse_Timestamp(Timestamp))
  conv %<>% mutate(conv_t = parse_Timestamp(Timestamp))
  NULL
}) %>%
  invisible()
# merging small datasets
clk.local <- clusterEvalQ(cl, {
  select(clk, BidID, clk_t)
}) %>%
  rbindlist() %>%
  tbl_dt() %>%
  group_by(BidID) %>%
  summarise(clk_t = min(clk_t)) %>%
  data.table()
stopifnot(!(clk.local$BidID %>% duplicated %>% any()))
conv.local <- clusterEvalQ(cl, {
  select(conv, BidID, conv_t)
}) %>%
  rbindlist() %>%
  tbl_dt() %>%
  group_by(BidID) %>%
  summarise(conv_t = min(conv_t)) %>%
  data.table()
stopifnot(!(conv.local$BidID %>% duplicated %>% any()))
clusterExport(cl, c("clk.local", "conv.local"))
clusterEvalQ(cl, {
  setkey(imp, "BidID")
  setkey(clk.local, "BidID")
  setkey(conv.local, "BidID")
  imp.clk.conv <- conv.local[clk.local[imp]]
  stopifnot(nrow(imp.clk.conv) == nrow(imp))
  stopifnot(all(imp.clk.conv$BidID == imp$BidID))
  stopifnot(!is.na(imp.clk.conv$BiddingPrice))
  stopifnot(!is.na(imp.clk.conv$PayingPrice))
  NULL
}) %>%
  invisible()

#rbindlist()
imp.clk.conv.id <- clusterEvalQ(cl, {
  select(imp.clk.conv, BidID) %>%
    mutate(id = .id)
}) %>%
  rbindlist() %>%
  select(BidID, id.imp = id)
setkey(imp.clk.conv.id, "BidID")
bid.id <- clusterEvalQ(cl, {
  select(bid, BidID) %>%
    mutate(id = .id)
}) %>%
  rbindlist() %>%
  select(BidID, id.bid = id)
setkey(bid.id, "BidID")
stopifnot(!(bid.id$BidID %>% duplicated() %>% any()))
gc()
imp.to.move <- local({
  # data.table syntax of: left_join(bid.id, imp.clk.conv.id, by = "BidID)
  .x <- imp.clk.conv.id[bid.id] %>%
    as.data.frame()
  stopifnot(!(.x$BidID %>% is.unsorted()))
  filter(.x, !is.na(id.imp)) %>%
    filter(id.imp != id.bid)
})
clusterExport(cl, "imp.to.move")

imp.clk.conv.to.move <- clusterEvalQ(cl, {
  imp.to.move %<>% data.table()
  target.bid.id <- filter(imp.to.move, id.imp == .id) %>%
    extract2("BidID")
  filter(imp.clk.conv, BidID %in% target.bid.id) %>%
    mutate(id.imp = .id) %>%
    left_join(select(imp.to.move, BidID, id.bid), by = "BidID", copy = TRUE)
}) %>%
  rbindlist()
clusterExport(cl, "imp.clk.conv.to.move")
clusterEvalQ(cl, {
  .imp <- rbind(
    imp.clk.conv, 
    filter(tbl_dt(imp.clk.conv.to.move), id.bid == .id) %>%
      select_(.dots = colnames(imp.clk.conv))
  ) %>% 
    data.table()
  setkey(.imp, "BidID")
  .imp2 <- .imp[,.(
    PayingPrice = max(PayingPrice, 1),
    usertag = usertag[which.max(nchar(usertag))],
    imp_t = min(imp_t),
    clk_t = min(clk_t),
    conv_t = min(conv_t)
  ),by=BidID]
  NULL
}) %>% 
  invisible()
clusterEvalQ(cl, {
  stopifnot(!is.na(.imp2$PayingPrice))
  stopifnot(!is.na(.imp2$BiddingPrice))
  stopifnot(!(.imp2$BidID %>% duplicated() %>% any()))
  NULL
}) %>% 
  invisible()

# add ip1,ip2,ip3,hour
clusterEvalQ(cl, {
  ips <- strsplit(bid$IP, split = ".", fixed = TRUE)
  ip1 <- sapply(ips, "[", 1)
  ip2 <- lapply(ips, "[", 1:2) %>%
    sapply(paste, collapse = ".")
  ip2[ip2 == "NA.NA"] <- NA
  ip3 <- lapply(ips, "[", 1:3) %>%
    sapply(paste, collapse = ".")
  ip3[ip3 == "NA.NA.NA"] <- NA
  .bid2 <-mutate(bid, ip1 = ip1, ip2 = ip2, ip3 = ip3, hour = format(bid_t, "%H"))
  NULL
}) %>%
  invisible()

# joining bid and imp.clk.conv
clusterEvalQ(cl, {
  if (is.null(key(.imp2))) setkey(.imp2, "BidID")
  if (is.null(key(.bid2))) setkey(.bid2, "BidID")
  NULL
}) %>%
  invisible()

clusterEvalQ(cl, {
  # left_join(.bid2, .imp2, by = "BidID")
  result <- .imp2[.bid2]
  NULL
}) %>% 
  invisible()

clusterEvalQ(cl, {
  .bid.hasimp <- filter(result, !is.na(imp_t))
  stopifnot(!is.na(.bid.hasimp$BiddingPrice))
  stopifnot(!is.na(.bid.hasimp$PayingPrice))
  NULL
}) %>%
  invisible()

# feature encoding
target.features <- c(
  "ip1", "ip2", "ip3", "Region", "City", "AdExchange", "Domain", "URL", "AdSlotId", "AdSlotWidth",
  "AdSlotHeight", "AdSlotVisibility", "AdSlotFormat", "hour"
)

clusterEvalQ(cl, {
  colnames(result)
}) %>%
  sapply(function(x) {
    all(target.features %in% x)
  }) %>%
  stopifnot()
target.levels <- sapply(target.features, simplify = FALSE, FUN = function(name) {
  force(name)
  cat(sprintf("Processing %s...\n", name))
  clusterExport(cl, "name", envir = environment())
  clusterEvalQ(cl, {
    table(result[[name]], exclude = NULL, dnn = "Value") %>%
      as.data.frame()
  }) %>%
    rbindlist() %>%
    group_by(Value) %>%
    summarise(Freq = sum(Freq)) %>%
    as.data.frame()
}) %>%
  lapply(function(df) {
    if (is.na(df$Value) %>% any() %>% `!`) {
      rbind(data.frame(Value = NA, Freq = 0L), arrange(df, desc(Freq)))
    } else {
      .x <- slice(df, order(df$Value, na.last = FALSE))
      rbind(
        slice(.x, 1),
        slice(.x, -1) %>%
          arrange(desc(Freq))
      )
    }
  })
sapply(target.levels, function(df) df$Value %>% is.na() %>% sum()) %>% `==`(1) %>% stopifnot()

total.n <- clusterEvalQ(cl, nrow(result)) %>%
  unlist() %>%
  sum()

# claim encoded.result
clusterExport(cl, "target.levels")
clusterEvalQ(cl, {
  encoded.result <- list()
  NULL
}) %>%
  invisible()
lapply(target.features, function(name) {
  force(name)
  cat(sprintf("Encoding %s...\n", name))
  clusterExport(cl, "name", envir = environment())
  clusterEvalQ(cl, {
    encoded.result[[name]] <- factor(result[[name]], levels = target.levels[[name]]$Value, exclude = NULL)
    NULL
  })
}) %>%
  invisible()

# usertag
clusterEvalQ(cl, {
  usertag.voc <- itoken(result$usertag, tokenizer = function(x) regexp_tokenizer(x, ","), progressbar = FALSE) %>%
    create_vocabulary()
  object.size(usertag.voc)
}) %>%
  unlist()

usertag.voc <- clusterEvalQ(cl, usertag.voc) %>%
  rbindlist() %>%
  as.data.frame() %>%
  group_by(term) %>%
  summarise_all(sum) %>%
  arrange(desc(term_count))
usertag.voc$term[usertag.voc$term == "NA"] <- NA
clusterExport(cl, "usertag.voc")
clusterEvalQ(cl, {
  it <- itoken(result$usertag, tokenizer = function(x) regexp_tokenizer(x, ","), progressbar = FALSE)
  encoded.result[["usertag"]] <- foreach(tokens = it$clone(deep = TRUE)) %do% {
    sapply(tokens$tokens, simplify = FALSE, function(token) {
      factor(token, levels = usertag.voc$term, exclude = NULL) %>% as.integer()
    })
  } %>%
    unlist(recursive = FALSE)
  attr(encoded.result[["usertag"]], "levels") <- usertag.voc$term
  stopifnot(length(encoded.result[["usertag"]]) == nrow(result))
  NULL
}) %>%
  invisible()

# save response
clusterEvalQ(cl, {
  responses <- list(
    imp = result$imp_t %>% is.na() %>% `!`,
    clk = result$clk_t %>% is.na() %>% `!`,
    conv = result$conv_t %>% is.na() %>% `!`,
    bp = result$BiddingPrice,
    wp = result$PayingPrice
  )
  lapply(c("imp", "clk", "conv", "bp"), function(name) {
    responses[[name]] %>%
      is.na() %>%
      sum()
  }) %>%
    sapply(`==`, 0) %>%
    lapply(stopifnot)
  stopifnot(responses$wp[responses$imp] %>% is.na() %>% sum() == 0)
  NULL
}) %>%
  invisible()

# validate sampled data

clusterEvalQ(cl, {
  for(name in names(encoded.result)) {
    if (is.factor(encoded.result[[name]])) {
      current <- encoded.result[[name]]
      expected <- result[[name]]
      stopifnot(current[!is.na(expected)] == expected[!is.na(expected)])
      stopifnot(current[is.na(expected)] %>% as.character() %>% is.na())
    } else if (is.list(encoded.result[[name]])) {
      .levels <- levels(encoded.result[[name]])
      .i <- length(encoded.result[[name]]) %>% seq_len()
      if (length(encoded.result[[name]]) > 100L) {
        .i <- sample(.i, 100L, replace = FALSE)
      }
      all.equal(
        encoded.result[[name]][.i] %>%
          lapply(function(i) .levels[i]) %>%
          sapply(function(x) {
            if (length(x) == 0) return("")
            if (is.na(x)) x else paste(x, collapse = ",")
          }),
        result[[name]][.i]
      ) %>%
        isTRUE() %>%
        stopifnot()
    } else stop(sprintf("TODO: %s", class(encoded.result[[name]][1])))
  }
  NULL
}) %>%
  invisible()
clusterEvalQ(cl, names(encoded.result))

# saving result to ipinyou.joined.dataset
clusterExport(cl, "argv")
dir.create(sprintf("../ipinyou.joined.dataset/%s", argv))
clusterEvalQ(cl, {
  root <- file.path(sprintf("../ipinyou.joined.dataset/%s", argv), .id)
  dir.create(root)
  NULL
}) %>% 
  invisible()

clusterEvalQ(cl, {
  saveRDS(result, file.path(root, "result.Rds"))
  NULL
}) %>% 
  invisible()

clusterEvalQ(cl, {
  saveRDS(encoded.result, file.path(root, "encoded.result.Rds"))
  NULL
}) %>% 
  invisible()

clusterEvalQ(cl, {
  saveRDS(responses, file.path(root, "responses.Rds"))
  NULL
}) %>% 
  invisible()

stopCluster(cl)
quit('no')
# it <- itoken(result$usertag, tokenizer = function(x) regexp_tokenizer(x, ","), progressbar = FALSE)
# usertag.voc <- create_vocabulary(it)
# saveRDS(result, sprintf("../ipinyou.joined.dataset/%s.Rds", argv))
