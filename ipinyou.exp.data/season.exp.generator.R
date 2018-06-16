FILTERED_VALUE_NAME <- "__filtered__"
library(magrittr)
library(parallel)
library(data.table)
library(dplyr)
library(dtplyr)
library(iterators)
library(itertools)
if (getwd() %>% basename() != "ipinyou.exp.data") setwd("ipinyou.exp.data")
# argv: <season> <filter.ratio>
if (interactive()) argv <- c("201310", "1e-4") else argv <- commandArgs(TRUE)
print(argv)
# list required workers
stopifnot(length(argv) >= 2)
root <- file.path("..", "ipinyou.joined.dataset", argv[1])
filter.ratio <- as.numeric(argv[2])
stopifnot(isTRUE(filter.ratio < 1) & isTRUE(filter.ratio >= 0))
stopifnot(file.exists(root))
workers <- dir(root)
# initializing workers
cl <- makeCluster(length(workers))
parLapply(cl, workers, function(id) {
  assign(".id", id, envir = globalenv())
  NULL
}) %>%
  invisible()
clusterEvalQ(cl, {
  library(magrittr)
  library(dplyr)
  library(data.table)
  NULL
}) %>%
  invisible()
# loading data
clusterExport(cl, "root")
clusterEvalQ(cl, {
  encoded.result <- readRDS(file.path(root, .id, "encoded.result.Rds"))
  sapply(encoded.result, class)
}) %>%
  lapply(function(x) {
    all.equal(
      x,
      structure(
        c("factor", "factor", "factor", "factor", "factor", 
          "factor", "factor", "factor", "factor", "factor", "factor", "factor", 
          "factor", "factor", "list"),
        .Names = c(
          "ip1", "ip2", "ip3", 
          "Region", "City", "AdExchange", "Domain", "URL", "AdSlotId", 
          "AdSlotWidth", "AdSlotHeight", "AdSlotVisibility", "AdSlotFormat", 
          "hour", "usertag"))
    ) %>%
      isTRUE() %>%
      stopifnot()
  }) %>%
  invisible()
clusterEvalQ(cl, {
  responses <- readRDS(file.path(root, .id, "responses.Rds"))
  sapply(responses, class)
}) %>%
  lapply(function(x) {
    all.equal(
      x,
      structure(
        c("logical", "logical", "logical", "numeric", "numeric"
        ), 
        .Names = c(
          "imp", "clk", "conv", "bp", "wp"))
    ) %>%
      isTRUE() %>%
      stopifnot()
  }) %>%
  invisible()

cat(sprintf("Filtering imp data...\n"))
clusterEvalQ(cl, {
  .i <- responses$imp
  for(name in names(encoded.result)) {
    .x <- encoded.result[[name]]
    encoded.result[[name]] <- .x[.i]
    attributes(encoded.result[[name]]) <- attributes(.x)
  }
  for(name in names(responses)) {
    .x <- responses[[name]]
    responses[[name]] <- .x[.i]
    attributes(responses[[name]]) <- attributes(.x)
  }
  NULL
}) %>%
  invisible()

# filter main.effect via filter.ratio
cat(sprintf("Filtering main.effect...\n"))
n.total <- clusterEvalQ(cl, length(responses$imp)) %>%
  unlist() %>%
  sum()
encoded.result.table <- clusterEvalQ(cl, {
  lapply(names(encoded.result), function(name) {
    if (encoded.result[[name]] %>% is.factor()) {
      table(encoded.result[[name]], dnn = "Value", exclude = NULL) %>%
        as.data.frame(stringsAsFactors = FALSE) %>%
        mutate(name = name) %>%
        select(name, Value, Freq)
    } else if (is.list(encoded.result[[name]])) {
      .x <- unlist(encoded.result[[name]]) %>%
        table(dnn = "Value", exclude = NULL) %>%
        as.data.frame(stringsAsFactors = FALSE)
      if (nrow(.x) > 0) {
        .x %<>% mutate(name = name)
      } else {
        .x <- data.frame(Value = integer(0), Freq = integer(0), name = character(0))
      }
      select(.x, name, Value, Freq)
    }
  }) %>%
    rbindlist() %>%
    mutate(name = as.character(name))
}) %>%
  rbindlist() %>%
  group_by(name, Value) %>%
  summarise(Freq = sum(Freq))
filtered.result.table <- filter(encoded.result.table, is.na(Value) | Freq > n.total * filter.ratio)
filtered.variables <- group_by(filtered.result.table, name) %>%
  summarise(nValue = length(Value)) %>%
  filter(nValue > 1) %>%
  extract2("name") %>%
  as.character()
filtered.result.table %<>% filter(name %in% filtered.variables)

filtered.result.table.new <- group_by(filtered.result.table, name) %>%
  summarise(Freq.filtered = sum(Freq)) %>%
  left_join(
    group_by(encoded.result.table, name) %>%
      summarise(Freq.origin = sum(Freq))
  ) %>%
  filter(Freq.origin > Freq.filtered) %>%
  mutate(Value = FILTERED_VALUE_NAME, Freq = Freq.origin - Freq.filtered) %>%
  select(name, Value, Freq)
filtered.result.table %<>% rbind(
  filtered.result.table.new 
)
clusterExport(cl, "FILTERED_VALUE_NAME")
clusterExport(cl, "filtered.result.table")
  
#claim result
cat("Generating exp.data...\n")
clusterEvalQ(cl, {
  exp.data <- list()
  attr(exp.data, "levels") <- list()
  na.is.value <- function(x, value) {
    if (sum(is.na(x)) > 0) x[is.na(x)] <- value
    x
  }
  NULL
}) %>%
  invisible()
# level1
clusterEvalQ(cl, {
  for(name.value in unique(filtered.result.table$name)) {
    .levels <- filter(filtered.result.table, name == name.value) %>%
      extract2("Value")
    if (is.factor(encoded.result[[name.value]])) {
      exp.data[[name.value]] <- factor(encoded.result[[name.value]], levels = .levels, exclude = NULL) %>%
        as.integer() %>%
        na.is.value(which(.levels == FILTERED_VALUE_NAME)) %>%
        `-`(1L)
      stopifnot(is.na(exp.data[[name.value]]) %>% sum() == 0)
      attr(exp.data[[name.value]], "levels") <- .levels
      attr(exp.data[[name.value]], "level") <- 1L
    } else if (is.list(encoded.result[[name.value]])) {
      if (is.null(exp.data[[name.value]])) exp.data[[name.value]] <- integer(0)
      .levels.df <- 
        filter(filtered.result.table, name == name.value) %>% 
          mutate(index.new = seq_len(nrow(.))) %>%
          select(index.origin = Value, index.new) %>%
        left_join(
          data.frame(
            stringsAsFactors = FALSE,
            Value = attr(encoded.result[[name.value]], "levels"),
            index.origin  = attr(encoded.result[[name.value]], "levels") %>% seq_along() %>% as.character()
          )
        )
      .levels.df$Value[.levels.df$index.origin == FILTERED_VALUE_NAME] <- FILTERED_VALUE_NAME
      exp.data[[name.value]] <- encoded.result[[name.value]] %>%
        unlist() %>%
        factor(levels = .levels, exclude = NULL) %>%
        as.integer() %>%
        na.is.value(which(.levels == FILTERED_VALUE_NAME)) %>%
        `-`(1L)
      attr(exp.data[[name.value]], "levels") <- .levels.df$Value
      attr(exp.data[[name.value]], "level") <- 1L
      attr(exp.data[[name.value]], "class") <- "compressed.list"
      attr(exp.data[[name.value]], "element.size") <- sapply(encoded.result[[name.value]], length)
    } else stop("TODO")
  }
  NULL
}) %>%
  invisible()
## validation

clusterEvalQ(cl, {
  for(name.value in names(exp.data)) {
    if (class(exp.data[[name.value]]) == "integer") {
      stopifnot(length(exp.data[[name.value]]) == length(encoded.result[[name.value]]))
    } else if (class(exp.data[[name.value]]) == "compressed.list") {
      stopifnot(length(exp.data[[name.value]]) == attr(exp.data[[name.value]], "element.size") %>% unlist() %>% sum())
      stopifnot(length(exp.data[[name.value]]) == lapply(encoded.result[[name.value]], length) %>% unlist() %>% sum())
    } else stop("Unknown class")
  }
  NULL
}) %>%
  invisible()
# level2
level1.names <- clusterEvalQ(cl, {
  level1.names <- Filter(f = function(x) attr(x, "level") == 1L, exp.data) %>%
    names()
})
lapply(level1.names, function(x) {
  all.equal(x, level1.names[[1]]) %>%
    isTRUE() %>%
    stopifnot()
}) %>%
  invisible()
level1.names <- level1.names[[1]]
level2.names <- combn(level1.names, 2) %>%
  iter(by = "column") %>%
  lapply(function(x) {
    x <- as.vector(x)
    attr(x, "name") <- paste(x, collapse = ":")
    x
  })
clusterExport(cl, "level2.names")
clusterEvalQ(cl, {
  for(name.list in level2.names) {
    e1 <- exp.data[[name.list[1]]]
    e2 <- exp.data[[name.list[2]]]
    name.value <- attr(name.list, "name")
    stopifnot(class(e1) == "integer")
    .levels1 <- levels(e1)
    .levels2 <- levels(e2)
    if (class(e2) == "integer") {
      exp.data[[name.value]] <- e2 * length(.levels1) + e1
      attr(exp.data[[name.value]], "level") <- 2L
    } else if (class(e2) == "compressed.list") {
      .start <- 0
      .e1.i <- 1
      exp.data[[name.value]] <- integer(length(e2))
      for(.size in attr(e2, "element.size")) {
        .i <- seq(.start + 1, length.out = .size, by = 1)
        exp.data[[name.value]][.i] <- e2[.i] * length(.levels1) + e1[.e1.i]
        .e1.i <- .e1.i + 1
        .start <- .start + .size
      }
      attr(exp.data[[name.value]], "level") <- 2L
      attr(exp.data[[name.value]], "class") <- "compressed.list"
      attr(exp.data[[name.value]], "element.size") <- attr(e2, "element.size")
    } else stop(sprintf("TODO: %s", class(e2) %>% paste(collapse = " ")))
  }
  NULL
}) %>%
  invisible()
# compressing level2
## Constructing counting table of levels...
levels2.table <- clusterEvalQ(cl, {
  Filter(f = function(x) attr(x, "level") == 2L, exp.data) %>%
    names() %>%
    lapply(function(name.value) {
      .r <- table(exp.data[[name.value]], dnn = "Value", exclude = NULL) %>%
        as.data.frame(stringsAsFactors = FALSE)
      if (nrow(.r) == 0) {
        data.frame(
          stringsAsFactors = FALSE,
          name = character(0),
          Value = character(0),
          Freq = integer(0)
        )
      } else {
        mutate(.r, name = name.value) %>%
          select(name, Value, Freq)
      }
    }) %>%
    rbindlist()
}) %>% 
  rbindlist() %>%
  group_by(name, Value) %>%
  summarise(Freq = sum(Freq))
## filter out levels
levels2.filtered.table <- filter(levels2.table, Freq > n.total * filter.ratio)
filtered.levels2.variables <- group_by(levels2.filtered.table, name) %>%
  summarise(nValue = length(Value)) %>%
  filter(nValue > 1) %>%
  extract2("name")
levels2.filtered.table %<>% filter(name %in% filtered.levels2.variables)
levels2.filtered.table.new <- group_by(levels2.filtered.table, name) %>%
  summarise(Freq.filtered = sum(Freq)) %>%
  left_join(
    group_by(levels2.table, name) %>%
      summarise(Freq.original = sum(Freq))
  ) %>%
  mutate(Value = FILTERED_VALUE_NAME, Freq = Freq.original - Freq.filtered) %>%
  select(name, Value, Freq)
levels2.filtered.table <- rbind(
  levels2.filtered.table, 
  levels2.filtered.table.new
)
clusterExport(cl, "levels2.filtered.table")
clusterEvalQ(cl, {
  levels2.names.list <- unique(levels2.filtered.table$name) %>%
    strsplit(split = ":")
  .backup.lv2 <- list()
  for(name.list in levels2.names.list) {
    name.value <- paste(name.list, collapse = ":")
    .backup.lv2[[name.value]] <- exp.data[[name.value]]
  }
  NULL
}) %>%
  invisible()
clusterEvalQ(cl, {
  for(name.value in names(exp.data)) {
    if (exp.data[[name.value]] %>% attr("level") == 2L) {
      if (!name.value %in% unique(levels2.filtered.table$name)) exp.data[[name.value]] <- NULL
    }
  }
  for(name.list in levels2.names.list) {
    name.value <- paste(name.list, collapse = ":")
    .x <- .backup.lv2[[name.value]]
    .levels <- filter(levels2.filtered.table, name == name.value) %>%
      extract2("Value")
    exp.data[[name.value]] <- factor(.x, levels = .levels, exclude = NULL) %>%
      as.integer() %>%
      na.is.value(which(.levels == FILTERED_VALUE_NAME)) %>%
      `-`(1L)
    attributes(exp.data[[name.value]]) <- attributes(.x)
    l1 <- exp.data[[name.list[1]]] %>% levels()
    l2 <- exp.data[[name.list[2]]] %>% levels()
    attr(exp.data[[name.value]], "levels") <- filter(levels2.filtered.table, name == name.value) %>%
      mutate(e1.value = as.integer(Value) %% length(l1), e2.value = as.integer(Value) %/% length(l1)) %>%
      mutate(e1.value = l1[e1.value + 1], e2.value = l2[e2.value + 1]) %>%
      mutate(level = ifelse(is.na(as.integer(Value)), FILTERED_VALUE_NAME, paste(e1.value, e2.value, sep = ":"))) %>%
      extract2("level")
  }
  NULL
}) %>%
  invisible()

# validations
clusterEvalQ(cl, {
  .expand <- function(x) {
    if (is.list(x)) {
      r <- unlist(x)
      attr(x, "levels")[r]
    } else as.character(x)
  }
  for(name.value in names(exp.data)) {
    .i <- seq_len(length(exp.data[[name.value]]))
    if (length(.i) == 0) next
    if (length(exp.data[[name.value]]) > 100L) .i <- sample(.i, 100L, FALSE)
    if (attr(exp.data[[name.value]], "level") == 1L) {
      .x <- data.frame(
        stringsAsFactors = FALSE,
        current = attr(exp.data[[name.value]], "levels")[exp.data[[name.value]][.i] + 1], 
        expected = encoded.result[[name.value]] %>% .expand() %>% `[`(.i)
      ) %>%
        mutate(filtered = current == FILTERED_VALUE_NAME)
      filter(.x, filtered) %>%
        mutate(check = !expected %in% attr(exp.data[[name.value]], "levels")) %>%
        extract2("check") %>%
        stopifnot()
      filter(.x, !filtered) %>%
        mutate(check = expected == current) %>%
        extract2("check") %>%
        stopifnot()
    } else if (attr(exp.data[[name.value]], "level") == 2L) {
      name.list <- strsplit(name.value, split = ":")[[1]]
      l1 <- levels(exp.data[[name.list[1]]])
      l2 <- levels(exp.data[[name.list[2]]])
      if (class(exp.data[[name.list[2]]]) == "integer") {
        .x <- data.frame(
          stringsAsFactors = FALSE,
          current = attr(exp.data[[name.value]], "levels")[exp.data[[name.value]][.i] + 1], 
          e1 = encoded.result[[name.list[1]]] %>% .expand() %>% `[`(.i),
          e2 = encoded.result[[name.list[2]]] %>% .expand() %>% `[`(.i)
        )
      } else if (class(exp.data[[name.list[2]]]) == "compressed.list") {
        .p <- attr(exp.data[[name.list[2]]], "element.size") %>% 
          cumsum()
        .x <- data.frame(
          stringsAsFactors = FALSE,
          current = attr(exp.data[[name.value]], "levels")[exp.data[[name.value]][.i] + 1], 
          e1 = encoded.result[[name.list[1]]] %>% .expand() %>% `[`(sapply(.i, function(i) which(.p >= i)[1])),
          e2 = encoded.result[[name.list[2]]] %>% .expand() %>% `[`(.i)
        )
      }
        .x %<>% mutate(
          left.filtered = grepl(sprintf("%s:", FILTERED_VALUE_NAME), current, fixed = TRUE),
          right.filtered = grepl(sprintf(":%s", FILTERED_VALUE_NAME), current, fixed = TRUE),
          lv2.filtered = grepl(sprintf("^%s$", FILTERED_VALUE_NAME), current),
          non.filtered = !(left.filtered | right.filtered | lv2.filtered),
          expected = paste(e1, e2, sep = ":")
        ) %>%
        mutate(
          current.left = strsplit(current, split = ":") %>% lapply(function(x) {
            if (length(x) != 2) NA else {
              x[1]
            }
          }),
          current.right = strsplit(current, split = ":") %>% lapply(function(x) {
            if (length(x) != 2) NA else {
              x[2]
            }
          })
        )
      select(.x, left.filtered : non.filtered) %>%
        apply(1, function(x) (x[1] | x[2]) + x[3] + x[4]) %>%
        `==`(1) %>%
          stopifnot()
      filter(.x, left.filtered) %>%
        mutate(check = !e1 %in% l1) %>%
        extract2("check") %>%
        stopifnot()
      filter(.x, !left.filtered, !lv2.filtered) %>%
        mutate(check = e1 == current.left) %>%
        extract2("check") %>%
        stopifnot()
      filter(.x, right.filtered) %>%
        mutate(check = !e2 %in% l2) %>%
        extract2("check") %>%
        stopifnot()
      filter(.x, !right.filtered, !lv2.filtered) %>%
        mutate(check = e2 == current.right) %>%
        extract2("check") %>%
        stopifnot()
      filter(.x, lv2.filtered) %>%
        mutate(check = !expected %in% attr(exp.data[[name.value]], "levels")) %>%
        extract2("check") %>%
        stopifnot()
      filter(.x, non.filtered) %>%
        mutate(check = current == expected) %>%
        extract2("check") %>%
        stopifnot()
    } else stop("TODO")
  }
  NULL
}) %>%
  invisible()

dst.root <- file.path("..", "ipinyou.exp.data", paste(argv, collapse = "_"))
dir.create(dst.root, showWarnings = FALSE)
clusterExport(cl, "dst.root")
clusterEvalQ(cl, {
  dir.create(file.path(dst.root, .id))
  saveRDS(responses, file = file.path(dst.root, .id, "responses.Rds"))
  saveRDS(exp.data, file = file.path(dst.root, .id, "exp.data.Rds"))
  NULL
}) %>% 
  invisible()

clusterEvalQ(cl, {
  library(hdf5r)
  fname <- file.path(dst.root, .id, "exp.data.h5")
  if (file.exists(fname)) file.remove(fname)
  file.h5 <- H5File$new(fname, mode = "w")
  lv1.grp <- file.h5$create_group("lv1")
  lv2.grp <- file.h5$create_group("lv2")
  responses.grp <- file.h5$create_group("responses")
  for(name in names(exp.data)) {
    if (attr(exp.data[[name]], "level") == 1L) {
      .g <- lv1.grp$create_group(name)
    } else if (attr(exp.data[[name]], "level") == 2L) {
      .g <- lv2.grp$create_group(name)
    } else stop("TODO")
    if (length(exp.data[[name]]) > 0) {
      .g[[".value"]] <- exp.data[[name]] %>% as.vector()
    } else .g[[".value"]] <- NA
    for(attr.name in attributes(exp.data[[name]]) %>% names()) {
      # h5::h5attr(file.h5[file.path("lv1", name)], attr.name) <- attr(exp.data[[name]], attr.name)
      if (length(attr(exp.data[[name]], attr.name)) > 0) {
        .g[[attr.name]] <- attr(exp.data[[name]], attr.name)
      } else .g[[attr.name]] <- NA
    }
    if (!"class" %in% .g$ls()$name) {
      .g[["class"]] <- class(exp.data[[name]])
    }
  }
  for(name in names(responses)) {
    .g <- responses.grp$create_group(name)
    if (length(responses[[name]]) > 0) {
      .g[[".value"]] <- responses[[name]]
    } else .g[[".value"]] <- NA
    for(attr.name in attributes(responses[[name]]) %>% names()) {
      if (length(attr(responses[[name]], attr.name)) > 0) {
        .g[[attr.name]] <- attr(responses[[name]], attr.name)
      } else .g[[attr.name]] <- NA
    }
    if (!"class" %in% .g$ls()$name) {
      .g[["class"]] <- class(responses[[name]])
    }
  }
  file.h5$close_all()
  NULL
}) %>%
  invisible()
stopCluster(cl)
#exp.data <- clusterEvalQ(cl, sapply(exp.data, head, simplify = FALSE))[[1]]
#attr(exp.data, "levels") <- clusterEvalQ(cl, sapply(attr(exp.data, "levels"), head, simplify = FALSE))[[1]]
