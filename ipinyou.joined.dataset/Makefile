.SECONDEXPANSION:


BIDS_2ND=$(shell find ../ipinyou.contest.dataset/training2nd | grep bid | Rscript get.target.R)
BIDS_3RD=$(shell find ../ipinyou.contest.dataset/training3rd | grep bid | Rscript get.target.R)


all : $(BIDS_3RD) $(BIDS_2ND)

%.Rds : $$(shell find ../ipinyou.contest.dataset/* | grep %)
	Rscript join.R $@


