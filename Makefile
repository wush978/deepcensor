
all : .progress/ipinyou.exp.data

.progress/ipinyou.contest.dataset : ipinyou.contest.dataset.zip
	-(ls ipinyou.contest.dataset && rm -rf ipinyou.contest.dataset) > /dev/null 2>&1
	unzip ipinyou.contest.dataset.zip
	cd ipinyou.contest.dataset && md5sum -c files.md5
	touch $@

.progress/ipinyou.joined.dataset : .progress/ipinyou.contest.dataset
	-rm -rf ipinyou.joined.dataset
	git checkout ipinyou.joined.dataset
	$(MAKE) -C ipinyou.joined.dataset
	touch $@

.progress/ipinyou.exp.data : .progress/ipinyou.joined.dataset
	-rm -rf ipinyou.exp.data
	git checkout ipinyou.exp.data
	$(MAKE) -C ipinyou.exp.data
	touch $@
