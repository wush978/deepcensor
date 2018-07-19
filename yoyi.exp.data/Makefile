all : test.obj.Rds train.obj.done

train.obj.done : parseYOYI.cpp
	#constructing train.obj.01.Rds train.obj.02.Rds train.obj.03.Rds train.obj.04.Rds train.obj.05.Rds train.obj.06.Rds train.obj.07.Rds train.obj.08.Rds train.obj.09.Rds train.obj.10.Rds
	Rscript train.obj.R
	touch $@

test.obj.Rds : ../yoyi/test.yzx.txt.gz parseYOYI.cpp
	Rscript -e "Rcpp::sourceCpp('parseYOYI.cpp');saveRDS(parseYOYI(readLines(gzfile('$<'))), '$@')"

