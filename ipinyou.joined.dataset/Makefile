.SECONDEXPANSION:

all : .201306.done .201310.done

.201310.done :
	Rscript season.join.R 201310
	touch $@

.201306.done :
	Rscript season.join.R 201306
	touch $@

