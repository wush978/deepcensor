all : .201306.done .201310.done

.201306.done :
	Rscript season.exp.generator.R 201306 1e-4
	touch $@

.201310.done :
	Rscript season.exp.generator.R 201310 1e-4
	touch $@


