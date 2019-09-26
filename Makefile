paper.pdf: paper.bbl paper.tex
	pdflatex paper.tex
	pdflatex paper.tex

paper.bbl: paper.tex paper.bib
	pdflatex paper.tex
	bibtex paper.aux

paper.tex: paper.md revtex.template
	pandoc -s --filter pandoc-fignos --filter pandoc-citeproc --filter pandoc-crossref --natbib paper.md -o paper.tex --bibliography paper.bib --listings --template revtex.template
