paper.pdf: bbl
	pdflatex paper.tex
	pdflatex paper.tex

bbl: paper.tex
	pdflatex paper.tex
	bibtex paper.aux

paper.tex:
	pandoc -s --filter pandoc-fignos --filter pandoc-citeproc --filter pandoc-crossref --natbib paper.md -o paper.tex --bibliography paper.bib --listings -H listings-setup.tex --template revtex.template
