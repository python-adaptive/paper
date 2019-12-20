paper.pdf: paper.bbl paper.tex figures/*.pdf
	pdflatex paper.tex
	pdflatex paper.tex

paper.bbl: paper.tex paper.bib
	pdflatex paper.tex
	bibtex paper.aux

paper.tex: Makefile paper.md pandoc/revtex.template
	pandoc \
        --read=markdown-auto_identifiers \
        --filter=pandoc-fignos \
        --filter=pandoc-citeproc \
        --filter=pandoc-crossref \
        --metadata="crossrefYaml=pandoc/pandoc-crossref.yaml" \
        --output=paper.tex \
        --bibliography=paper.bib \
        --abbreviations=pandoc/abbreviations.txt \
        --wrap=preserve \
        --template=pandoc/revtex.template \
        --standalone \
        --natbib \
        --listings \
        paper.md

.PHONY: all clean

clean:
	rm -f paper.pdf paper.aux paper.blg paper.bbl paper.log paper.tex paperNotes.bib
