PAPER=main
FIGURES=figures/*


all: ${PAPER}.pdf

${PAPER}.pdf: $(shell ls source/*.tex *.tex *.bib)
	pdflatex ${PAPER}
	bibtex ${PAPER}
	pdflatex ${PAPER}
	pdflatex ${PAPER}
	rm $(PAPER)-embed.pdf *.bbl *.blg *.dvi *.fls *.log *.tex~ *.tex.backup *.bib~ *.bak *.sav *.aux || echo none


clean:
	rm ${PAPER}.pdf ${PAPER}-embed.pdf *.blg *.dvi *.fls *.log *.tex~ *.tex.backup *.bib~ *.bak *.sav || echo none

open:
	open ${PAPER}.pdf
