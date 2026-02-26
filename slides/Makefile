MAIN = main
TEXFILES = $(wildcard *.tex)

.PHONY: install check build clean watch

install:
	tlmgr update --self
	tlmgr install latexmk beamer pgf booktabs hyperref amsmath amsfonts xcolor tikz

check:
	tex-fmt --check $(TEXFILES)

build: $(MAIN).pdf

$(MAIN).pdf: $(TEXFILES)
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(MAIN).tex

watch:
	latexmk -pdf -pvc -interaction=nonstopmode $(MAIN).tex

clean:
	latexmk -C
	rm -f *.bbl *.run.xml *.synctex.gz *.nav *.snm *.vrb
