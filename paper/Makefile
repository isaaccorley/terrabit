MAIN = main
TEXFILES = $(wildcard *.tex)
BIBFILES = $(wildcard *.bib)

.PHONY: install check build clean watch

install:
	tlmgr update --self
	tlmgr install latexmk collection-fontsrecommended booktabs natbib microtype hyperref environ trimspaces xcolor graphicx amsmath amsfonts units

check:
	tex-fmt --check $(TEXFILES)

build: $(MAIN).pdf

$(MAIN).pdf: $(TEXFILES) $(BIBFILES) neurips_2025.sty
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(MAIN).tex

watch:
	latexmk -pdf -pvc -interaction=nonstopmode $(MAIN).tex

clean:
	latexmk -C
	rm -f *.bbl *.run.xml *.synctex.gz
