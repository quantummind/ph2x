TEX = pdflatex -shell-escape -interaction=nonstopmode -file-line-error
MAKEPLOTS = python lab3.py

all:	report.pdf

view:
	open report.pdf

clean:
	rm -f report.pdf report.aux report.log report.out *.png

1.png 2.png 3.png 4.png 5.png 6.png : lab3.py
	$(MAKEPLOTS)

report.pdf: 1.png 2.png 3.png 4.png 5.png 6.png
	$(TEX) report.tex