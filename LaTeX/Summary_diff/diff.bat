@echo off
setlocal

set "old_path=..\Summary_v3\"
set "new_path=..\Summary_v4\"
set "doc_name_filename=Summary"

echo Generate %doc_name_filename%_flat.tex for %new_path%
cd %new_path%
latexpand %doc_name_filename%.tex > %doc_name_filename%_flat.tex

echo Generate %doc_name_filename%_flat.tex for %old_path%
cd %old_path%
latexpand %doc_name_filename%.tex > %doc_name_filename%_flat.tex

echo Generate diff
cd %new_path%
latexdiff %old_path%%doc_name_filename%_flat.tex %doc_name_filename%_flat.tex > diff.tex
latexmk  -synctex=1 -interaction=nonstopmode -file-line-error -pdf -outdir=%OUTDIR% %DOC% -shell-escape diff.tex
echo PDF generated in case of problems see diff.log

echo Cleaning up
del %doc_name_filename%_flat.tex
del %old_path%%doc_name_filename%_flat.tex

pause