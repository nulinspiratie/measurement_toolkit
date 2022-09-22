@echo off
cls

:loop
:: Your code using %1
ipython C:\Users\%username%\Documents\GitHub\measurement_toolkit\measurement_toolkit\tools\notebook_to_html_scripts\export_notebook_script.py %1

:: Check for further batch arguments.     
shift /1
IF [%1]==[] (
goto end
) ELSE (
goto loop
)

:end