@echo off
setlocal

if not "%RSCRIPT%"=="" if exist "%RSCRIPT%" goto run

for /d %%D in ("C:\Program Files\R\R-*") do set "RSCRIPT=%%D\bin\Rscript.exe"
if exist "%RSCRIPT%" goto run

for /d %%D in ("C:\Program Files (x86)\R\R-*") do set "RSCRIPT=%%D\bin\Rscript.exe"
if exist "%RSCRIPT%" goto run

echo Rscript.exe was not found. Install R or set RSCRIPT to the full Rscript.exe path.
exit /b 1

:run
if "%R_LIBS_USER%"=="" set "R_LIBS_USER=%~dp0.Rlibs"
echo Using Rscript: %RSCRIPT%
echo Using R_LIBS_USER: %R_LIBS_USER%
"%RSCRIPT%" "%~dp0install_r_packages.R"
exit /b %ERRORLEVEL%
