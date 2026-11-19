@echo off
title Menjalankan Streamlit - Sistem Analisis Tidur
cd /d "%~dp0"

echo ========================================================
echo   Sistem Analisis Tidur - Machine Learning Platform
echo   Menjalankan antarmuka Streamlit...
echo ========================================================
echo.

if exist "C:\laragon\bin\python\python-3.13\python.exe" (
    set "PATH=C:\laragon\bin\python\python-3.13;C:\laragon\bin\python\python-3.13\Scripts;%PATH%"
    set PYTHON_EXE=C:\laragon\bin\python\python-3.13\python.exe
) else (
    set PYTHON_EXE=python
)

echo Menggunakan Python: %PYTHON_EXE%
"%PYTHON_EXE%" -m streamlit run app.py

pause
