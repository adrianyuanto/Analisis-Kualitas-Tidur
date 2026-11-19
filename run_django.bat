@echo off
title Menjalankan Django - Sistem Analisis Tidur
cd /d "%~dp0"

echo ========================================================
echo   Sistem Analisis Tidur - Machine Learning Platform
echo   Menjalankan server lokal Django...
echo ========================================================
echo.

if exist "C:\laragon\bin\python\python-3.13\python.exe" (
    set "PATH=C:\laragon\bin\python\python-3.13;C:\laragon\bin\python\python-3.13\Scripts;%PATH%"
    set PYTHON_EXE=C:\laragon\bin\python\python-3.13\python.exe
) else (
    set PYTHON_EXE=python
)

echo Menggunakan Python: %PYTHON_EXE%
echo Membuka browser ke http://127.0.0.1:8000 ...
start http://127.0.0.1:8000

"%PYTHON_EXE%" manage.py runserver 127.0.0.1:8000

pause
