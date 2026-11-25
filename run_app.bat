@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
title VAA FlowVision - 1 Camera
python --version >nul 2>&1
if errorlevel 1 (
 echo [LOI] Chua tim thay Python. Cai Python 3.10+ va chon Add Python to PATH.
 pause
 exit /b 1
)
python -c "import streamlit,cv2,ultralytics,plotly,pandas" >nul 2>&1
if errorlevel 1 (
 echo Dang cai thu vien tu requirements.txt...
 python -m pip install -r requirements.txt
 if errorlevel 1 (echo Cai dat loi.& pause & exit /b 1)
)
echo Truy cap: http://localhost:8501
echo Luu y: Video 30 phut nen xu ly tren may co GPU neu co.
python -m streamlit run app.py
pause
