@echo off
chcp 65001 > nul
title Kaggle Notebook Radar
echo ====================================================================
echo   Starting Kaggle Notebook Radar Web Dashboard...
echo   URL: http://127.0.0.1:8792
echo ====================================================================
cd /d "%~dp0"
python tools\notebook_radar\notebook_radar.py serve --port 8792
pause
