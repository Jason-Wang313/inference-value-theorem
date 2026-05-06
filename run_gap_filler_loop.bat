@echo off
title Gap Filler - Auto Restart
cd /d "%~dp0"

:loop
echo [%date% %time%] Starting gap filler...
"C:\Users\wangz\AppData\Local\Programs\Python\Python310\python.exe" run_gap_filler.py --workers 3
echo [%date% %time%] Process exited with code %ERRORLEVEL%
echo Restarting in 10 seconds...
timeout /t 10 /nobreak >nul
goto loop
