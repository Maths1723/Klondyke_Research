@echo off
REM Creates .venv with the Windows py launcher, installs dependencies, runs the tests.
cd /d "%~dp0"
py -3 -m venv .venv || goto :err
.venv\Scripts\python.exe -m pip install --upgrade pip || goto :err
.venv\Scripts\python.exe -m pip install -r requirements.txt || goto :err
.venv\Scripts\python.exe -m pytest -q tests || goto :err
echo.
echo Environment ready. Activate with:  .venv\Scripts\activate
goto :eof
:err
echo Setup failed - see the message above.
exit /b 1
