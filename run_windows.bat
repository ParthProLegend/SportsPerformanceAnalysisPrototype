@echo off

echo Activating virtual environment...
powershell -ExecutionPolicy Bypass -File %~dp0.venv\Scripts\Activate.ps1 || echo Failed to activate virtual environment.

@REM If the .ps1 execution popup doesn't appear for you, use this command instead:
@REM call %~dp0.venv\Scripts\Activate.ps1 || echo Failed to activate virtual environment.

echo Installing dependencies...
python -m pip install -r %~dp0requirements.txt || echo Failed to install dependencies.

echo Running main.py...
python %~dp0src\SportsPerformance\main.py || echo Failed to run main.py.

@REM call %~dp0src\SportsPerformance\.venv\Scripts\deactivate.bat || echo Failed to deactivate virtual environment.
echo Script execution completed.