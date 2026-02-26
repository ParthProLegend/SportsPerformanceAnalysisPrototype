@echo off

set "VENV_DIR=%~dp0.venv"

echo Activating virtual environment...

if not exist "%VENV_DIR%\Scripts\Activate.ps1" (
	echo Virtual environment not found. Creating one at %VENV_DIR% ...
	python -m venv "%VENV_DIR%" || (
		echo Failed to create virtual environment. Ensure Python is installed and on PATH.
		goto :after_venv
	)
)

powershell -ExecutionPolicy Bypass -File "%VENV_DIR%\Scripts\Activate.ps1" || echo Failed to activate virtual environment.

@REM Alternatively, use this command instead:
@REM call %~dp0.venv\Scripts\Activate.ps1 || echo Failed to activate virtual environment.

echo Installing dependencies...
python -m pip install -r %~dp0requirements.txt || echo Failed to install dependencies.

echo Running main.py...
python %~dp0src\SportsPerformance\main.py || echo Failed to run main.py.

@REM call %~dp0src\SportsPerformance\.venv\Scripts\deactivate.bat || echo Failed to deactivate virtual environment.
echo Script execution completed.

:after_venv