@echo off
set PATH=%~dp0tools\ffmpeg\bin;%PATH%
start "" "%~dp0.venv\Scripts\pythonw.exe" "%~dp0reconstruction_gui\reconstruction_zone.py"
