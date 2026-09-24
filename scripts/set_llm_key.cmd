@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0set_llm_key_gui.ps1"
if errorlevel 1 pause
