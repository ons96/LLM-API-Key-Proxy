@echo off
REM ============================================================================
REM WSL Launcher for LLM Gateway Testing
REM ============================================================================
REM Double-click this file to open WSL and run all tests automatically

echo Starting WSL and running LLM Gateway tests...
echo.

wsl bash -c "cd /home/owens/CodingProjects/LLM-API-Key-Proxy && ./test_vps_gateway.sh"

echo.
echo Tests completed!
echo.
echo Press any key to close this window...
pause > nul
