@echo off
REM Quick access to map visualization tools

echo.
echo ========================================
echo   DIABLO 2 BOT - MAP VISUALIZER
echo ========================================
echo.
echo Choose an option:
echo.
echo 1. Generate all visualizations
echo 2. Interactive explorer
echo 3. Quick overview
echo 4. Monitor maps (real-time)
echo 5. Show statistics
echo 6. List zones
echo 0. Exit
echo.
echo ========================================
echo.

set /p choice="Enter choice: "

if "%choice%"=="1" (
    echo.
    echo Generating visualizations...
    .venv\Scripts\python.exe visualize_maps.py
    pause
    goto :end
)

if "%choice%"=="2" (
    echo.
    echo Starting interactive explorer...
    .venv\Scripts\python.exe explore_maps.py
    goto :end
)

if "%choice%"=="3" (
    echo.
    echo Showing overview...
    .venv\Scripts\python.exe show_maps.py
    pause
    goto :end
)

if "%choice%"=="4" (
    echo.
    echo Starting real-time monitor...
    echo Press Ctrl+C to stop
    .venv\Scripts\python.exe watch_maps.py
    goto :end
)

if "%choice%"=="5" (
    echo.
    echo Showing statistics...
    .venv\Scripts\python.exe explore_maps.py --stats
    pause
    goto :end
)

if "%choice%"=="6" (
    echo.
    echo Listing zones...
    .venv\Scripts\python.exe explore_maps.py --list
    pause
    goto :end
)

if "%choice%"=="0" (
    goto :end
)

echo.
echo Invalid choice!
pause

:end
