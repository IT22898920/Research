@echo off
echo Setting up adb reverse for all devices...
for /f "tokens=1" %%d in ('adb devices ^| findstr /v "List" ^| findstr "device"') do (
    echo Setting up %%d...
    adb -s %%d reverse tcp:5000 tcp:5000
    adb -s %%d reverse tcp:8081 tcp:8081
)
echo.
echo Done! Ports 5000 and 8081 forwarded.
echo Run this again if connection drops.
pause
