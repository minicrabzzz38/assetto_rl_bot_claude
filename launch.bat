@echo off
title AC RL Bot — Claude Dashboard
cd /d "%~dp0"

echo.
echo  ========================================
echo    ASSETTO RL BOT — Claude
echo    Interface Web de controle
echo  ========================================
echo.

:: Verifier que le venv existe
if not exist "venv\Scripts\python.exe" (
    echo [SETUP] Creation de l'environnement Python...
    python -m venv venv
    echo [SETUP] Installation des dependances ^(peut prendre 2-3 min^)...
    call venv\Scripts\activate
    pip install -r requirements.txt --quiet
    pip install flask --quiet
    echo [SETUP] Installation terminee!
    echo.
)

call venv\Scripts\activate

:: Verifier Flask
venv\Scripts\python -c "import flask" 2>nul
if errorlevel 1 (
    echo [SETUP] Installation de Flask...
    pip install flask --quiet
)

:: Ouvrir le navigateur apres 2 secondes
echo [LAUNCH] Demarrage du serveur web...
echo [LAUNCH] Ouverture de http://localhost:5000 dans le navigateur...
echo.
echo  Pour arreter : fermer cette fenetre ou Ctrl+C
echo.
start "" cmd /c "timeout /t 2 /nobreak >nul && start http://localhost:5000"

:: Lancer le serveur Flask
venv\Scripts\python web\app.py

pause
