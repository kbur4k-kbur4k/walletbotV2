@echo off
chcp 65001 >nul
title WalletBot - Calisiyor

cd /d C:\Users\ASUS\OneDrive\Desktop\walletbot-fixed

call C:\Users\ASUS\anaconda3\Scripts\activate.bat walletbot
if errorlevel 1 (
    echo HATA: walletbot conda ortami bulunamadi!
    pause
    exit /b 1
)

set KMP_DUPLICATE_LIB_OK=TRUE
set PYTHONPATH=%cd%

echo.
echo ============================================================
echo  WALLETBOT BASLATILIYOR
echo  %date% %time%
echo ============================================================
echo.

python scripts/setup_wizard.py --validate-only
if errorlevel 1 (
    echo.
    echo HATA: kurulum dogrulamasi basarisiz. Lutfen yukaridaki hatalari duzelt.
    pause
    exit /b 1
)

python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
echo.

python -m wallet_manager.orchestrator

echo.
echo ============================================================
echo Bot durdu. Cikti icin bot_log.txt dosyasina bak.
echo ============================================================
pause
