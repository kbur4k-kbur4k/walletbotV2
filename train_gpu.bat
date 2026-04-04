@echo off
chcp 65001 >nul
title WalletBot - GPU Training

cd /d C:\Users\ASUS\OneDrive\Desktop\walletbot-fixed

:: walletbot conda ortamini aktif et
call C:\Users\ASUS\anaconda3\Scripts\activate.bat walletbot

:: OpenMP cakismasini engelle
set KMP_DUPLICATE_LIB_OK=TRUE

echo.
python -c "import torch; gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'YOK'; print(f'GPU: {gpu} | PyTorch: {torch.__version__}')"
echo CPCV esigi: 0.45 (wallet_manager/.env)
echo.
echo Training basliyor...
echo.

python scripts/train_models.py 2>&1

echo.
echo ============================================================
echo Training tamamlandi.
echo ============================================================
pause
