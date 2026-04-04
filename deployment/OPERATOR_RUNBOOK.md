# WalletBot Operator Runbook

## 1. Bootstrap

```bash
pip install -r requirements.txt
python scripts/setup_wizard.py
```

Bu adım:
- `.env.example` dosyasindan `.env` uretir
- `wallets.json.example` dosyasindan `wallets.json` uretir
- `wallet_manager/.env` dosyasini root `.env` ile senkron tutar
- mevcut konfigrasyonu dogrular

## 2. Minimum safe startup

`.env` icin ilk guvenli ayarlar:

```dotenv
EXECUTION_MODE=robust_paper
PAPER_TRADE=true
LIVE_TRADING_ENABLED=false
LIVE_TINY_CAP_MODE=true
LIVE_MAX_TOTAL_CAPITAL_USD=250.0
LIVE_MAX_WALLETS=1
```

## 3. Start

```bash
python -m wallet_manager.orchestrator
```

Windows kullaniyorsan:

```bat
botu_baslat.bat
```

Bu betik artik startup oncesi `scripts/setup_wizard.py --validate-only`
calistirir. Konfigrasyonda kritik hata varsa bot baslamaz.

## 4. Runtime surfaces

- Metrics: `http://localhost:9090/metrics`
- Health: `http://localhost:9090/health`
- Runtime API: `http://localhost:9090/api/runtime`
- Dashboard: `http://localhost:9090/dashboard`

## 5. Live guard

Canli moda gecis icin su kosullar birlikte kontrol edilir:

- `LIVE_TRADING_ENABLED=true`
- wallet sayisi `LIVE_MAX_WALLETS` limitini asmamali
- toplam gozlenen sermaye `LIVE_MAX_TOTAL_CAPITAL_USD` limitini asmamali
- gerekli paper trade esigi gecilmis olmali

Bu bilgiler runtime audit snapshot icinde saklanir ve dashboard/API tarafinda
gorunur.

## 6. Audit and rollback

- Son session audit kayitlari analytics DB icinde tutulur
- Hata halinde orchestrator failure ozeti ve fatal error bilgisini audit notuna yazar
- Cuzdan anahtari migrasyonu `wallets.json.bak` olusturarak ilerler

## 7. Before real capital

- `scripts/setup_wizard.py --validate-only`
- `pytest tests/ -q`
- `/dashboard` uzerinden guard ve paper gate durumunu kontrol et
- kucuk sermaye ve tek wallet ile basla
