# Azure Runtime

This file is the canonical operational handoff for Azure access and runtime truth.

## Current Azure Target

- VM name: `walletbot-vm`
- resource group: `walletbot-rg-westeurope`
- region: `West Europe`
- public IP: `51.124.156.116`
- SSH user: `azureuser`
- local SSH key path: `C:\Users\ASUS\.ssh\id_rsa`

## Primary Endpoints

- dashboard: `http://51.124.156.116:9090/`
- dashboard direct: `http://51.124.156.116:9090/dashboard`
- health: `http://51.124.156.116:9090/health`
- runtime API: `http://51.124.156.116:9090/api/runtime`
- research API: `http://51.124.156.116:9090/api/research`
- metrics: `http://51.124.156.116:9090/metrics`

## Canonical Access Commands

```powershell
ssh -i C:\Users\ASUS\.ssh\id_rsa azureuser@51.124.156.116
```

```powershell
ssh -i C:\Users\ASUS\.ssh\id_rsa azureuser@51.124.156.116 "cd ~/walletbot && sudo docker compose ps"
```

```powershell
ssh -i C:\Users\ASUS\.ssh\id_rsa azureuser@51.124.156.116 "cd ~/walletbot && sudo docker compose logs --tail 120 wallet_manager"
```

```powershell
curl.exe -s http://51.124.156.116:9090/health
curl.exe -s http://51.124.156.116:9090/api/runtime
curl.exe -s http://51.124.156.116:9090/api/research
```

## Last Confirmed Runtime Truth

Confirmed on 2026-04-04:

- health endpoint returned `ok`
- runtime API responded
- research API responded
- execution mode reported as `robust_paper`
- live trading guard inactive

## Last Confirmed Behavioral Findings

High-confidence observed no-trade reasons:

- `RISK_BLOCKED(normal)`
- `RISK_COMMITTEE:ModelDisagreement`
- `TRADE_BUDGET_TOO_SMALL`

Recent runtime logs also showed:

- occasional stress-test rejection
- metrics server bind warning on port `9090`

## Deployment Rule

If Azure runtime facts change, update this file in the same work session.

Do not store secret contents here.
