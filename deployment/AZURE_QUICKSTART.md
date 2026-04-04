# Azure Quickstart

Bu repo Docker ile rahat deploy olur. `docker-compose.yml` icindeki kaynak limiti
`1 CPU / 2 GB RAM` oldugu icin ilk hedef 2 GB ve ustu herhangi bir Linux VM
acmak olmali; sonra gerekirse upscale edilir.

## 1. Cloud Shell icin otomatik probe

Cloud Shell'de yeni bir dosya olusturup `scripts/azure_probe_and_deploy.sh`
icerigini yapistir, sonra su komutla calistir:

```bash
bash azure_probe_and_deploy.sh
```

Bu script su kombinasyonlari sirayla dener:

- Region: `northeurope`, `westeurope`, `swedencentral`, `germanywestcentral`, `francecentral`, `eastus2`
- Size: `Standard_D2as_v5`, `Standard_D2ads_v5`, `Standard_D2s_v3`, `Standard_A2_v2`, `Standard_A2m_v2`, `Standard_B2s`, `Standard_B2ms`, `Standard_B1ms`

Ilk basarili olusan VM'de durur.

## 2. Elle calistirma

Repo makinedeyse:

```bash
bash scripts/azure_probe_and_deploy.sh
```

Istersen listeyi override edebilirsin:

```bash
REGIONS_OVERRIDE="northeurope westeurope eastus2" \
SIZES_OVERRIDE="Standard_D2as_v5 Standard_D2s_v3 Standard_B1ms" \
bash scripts/azure_probe_and_deploy.sh
```

## 3. Neden bu siralama?

- `B` serisi ucuz ama Free Trial'da en cok kapasite ve kota sorunu cikaran aile.
- `D2*` ve `A2*` serileri cogu subscription'da daha kolay aciliyor.
- `B1ms` son care; RAM dusuk ama botu `robust_paper` modunda kaldirmak icin
  gecici olarak yeterli olabilir.

## 4. VM actiktan sonra

1. `ssh azureuser@<PUBLIC_IP>`
2. Docker kur
3. Repo'yu kopyala
4. `.env` ve `wallets.json` dosyalarini yerlestir
5. `docker compose up -d --build`

## 5. Onemli not

Azure `az vm list-skus` ciktisi tek basina kapasite garantisi vermez. Bu yuzden
script gercek `az vm create` denemesi yapar ve ilk basarida durur.
