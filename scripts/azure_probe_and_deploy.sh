#!/usr/bin/env bash
set -euo pipefail

# Probe a small set of low-cost Azure VM sizes across regions and stop at the
# first successful deployment. Designed for Azure Cloud Shell.

VM_NAME="${VM_NAME:-walletbot-vm}"
ADMIN_USER="${ADMIN_USER:-azureuser}"
IMAGE="${IMAGE:-Ubuntu2204}"
DISK_GB="${DISK_GB:-30}"

REGIONS=()
SIZES=()

if [[ -n "${REGIONS_OVERRIDE:-}" ]]; then
  read -r -a REGIONS <<< "${REGIONS_OVERRIDE}"
else
  REGIONS=(
    northeurope
    westeurope
    swedencentral
    germanywestcentral
    francecentral
    eastus2
  )
fi

if [[ -n "${SIZES_OVERRIDE:-}" ]]; then
  read -r -a SIZES <<< "${SIZES_OVERRIDE}"
else
  SIZES=(
    Standard_D2as_v5
    Standard_D2ads_v5
    Standard_D2s_v3
    Standard_A2_v2
    Standard_A2m_v2
    Standard_B2s
    Standard_B2ms
    Standard_B1ms
  )
fi

echo "Using subscription:"
az account show --query "{name:name,id:id}" -o table
echo

for region in "${REGIONS[@]}"; do
  rg="walletbot-rg-${region}"
  echo "==> Region: ${region}"
  az group create --name "${rg}" --location "${region}" --output none

  for size in "${SIZES[@]}"; do
    echo "   -> Trying size: ${size}"
    if az vm create \
      --resource-group "${rg}" \
      --name "${VM_NAME}" \
      --image "${IMAGE}" \
      --size "${size}" \
      --admin-username "${ADMIN_USER}" \
      --generate-ssh-keys \
      --public-ip-sku Standard \
      --os-disk-size-gb "${DISK_GB}" \
      --output jsonc; then
      echo
      echo "SUCCESS: ${region} / ${size}"
      echo "Resource group: ${rg}"
      exit 0
    fi

    echo "   x Failed: ${region} / ${size}"
    az vm delete --resource-group "${rg}" --name "${VM_NAME}" --yes --no-wait >/dev/null 2>&1 || true
  done

  echo "No candidate worked in ${region}."
  echo
done

echo "No deployment succeeded in the candidate region/size list."
echo "Tip: try adding more regions with REGIONS_OVERRIDE or different SKUs with SIZES_OVERRIDE."
exit 1
