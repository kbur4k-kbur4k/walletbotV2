"""
Oracle Cloud A1.Flex kapasitesi açılır açılmaz instance oluşturur.
Telegram bildirimi gönderir.

Kullanım: python oci_retry.py
"""
import time, requests, oci, sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ── Ayarlar ──────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN   = "8464735489:AAFmkaXFdgFXzIp8DurSEvXjnpnLPK99b08"
TELEGRAM_CHAT_ID = "5010884435"
RETRY_INTERVAL   = 300  # 5 dakika

TENANCY_ID  = "ocid1.tenancy.oc1..aaaaaaaabc4uqhfbjjv3mcjigfhin36vo6vbic7tapfmgdnde33vk6dnxz4q"
IMAGE_ID    = "ocid1.image.oc1.eu-frankfurt-1.aaaaaaaaimlbvu2dnd46l4gmgpcykuuqm6v52u67tqki7hxmptppe4wdhwea"
SUBNET_ID   = "ocid1.subnet.oc1.eu-frankfurt-1.aaaaaaaarpq6l26m7cmbjcjwxgumo46cjl5loi6i4cic6wzybpmfqgsoosta"
SSH_KEY     = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIGoKjnG1Aca7v903+b2tDrmqrFPMjIn0rHOfwbZryZ2F walletbot-oracle"
ADS         = ["AD-1", "AD-2", "AD-3"]
# ─────────────────────────────────────────────────────────────────────────────


def telegram(msg: str) -> None:
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": msg},
            timeout=10,
        )
    except Exception:
        pass


def try_create(compute, ad_name: str) -> bool:
    ad_map = {
        "AD-1": f"vUeJ:EU-FRANKFURT-1-AD-1",
        "AD-2": f"vUeJ:EU-FRANKFURT-1-AD-2",
        "AD-3": f"vUeJ:EU-FRANKFURT-1-AD-3",
    }
    details = oci.core.models.LaunchInstanceDetails(
        availability_domain=ad_map[ad_name],
        compartment_id=TENANCY_ID,
        display_name="wallet-bot",
        shape="VM.Standard.A1.Flex",
        shape_config=oci.core.models.LaunchInstanceShapeConfigDetails(
            ocpus=4, memory_in_gbs=24
        ),
        create_vnic_details=oci.core.models.CreateVnicDetails(
            subnet_id=SUBNET_ID,
            assign_public_ip=True,
        ),
        metadata={"ssh_authorized_keys": SSH_KEY},
        source_details=oci.core.models.InstanceSourceViaImageDetails(
            image_id=IMAGE_ID,
            boot_volume_size_in_gbs=50,
        ),
    )
    try:
        resp = compute.launch_instance(details)
        ip_info = f"\nPublic IP birkaç dakika içinde atanacak."
        telegram(f"✅ walletbot instance OLUŞTU! ({ad_name}){ip_info}")
        print(f"[OK] Instance oluştu — {ad_name} | {resp.data.id}")
        return True
    except oci.exceptions.ServiceError as e:
        if e.status == 500 and "Out of host capacity" in str(e.message):
            print(f"[{ad_name}] Kapasite yok.")
            return False
        print(f"[{ad_name}] Hata {e.status}: {e.message}")
        return False


def main():
    config = oci.config.from_file()
    compute = oci.core.ComputeClient(config)
    print("OCI retry başladı. Ctrl+C ile durdur.")
    telegram("🔄 OCI retry başladı — kapasite açılınca otomatik oluşturulacak.")
    attempt = 0
    while True:
        attempt += 1
        print(f"\n── Deneme #{attempt} ({time.strftime('%H:%M:%S')}) ──")
        for ad in ADS:
            if try_create(compute, ad):
                return
        print(f"Tüm AD'ler dolu. {RETRY_INTERVAL//60} dk sonra tekrar...")
        time.sleep(RETRY_INTERVAL)


if __name__ == "__main__":
    main()
