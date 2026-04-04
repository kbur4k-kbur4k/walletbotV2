# Security Policy

## Supported versions

| Version | Status |
|---------|--------|
| 2.x | Active — receives security fixes |
| 1.x | End of life — no further updates |

## Reporting a vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Report security issues through GitHub's private advisory feature:
`https://github.com/<owner>/wallet_manager/security/advisories/new`

Please include:
- A clear description of the vulnerability
- Steps to reproduce
- Potential impact assessment
- A suggested fix (optional)

You will receive an acknowledgement within **72 hours** and a status update
within **7 days**.

## Security architecture

| Control | Implementation |
|---------|----------------|
| Private key encryption | AES-256 Fernet (`security/vault.py`) |
| Log sanitisation | `SensitiveDataFilter` applied to every logger via `build_logger()` |
| Key rotation | `Vault.rotate_key()` — atomic file replacement, no partial state |
| Credential loading | Environment variables / `.env` file (never hardcoded) |
| Telegram token masking | Redacted in all log output |

The `.env` file is listed in `.gitignore` and `.dockerignore` and must
**never** be committed or included in Docker images.
