# Contributing to wallet_manager

## Getting started

```bash
git clone <repo>
cd wallet_manager
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e ".[dev]"
pre-commit install
```

## Running tests

```bash
pytest tests/ -v --cov=wallet_manager --cov-fail-under=80
```

## Code quality

```bash
ruff check .          # lint
ruff format .         # format
mypy wallet_manager/  # type check
bandit -r wallet_manager/ -c pyproject.toml -ll  # security scan
```

## Pull request checklist

- [ ] All tests pass: `pytest tests/ -v`
- [ ] Coverage does not drop below 80 %
- [ ] `ruff check .` exits with code 0
- [ ] `mypy wallet_manager/` exits with code 0
- [ ] New public functions and classes have docstrings
- [ ] `CHANGELOG.md` updated under `[Unreleased]`

## Security issues

Do **not** open a public issue for security vulnerabilities.
See [SECURITY.md](SECURITY.md) for the responsible disclosure process.

## Hard constraints — never break these

- Chain is fixed: **Base, chain ID 8453**. Do not add other chains.
- Do not modify ABC signatures in `core/interfaces.py`.
- All I/O must be `async`/`await`. No blocking calls in the event loop.
- Private keys must never reach logs (`SensitiveDataFilter` is non-negotiable).
- The `legacy` execution mode must always remain functional.
- `psutil` must always be guarded with `try/except ImportError`.
