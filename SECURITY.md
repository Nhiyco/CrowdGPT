# Releasing CrowdGPT securely (signed auto-updates)

Running clients only install an update if `docs/version.txt` carries a valid Ed25519 signature
(`docs/version.txt.sig`) from a key embedded in `client.py`, **and** `client.py` and `client.html` hash to the SHA-256 values written
in that signed file. Pushing to `main` (even from a hijacked account) therefore cannot reach clients.

## One-time setup
1. `pip install cryptography` (on your dev machine only).
2. `python tools/sign_release.py gen-key --out ~/secure/crowdgpt_release.pem` — enter a passphrase.
3. Run it a second time with another `--out` for the **recovery** key; store that one somewhere different
   (offline). Paste both printed lines into `UPDATE_PUBKEYS` in `client.py`.
4. Keep private keys **off GitHub** (no Actions secrets, no repo). Back them up.

## Every release
1. Bump `CLIENT_VERSION` in `client.py`, commit your changes.
2. `python tools/sign_release.py release --key ~/secure/crowdgpt_release.pem`
3. `python tools/sign_release.py verify` (must print OK), then commit `client.py client.html docs/version.txt docs/version.txt.sig`, push.

A push that changes `client.py` without a fresh signature is simply ignored by clients.
Re-sign at least every 180 days (`expires:`), or pass `--no-expiry` to opt out of replay protection.

## Rotation / loss
* Rotate: release a client signed with the old key whose `UPDATE_PUBKEYS` lists the new key.
* Lost primary key: sign with the recovery key (any listed key is accepted).
* Both lost: users must update `client.py` manually once.
