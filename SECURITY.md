# How CrowdGPT signed updates work

CrowdGPT clients update themselves from this repository. To make sure nobody (a hijacked GitHub account, a
bad merge, a stolen token) can push code that runs on your machine, **updates are cryptographically signed**.
A client only installs an update that was signed by the maintainer's private key.

## The idea in 30 seconds

| Piece | What it is | Where it lives |
|---|---|---|
| **Private key** | The maintainer's secret. Only it can create valid signatures. | Maintainer's computer only, passphrase-protected. Never on GitHub. |
| **Public key** | Can *check* a signature but not make one. | Built into `client.py` (`UPDATE_PUBKEYS`). |
| **`docs/version.txt`** | The release manifest: version, SHA-256 of `client.py`, SHA-256 of `client.html`, expiry date. | This repo |
| **`docs/version.txt.sig`** | The signature of `version.txt`. | This repo |

A SHA-256 hash is a fingerprint of a file: change one character and the fingerprint changes completely.
The signature covers `version.txt`, so it covers those fingerprints. It says: *"I approve exactly these file contents."*

## What a client does on every update check

1. Download `version.txt` and `version.txt.sig`.
2. Verify the signature with the public key(s) built into the running client. **Invalid or missing: stop.**
3. Check the signed `expires:` date hasn't passed (stops replay of an old release).
4. Check the version is newer than the running one (no downgrades).
5. Download `client.py` and `client.html` and check each matches the SHA-256 in the signed manifest. **Mismatch: stop.**
6. Install (with backup and rollback), then restart.

An attacker who edits `client.py` breaks the hash. If they also edit the hash in `version.txt`, the signature breaks.
To pass both they would need the private key.

## Why not just trust the hash?

Hashes in the same repo as the code protect against corrupted downloads, not attackers: whoever can change the code
can change the hash too. The signature is the part an attacker cannot forge.

## Two keys

`UPDATE_PUBKEYS` lists a **main** key and a **recovery** key. A client accepts a release signed by either.
The recovery key stays offline and is only used if the main key is lost or stolen (a release signed with it can
drop the compromised key from the list). Keys must be in clients *before* they are needed.

## For the maintainer: releasing

```bash
pip install cryptography                      # on your own computer only
python3 tools/sign_release.py gen-key --out ~/release.pem     # once
python3 tools/sign_release.py gen-key --out ~/recovery.pem    # once; store it offline
# paste both printed lines into UPDATE_PUBKEYS in client.py

# every release:
#   1. bump CLIENT_VERSION in client.py
#   2. python3 sign.py            (hashes both files, signs, self-verifies; asks for the passphrase once)
#   3. git add client.py client.html docs/version.txt docs/version.txt.sig && git commit && git push
```

**Merge nothing into `client.py` / `client.html` after signing.** Any change, even one character, invalidates the
signature and clients will (safely) refuse the update until you sign again.

## What this does and does not protect against

Protects against: a compromised GitHub account/token, a malicious or mistaken commit to `main`, tampering of the
download in transit. Pushing to the repo is no longer enough to reach users.

Does **not** protect against: theft of the private key *and* its passphrase; a malicious change that the
maintainer reviews carelessly and then signs; clients older than the first signed release (they still trust `main`
until they update once); other code paths such as `install.sh` / `install.bat` or `trust_remote_code`.

## For users

Nothing to do. If an update is refused you'll see `Update refused: ...` in the log and the client keeps running
the version you have. Setting `UPDATE_PUBKEYS = []` makes a client refuse all updates.
