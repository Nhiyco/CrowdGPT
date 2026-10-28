#!/usr/bin/env python3
"""One command to sign a CrowdGPT release:   python3 sign.py   [--key ~/release.pem]
Hashes client.py + client.html, writes docs/version.txt, signs it, and self-verifies. Asks for your passphrase once."""
import runpy, sys
from pathlib import Path
sys.argv = [sys.argv[0], "release"] + sys.argv[1:]
runpy.run_path(str(Path(__file__).resolve().parent / "tools" / "sign_release.py"), run_name="__main__")
