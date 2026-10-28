<p align="center">
  <a href="https://crowdgpt.net">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Vxtzq/CrowdGPT/main/docs/logo-dark.png">
      <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/Vxtzq/CrowdGPT/main/docs/logo-light.png">
      <img src="https://raw.githubusercontent.com/Vxtzq/CrowdGPT/main/res/logo-light.png" alt="CrowdGPT" width="100%" style="max-width: 800px; border-radius: 12px;">
    </picture>
  </a>
</p>

<p align="center">
  <a href="https://crowdgpt.net">Website</a> ·
  <a href="https://discord.gg/2Shwqj3Xpx">Discord</a> ·
  <a href="./docs/ARCHITECTURE.md">Architecture</a> ·
  <a href="./docs/DATA.md">Training Data</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge">
  <img src="https://img.shields.io/badge/Version-v1.8-green?style=for-the-badge">
  <a href="https://discord.gg/2Shwqj3Xpx"><img src="https://img.shields.io/badge/Discord-Join%20Server-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord"></a>
  <img src="https://img.shields.io/endpoint?style=for-the-badge&url=https%3A%2F%2Fhuggingface.co%2FVxtzq%2FCrowd-v1%2Fraw%2Fmain%2Fbadge_tokens.json" alt="Tokens Processed">
</p>

<p align="center">
  <strong>The best 0$ AI.</strong>
</p>

# ⚠️ CrowdGPT v1.8 Changelog ⚠️
- Revert back from broken weights (7 steps back on server)
- UI fixes
- client.py cryptographical signing
- Engram training enabled by default
- Tiered matchmaking (precision & optimizer aware)

## What is CrowdGPT?

CrowdGPT is a **project for training arbitrary big LLMs** using volunteered compute.

Instead of one datacenter doing all the work, CrowdGPT distributes training across participating machines.

**All of this working without money, API keys or any centralized GPU clusters.**

### How it works

See [ARCHITECTURE.md](https://github.com/Vxtzq/CrowdGPT/blob/main/docs/ARCHITECTURE.md)

## ⚡ Quick installation (one command)

For Linux/MacOS

```bash
curl -sSL https://raw.githubusercontent.com/Vxtzq/CrowdGPT/main/install.sh | bash
