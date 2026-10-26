# Synergie CrowdGPT × HiveMind : Architecture Tronc Commun + Greffons

Ce document décrit le pont technique entre **[CrowdGPT](https://github.com/Vxtzq/CrowdGPT)** (`Vxtzq/Crowd-v1`, modèle dense de 1B paramètres entraîné sur des shards binaires `uint16` FineWeb de 1 Mo) et **[HiveMind](https://github.com/xaviercallens/HiveMind)** (La Ruche Sparse MoE & routeur OpenAI-compatible).

## 1. Convergence Architecturale (Hub-and-Spoke DiLoCo)

Le cycle d'entraînement de `CrowdGPT/client.py` (téléchargement du checkpoint `bf16`, entraînement local pendant 1 heure sur des sous-chunks `uint16` avec `AdamW8bit`, puis envoi du delta compressé `_base_delta_bf16` au coordinateur) correspond exactement au schéma **Hub-and-Spoke DiLoCo (Distributed Low-Communication)**.

## 2. Brique `hivemind_bridge.py` (Zéro Dépendance Supplémentaire)

Le module autonome `hivemind_bridge.py` (utilisant uniquement `torch` et `numpy`, déjà présents dans `requirements.txt`) fournit trois briques prêtes à l'emploi pour le coordinateur CrowdGPT lorsqu'il monte à 100 contributeurs simultanés :

1. **`unpack_crowdgpt_delta_stream` / `pack_crowdgpt_delta_stream`** : lecture et écriture exactes du flux gzippé `struct.pack('<I', len(pl)) + pl + (dbf, si, sv)` généré par `client.py`.
2. **`multi_krum_select_and_aggregate`** : filtrage géométrique **Multi-Krum** par blocs mémoire bornés (`chunk_elements=1_048_576`) qui écarte automatiquement les deltas corrompus ou empoisonnés avant la moyenne pondérée par `tokensProcessed`.
3. **`LossRatchetGate`** : porte de validation monotone sur jeu held-out qui n'avance le checkpoint global (`Vxtzq/Crowd-v1`) que lorsque la loss held-out s'améliore (avec empreinte `SHA-256`).

## 3. Intégration dans le Routeur HiveMind OpenRouter & La Ruche

Le modèle `Vxtzq/Crowd-v1` est pré-câblé dans le catalogue et le routeur HiveMind sous l'identifiant **`hivemind/crowd-v1`** (alias `crowdgpt/crowd-v1` et `Vxtzq/Crowd-v1`), permettant :
- de discuter gratuitement avec `Crowd-v1` dans le playground navigateur (`0 €`, aucun abonnement) ;
- d'y greffer des adaptateurs QLoRA spécialisés de 18 Mo (`loraRank > 0`, déjà prévu dans l'en-tête JSON de `client.py`) sans diluer l'entraînement du tronc commun 1B.
