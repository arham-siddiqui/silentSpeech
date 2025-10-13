#!/usr/bin/env bash
set -euo pipefail
BASE="${1:-data/RVTALL}"
echo "[i] Checking ${BASE} ..."
mkdir -p "${BASE}"
if [ -d "${BASE}/Processed_sliced_data" ]; then
  echo "[OK] Found Processed_sliced_data"
else
  echo "[!] Place Processed_sliced_data.zip in ${BASE} and unzip it."
fi
if [ -d "${BASE}/RVTALL-Preprocess-main" ]; then
  echo "[OK] Found RVTALL-Preprocess-main"
else
  echo "[!] Place Code.zip in ${BASE} and unzip it."
fi
tree "${BASE}" | head -n 120 || true
