#!/usr/bin/env bash
set -euo pipefail
echo "[i] Removing parent .git to avoid submodule/orphan errors..."
cd "$(dirname "$0")"/..
PARENT="$(pwd)/.."
if [ -d "${PARENT}/.git" ]; then
  rm -rf "${PARENT}/.git"
  echo "[OK] Removed ${PARENT}/.git"
else
  echo "[i] Parent .git not found."
fi
echo "[i] Reinitializing here..."
git init
git add .
git commit -m "portable universal alpha"
git branch -M main
echo "[i] Now set your remote, e.g.:"
echo "  git remote add origin https://github.com/achiii800/mri"
echo "  git push -u origin main"

#https://github.com/achiii800/mri

# https://github.com/arham-siddiqui/silentSpeech.git

