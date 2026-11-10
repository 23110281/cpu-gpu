#!/usr/bin/env bash
# scripts/cpu/backup_results.sh [MACHINE_NAME]
# Simple, autonomous GitHub backup for this machine's CSV results.

set -euo pipefail

# ── 1. Your Personal Backup Repository Credentials ──────────────────────────
GH_USER="${BACKUP_GH_USER:-KushalMehta17}"
GH_REPO="${BACKUP_GH_REPO:-cpu-gpu-proj-course-backup}"

# Read token securely from environment variable or local ~/.git_backup_token
TOKEN_PATH="${HOME}/.git_backup_token"
if [ ! -f "$TOKEN_PATH" ] && [ -n "${SUDO_USER:-}" ]; then
    # When executed via sudo, also check the real user's home folder
    [ -f "/users/${SUDO_USER}/.git_backup_token" ] && TOKEN_PATH="/users/${SUDO_USER}/.git_backup_token"
    [ -f "/home/${SUDO_USER}/.git_backup_token" ] && TOKEN_PATH="/home/${SUDO_USER}/.git_backup_token"
fi

GH_TOKEN="${BACKUP_GH_TOKEN:-}"
if [ -z "$GH_TOKEN" ] && [ -f "$TOKEN_PATH" ]; then
    GH_TOKEN=$(cat "$TOKEN_PATH" 2>/dev/null || echo "")
fi

# Exit cleanly if credentials are not configured on this machine
if [ -z "$GH_TOKEN" ] || [ -z "$GH_USER" ]; then
    echo "[BACKUP] Backup skipped (No token found in BACKUP_GH_TOKEN or ~/.git_backup_token)."
    exit 0
fi

# ── 2. Determine Machine Name ───────────────────────────────────────────────
MACHINE_NAME="${1:-}"
if [ -z "$MACHINE_NAME" ]; then
    MACHINE_NAME=$(lscpu | grep "Model name:" | sed 's/Model name:[ \t]*//' | tr -s ' ' '_' | tr -d '()' | tr -d '@')
fi
[ -n "$MACHINE_NAME" ] || MACHINE_NAME="Unknown_CPU"

LOCAL_DATA_DIR="results/data/native/cpu/${MACHINE_NAME}"
if [ ! -d "$LOCAL_DATA_DIR" ]; then
    echo "[BACKUP] No data directory found at $LOCAL_DATA_DIR. Nothing to backup."
    exit 0
fi

echo "======================================================================"
echo " [AUTONOMOUS BACKUP] Syncing ${MACHINE_NAME} to GitHub: ${GH_REPO}"
echo "======================================================================"

TMP_DIR="/tmp/backup_${MACHINE_NAME}_$$"
REPO_URL="https://${GH_TOKEN}@github.com/${GH_USER}/${GH_REPO}.git"

cleanup() { rm -rf "$TMP_DIR"; }
trap cleanup EXIT

# ── 3. Clone Repository Silently ────────────────────────────────────────────
git clone --depth 1 "$REPO_URL" "$TMP_DIR" >/dev/null 2>&1 || {
    echo "[BACKUP ERROR] Could not clone backup repo. Please verify GH_USER and GH_TOKEN."
    exit 0
}

# ── 4. Copy ONLY This Machine's Folder ──────────────────────────────────────
mkdir -p "${TMP_DIR}/data/${MACHINE_NAME}"
cp -r "${LOCAL_DATA_DIR}"/* "${TMP_DIR}/data/${MACHINE_NAME}/"

# ── 5. Pull & Push with Concurrency Handling ────────────────────────────────
cd "$TMP_DIR"
git config user.name "CloudLab Benchmark"
git config user.email "cloudlab-backup@noreply.github.com"

git add "data/${MACHINE_NAME}"

# If nothing changed, exit cleanly
if git diff --staged --quiet; then
    echo "[BACKUP INFO] Backup repository is already up-to-date for ${MACHINE_NAME}."
    exit 0
fi

git commit -m "Auto-backup for ${MACHINE_NAME} from $(hostname) [$(date '+%Y-%m-%d %H:%M:%S')]" >/dev/null 2>&1

# Pull with rebase & push (retry up to 3 times if another machine pushes simultaneously)
PUSHED=0
for attempt in 1 2 3; do
    git pull --rebase origin main >/dev/null 2>&1 || true
    if git push origin main >/dev/null 2>&1; then
        PUSHED=1
        break
    fi
    sleep 3
done

if [ $PUSHED -eq 1 ]; then
    echo " [BACKUP SUCCESS] Successfully backed up ${MACHINE_NAME} to GitHub!"
    echo " View your data at: https://github.com/${GH_USER}/${GH_REPO}/tree/main/data/${MACHINE_NAME}"
else
    echo "[BACKUP WARN] Git push encountered a conflict, but local CSVs remain intact."
fi
echo "======================================================================"
