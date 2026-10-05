#!/usr/bin/env bash
set -Eeuo pipefail

# Install and run Claw AI Lab on a Linux server.
# Usage: CLAW_LLM_BASE_URL=https://provider.example/v1 ./deploy/install_server.sh MODEL

umask 077

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:-${CLAW_LLM_MODEL:-}}"
API_KEY="${CLAW_LLM_API_KEY:-}"

if [[ -z "$MODEL" ]]; then
  echo "Usage: $0 MODEL (API key via CLAW_LLM_API_KEY or interactive prompt)" >&2
  exit 2
fi
if [[ -z "$API_KEY" ]]; then
  read -rsp "API key: " API_KEY
  echo
fi
[[ -n "$API_KEY" ]] || { echo "API key is required" >&2; exit 2; }

RUNTIME_ROOT="$ROOT/.runtime"
mkdir -p "$RUNTIME_ROOT"

download() {
  local url="$1" destination="$2"
  command -v wget >/dev/null || { echo "wget is required to bootstrap runtimes" >&2; exit 1; }
  echo "Downloading $(basename "$destination")..."
  wget -q --show-progress -O "$destination" "$url"
}

# The server may only have an older system Python and no sudo access. Keep the
# runtime inside the repository so setup remains reproducible and user-scoped.
PYTHON=""
if command -v python3 >/dev/null; then
  if python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)'; then
    PYTHON="$(command -v python3)"
  fi
fi
if [[ -z "$PYTHON" ]]; then
  MINIFORGE="$RUNTIME_ROOT/miniforge3"
  if [[ ! -x "$MINIFORGE/bin/conda" ]]; then
    MINIFORGE_INSTALLER="$RUNTIME_ROOT/Miniforge3.sh"
    if [[ ! -s "$MINIFORGE_INSTALLER" ]]; then
      download "${MINIFORGE_URL:-https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh}" "$MINIFORGE_INSTALLER"
    fi
    bash "$MINIFORGE_INSTALLER" -b -p "$MINIFORGE"
    rm -f "$MINIFORGE_INSTALLER"
  fi
  PY_ENV="$RUNTIME_ROOT/python"
  if [[ ! -x "$PY_ENV/bin/python" ]]; then
    "$MINIFORGE/bin/conda" create -y -p "$PY_ENV" python=3.11
  fi
  PYTHON="$PY_ENV/bin/python"
fi

NODE=""
if command -v node >/dev/null; then
  NODE_MAJOR="$(node -p 'process.versions.node.split(".")[0]')"
  NODE_MINOR="$(node -p 'process.versions.node.split(".")[1]')"
  if (( (NODE_MAJOR == 20 && NODE_MINOR >= 19) || (NODE_MAJOR == 22 && NODE_MINOR >= 12) || NODE_MAJOR > 22 )); then
    NODE="$(command -v node)"
  fi
fi
if [[ -z "$NODE" ]]; then
  NODE_VERSION="${NODE_VERSION:-20.19.5}"
  NODE_DIR="$RUNTIME_ROOT/node-v$NODE_VERSION-linux-x64"
  NODE_ARCHIVE="$RUNTIME_ROOT/node-v$NODE_VERSION-linux-x64.tar.xz"
  if [[ ! -x "$NODE_DIR/bin/node" ]]; then
    download "${NODE_URL:-https://nodejs.org/dist/v$NODE_VERSION/node-v$NODE_VERSION-linux-x64.tar.xz}" "$NODE_ARCHIVE"
    tar -xJf "$NODE_ARCHIVE" -C "$RUNTIME_ROOT"
    rm -f "$NODE_ARCHIVE"
  fi
  NODE="$NODE_DIR/bin/node"
  export PATH="$NODE_DIR/bin:$PATH"
fi
NPM="$(dirname "$NODE")/npm"
[[ -x "$NPM" ]] || { echo "npm was not found beside $NODE" >&2; exit 1; }

# Keep this instance separate from other users' copies on a shared server.
if [[ -z "${FRONTEND_PORT:-}" && -z "${RESOURCE_MONITOR_PORT:-}" && -z "${AGENT_BRIDGE_PORT:-}" ]]; then
  OFFSET=0
  while ss -ltn 2>/dev/null | grep -Eq ":$((5903 + OFFSET)) |:$((8905 + OFFSET)) |:$((8906 + OFFSET)) "; do
    OFFSET=$((OFFSET + 10))
  done
  FRONTEND_PORT=$((5903 + OFFSET))
  RESOURCE_MONITOR_PORT=$((8905 + OFFSET))
  AGENT_BRIDGE_PORT=$((8906 + OFFSET))
fi
FRONTEND_PORT="${FRONTEND_PORT:-5903}"
RESOURCE_MONITOR_PORT="${RESOURCE_MONITOR_PORT:-8905}"
AGENT_BRIDGE_PORT="${AGENT_BRIDGE_PORT:-8906}"
if [[ -z "${TOTAL_GPUS:-}" ]]; then
  TOTAL_GPUS=0
  if command -v nvidia-smi >/dev/null 2>&1; then
    TOTAL_GPUS="$(nvidia-smi -L | wc -l)"
  fi
fi

# Infer the standard endpoint from well-known model names. For a private or
# proxy endpoint, set CLAW_LLM_BASE_URL before running this script.
PROVIDER="openai-compatible"
BASE_URL="${CLAW_LLM_BASE_URL:-}"
case "$MODEL" in
  claude-*)
    PROVIDER="anthropic"
    BASE_URL="${BASE_URL:-https://api.anthropic.com}"
    ;;
  deepseek-*)
    PROVIDER="deepseek"
    BASE_URL="${BASE_URL:-https://api.deepseek.com/v1}"
    ;;
  gpt-*|o[1-9]*)
    PROVIDER="openai"
    BASE_URL="${BASE_URL:-https://api.openai.com/v1}"
    ;;
  *)
    BASE_URL="${BASE_URL:-https://api.openai.com/v1}"
    ;;
esac

VENV="$ROOT/.venv"
if [[ ! -x "$VENV/bin/python" ]]; then
  "$PYTHON" -m venv "$VENV"
fi
if [[ "${INSTALL_BACKEND_DEPS:-1}" == "1" ]]; then
  "$VENV/bin/python" -m pip install --upgrade pip
  "$VENV/bin/python" -m pip install -e "$ROOT/backend/agent[all]" websockets psutil
fi

if [[ "${INSTALL_ML_DEPS:-1}" == "1" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    "$VENV/bin/python" -m pip install torch==2.3.1 torchvision==0.18.1 \
      'numpy<2' nvidia-nvjitlink-cu12==12.1.105
  else
    "$VENV/bin/python" -m pip install torch torchvision
  fi
  "$VENV/bin/python" -m pip install \
    'numpy<2' diffusers==0.30.3 transformers==4.46.3 accelerate==1.2.1 safetensors datasets \
    huggingface_hub 'opencv-python<4.12' pandas matplotlib scikit-image scipy einops tqdm
fi

if [[ "${INSTALL_FRONTEND_DEPS:-1}" == "1" ]]; then
  cd "$ROOT/frontend"
  "$NPM" ci
  cd "$ROOT"
fi

export CLAW_ROOT="$ROOT"
export CLAW_MODEL="$MODEL"
export CLAW_API_KEY="$API_KEY"
export CLAW_PROVIDER="$PROVIDER"
export CLAW_BASE_URL="$BASE_URL"
export CLAW_PYTHON="$VENV/bin/python"

# Use the repository's YAML parser so quoting and existing settings remain valid.
"$VENV/bin/python" - <<'PY'
import os
from pathlib import Path
import yaml

root = Path(os.environ["CLAW_ROOT"])
template = root / "examples" / "config_template.yaml"
with template.open(encoding="utf-8") as handle:
    config = yaml.safe_load(handle) or {}

llm = config.setdefault("llm", {})
model = os.environ["CLAW_MODEL"]
llm.update({
    "provider": os.environ["CLAW_PROVIDER"],
    "base_url": os.environ["CLAW_BASE_URL"],
    "api_key": "",
    "api_key_env": "RESEARCHCLAW_API_KEY",
    "primary_model": model,
    "coding_model": model,
    "image_model": os.environ.get("CLAW_IMAGE_MODEL", "gpt-image-1"),
    "fallback_models": [],
})
config.setdefault("experiment", {}).setdefault("sandbox", {})["python_path"] = os.environ["CLAW_PYTHON"]
config.setdefault("experiment", {})["mode"] = "sandbox"

with template.open("w", encoding="utf-8") as handle:
    yaml.safe_dump(config, handle, allow_unicode=False, sort_keys=False)
PY

API_KEY_ESCAPED="$(printf '%q' "$API_KEY")"
MODEL_ESCAPED="$(printf '%q' "$MODEL")"
BASE_URL_ESCAPED="$(printf '%q' "$BASE_URL")"
cat > "$ROOT/.env" <<EOF
RESEARCHCLAW_API_KEY=$API_KEY_ESCAPED
RESEARCHCLAW_MODEL=$MODEL_ESCAPED
RESEARCHCLAW_BASE_URL=$BASE_URL_ESCAPED
PYTHON_PATH=$VENV/bin/python
DISCUSSION_MODELS=$MODEL_ESCAPED
FRONTEND_PORT=$FRONTEND_PORT
RESOURCE_MONITOR_PORT=$RESOURCE_MONITOR_PORT
AGENT_BRIDGE_PORT=$AGENT_BRIDGE_PORT
TOTAL_GPUS=$TOTAL_GPUS
EOF
chmod 600 "$ROOT/.env"

cat > "$ROOT/deploy/run.sh" <<'EOF'
#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for RUNTIME_NODE in "$ROOT"/.runtime/node-v*/bin; do
  if [[ -d "$RUNTIME_NODE" ]]; then
    export PATH="$RUNTIME_NODE:$PATH"
    break
  fi
done
if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi
export PYTHON_PATH="${PYTHON_PATH:-$ROOT/.venv/bin/python}"
export DISCUSSION_MODELS="${DISCUSSION_MODELS:-${RESEARCHCLAW_MODEL:-gpt-4o}}"
exec "$ROOT/start.sh" "${1:-start}"
EOF
chmod 700 "$ROOT/deploy/run.sh"

echo "Installation complete. Starting Claw AI Lab..."
"$ROOT/deploy/run.sh" start
echo "Access via SSH port forwarding: ssh -N -L ${FRONTEND_PORT}:127.0.0.1:${FRONTEND_PORT} USER@SERVER"
echo "Then open http://localhost:${FRONTEND_PORT}/"
