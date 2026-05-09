#!/bin/bash
# Run once (e.g. login node): build venv under $TMPDIR, pip install there, copy to Evolution/venv.
# Avoids slow NFS during unpack; final venv still lives on NFS for sbatch to source.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
module load python/3.12.11-fasrc01

TMPROOT="${TMPDIR:-/tmp}/${USER:-user}"
mkdir -p "$TMPROOT"
export TMPDIR="$TMPROOT"

STAGING=$(mktemp -d "${TMPROOT}/evolution_venv_build.XXXXXX")
trap 'rm -rf "$STAGING"' EXIT

python -m venv "$STAGING"
# shellcheck source=/dev/null
source "$STAGING/bin/activate"
python -m pip install -U pip wheel
python -m pip install -r "$ROOT/requirements.txt"
deactivate

STAGE_PATH="$STAGING"
rm -rf "$ROOT/venv"
cp -a "$STAGING" "$ROOT/venv"
trap - EXIT
rm -rf "$STAGING"

# venv/* scripts embed STAGE_PATH in shebangs; fix after copy from $TMPDIR
python3 <<PY
import pathlib
old = pathlib.Path(r"${STAGE_PATH}").as_posix()
new = (pathlib.Path(r"${ROOT}") / "venv").as_posix()
bindir = pathlib.Path(new) / "bin"
for p in bindir.iterdir():
    if not p.is_file():
        continue
    try:
        t = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        continue
    if old not in t:
        continue
    p.write_text(t.replace(old, new), encoding="utf-8")
cfg = pathlib.Path(new) / "pyvenv.cfg"
if cfg.is_file():
    t = cfg.read_text(encoding="utf-8")
    if old in t:
        cfg.write_text(t.replace(old, new), encoding="utf-8")
PY

if [[ ! -f .env ]]; then
  if [[ -f "$ROOT/../GTOGoblin/.env" ]]; then
    cp "$ROOT/../GTOGoblin/.env" "$ROOT/.env"
    chmod 600 "$ROOT/.env"
    echo "Created $ROOT/.env from GTOGoblin/.env (standalone from here on)."
  else
    echo "Create $ROOT/.env with HF_TOKEN=your_token"
    exit 1
  fi
fi
echo "Bootstrap done. Activate with: source $ROOT/venv/bin/activate"
