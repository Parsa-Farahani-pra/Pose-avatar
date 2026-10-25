#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PY=""
for CAND in "$HERE/.venv/bin/python" "$HERE/venv/bin/python" "$(command -v python3 || true)"; do
    if [[ -x "$CAND" ]] && "$CAND" -c "import cv2, mediapipe; mediapipe.solutions.pose" 2>/dev/null; then
        PY="$CAND"; break
    fi
done
if [[ -z "$PY" ]]; then
    echo "No working venv found." >&2
    echo "MediaPipe (mp.solutions.pose) is required. Recreate the venv:" >&2
    echo "  python3.12 -m venv .venv && .venv/bin/pip install -r requirements-lock.txt" >&2
    exit 1
fi

# Default script is pose_avatar.py.  A leading "-" means the user passed a
# flag only (e.g. "./run.sh --help"), so the default script is used.
if [[ $# -gt 0 && "$1" == -* ]]; then
    SCRIPT="pose_avatar.py"
else
    SCRIPT="${1:-pose_avatar.py}"
    shift || true
fi

# The 3D avatar additionally needs Ursina.
if [[ "$SCRIPT" == "pose_avatar.py" ]] && ! "$PY" -c "import ursina" 2>/dev/null; then
    echo "ursina is not installed in $PY -- run: $PY -m pip install 'ursina>=7.0,<8'" >&2
    exit 1
fi

exec "$PY" "$HERE/$SCRIPT" "$@"
