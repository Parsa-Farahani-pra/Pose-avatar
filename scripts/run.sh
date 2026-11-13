#!/usr/bin/env bash
set -euo pipefail

# Get the project root directory (parent of scripts/)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Check for venv in multiple common locations
PY=""
for CAND in \
    "$PROJECT_ROOT/.venv/bin/python" \
    "$PROJECT_ROOT/venv/bin/python" \
    "$HOME/.venv/bin/python" \
    "/venv/bin/python" \
    "$(command -v python3 || true)"
do
    # Check if python exists and is executable
    if [[ -x "$CAND" ]]; then
        # Test imports - try with 'mediapipe.solutions.pose' specifically
        if "$CAND" -c "import cv2, mediapipe; mediapipe.solutions.pose" 2>/dev/null || \
           "$CAND" -c "import cv2; import mediapipe; mediapipe.solutions.pose" 2>/dev/null; then
            PY="$CAND"
            break
        fi
    fi
done

if [[ -z "$PY" ]]; then
    echo "No working venv found." >&2
    echo "MediaPipe (mp.solutions.pose) is required. Recreate the venv:" >&2
    echo "  cd $PROJECT_ROOT" >&2
    echo "  python3.12 -m venv .venv" >&2
    echo "  .venv/bin/pip install -r requirements/requirements-lock.txt" >&2
    exit 1
fi

# Default script is pose_avatar.py. A leading "-" means the user passed a flag only
# (e.g. "./run.sh --help"), so the default script is used.
if [[ $# -gt 0 && "$1" == -* ]]; then
    SCRIPT="$PROJECT_ROOT/src/pose_avatar.py"
else
    SCRIPT="$PROJECT_ROOT/src/${1:-pose_avatar.py}"
    shift || true
fi

# The 3D avatar additionally needs Ursina.
if [[ "$SCRIPT" == "$PROJECT_ROOT/src/pose_avatar.py" ]] && ! "$PY" -c "import ursina" 2>/dev/null; then
    echo "Ursina is not installed in $PY" >&2
    echo "Run: $PY -m pip install 'ursina>=7.0,<8'" >&2
    exit 1
fi

# Run the script from the project root so src/ imports work
cd "$PROJECT_ROOT"
CPPATH="$PROJECT_ROOT"
export PYTHONPATH="$CPPATH:${PYTHONPATH:-}"
exec "$PY" "$SCRIPT" "$@"