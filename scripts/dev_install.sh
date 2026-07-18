#!/usr/bin/env bash
# dev_install.sh — link the addon into FreeCAD so edits in this repo are
# live on the next FreeCAD restart (no copying, no packaging).
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"

# Ask FreeCAD where its user data lives (e.g. ~/.local/share/FreeCAD/v1-1/)
# instead of hardcoding a path that breaks on the next FreeCAD version.
APPDATA="$(freecadcmd -c 'import FreeCAD; print(FreeCAD.getUserAppDataDir())' 2>/dev/null | tail -1)"
MOD_DIR="${APPDATA}Mod"

mkdir -p "$MOD_DIR"
ln -sfn "$REPO/PhysicsIQ" "$MOD_DIR/PhysicsIQ"
echo "linked: $MOD_DIR/PhysicsIQ -> $REPO/PhysicsIQ"
echo "restart FreeCAD and pick the 'PhysicsIQ' workbench."
