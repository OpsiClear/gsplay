#!/bin/bash
# Build frontend and copy to package static directory

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$SCRIPT_DIR/frontend"
STATIC_DIR="$SCRIPT_DIR/gsplay_launcher/static"

echo "Building frontend..."
cd "$FRONTEND_DIR"

# Check if deno is available
if command -v deno &> /dev/null; then
    deno task build
elif [ -f "$HOME/.deno/bin/deno" ]; then
    "$HOME/.deno/bin/deno" task build
else
    echo "Error: deno not found. Please install deno first."
    exit 1
fi

echo "Copying to package static directory..."
rm -rf "$STATIC_DIR"
mkdir -p "$STATIC_DIR"
cp -r "$FRONTEND_DIR/dist/"* "$STATIC_DIR/"

echo "Frontend built and copied to $STATIC_DIR"
ls -la "$STATIC_DIR"
