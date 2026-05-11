#!/usr/bin/env bash
# Install ripgrep from its upstream GitHub release into ~/.local/bin.
# Supports macOS and Linux. Requires curl and tar. No sudo.

set -euo pipefail

RIPGREP_VERSION="14.1.1"
BIN_DIR="${HOME}/.local/bin"

target() {
    case "$(uname -s)-$(uname -m)" in
        Linux-x86_64)               echo "x86_64-unknown-linux-musl" ;;
        Linux-aarch64|Linux-arm64)  echo "aarch64-unknown-linux-gnu" ;;
        Darwin-x86_64)              echo "x86_64-apple-darwin" ;;
        Darwin-arm64)               echo "aarch64-apple-darwin" ;;
        *) echo "unsupported platform: $(uname -sm)" >&2; exit 1 ;;
    esac
}

if command -v rg &>/dev/null; then
    echo "ripgrep already installed: $(rg --version | head -1)"
    exit 0
fi

t="$(target)"
name="ripgrep-${RIPGREP_VERSION}-${t}"
url="https://github.com/BurntSushi/ripgrep/releases/download/${RIPGREP_VERSION}/${name}.tar.gz"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

echo "Downloading $url"
curl -fsSL "$url" -o "$tmp/rg.tar.gz"

echo "Installing to $BIN_DIR/rg"
tar -xzf "$tmp/rg.tar.gz" -C "$tmp"
mkdir -p "$BIN_DIR"
install -m 0755 "$tmp/${name}/rg" "$BIN_DIR/rg"

echo "Installed: $("$BIN_DIR/rg" --version | head -1)"

case ":${PATH}:" in
    *":${BIN_DIR}:"*) ;;
    *) echo "Note: $BIN_DIR is not on PATH. Add to your shell rc:"
       echo "    export PATH=\"\$HOME/.local/bin:\$PATH\"" ;;
esac
