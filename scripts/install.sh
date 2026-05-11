#!/usr/bin/env bash
# Install Willow as a user-level `willow` command.
#
# Usage from a local checkout:
#   scripts/install.sh
#
# Usage from a hosted install script:
#   curl -fsSL https://raw.githubusercontent.com/<owner>/willow/main/scripts/install.sh \
#     | WILLOW_REPO_URL=https://github.com/<owner>/willow.git bash

set -euo pipefail

INSTALL_ROOT="${WILLOW_INSTALL_ROOT:-${HOME}/.willow}"
REPO_DIR="${WILLOW_REPO_DIR:-${INSTALL_ROOT}/willow}"
REPO_URL="${WILLOW_REPO_URL:-}"
BIN_DIR="${HOME}/.local/bin"

is_willow_repo() {
    [[ -f "pyproject.toml" && -f "src/willow/cli.py" ]] \
        && grep -q '^name = "willow"$' pyproject.toml
}

ensure_uv() {
    if command -v uv >/dev/null 2>&1; then
        return
    fi

    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${BIN_DIR}:${PATH}"
}

checkout_repo() {
    if is_willow_repo; then
        REPO_DIR="$(pwd)"
        return
    fi

    if [[ -z "${REPO_URL}" ]]; then
        echo "WILLOW_REPO_URL is required when not running from a Willow checkout." >&2
        echo "Example: curl -fsSL https://raw.githubusercontent.com/<owner>/willow/main/scripts/install.sh | WILLOW_REPO_URL=https://github.com/<owner>/willow.git bash" >&2
        exit 1
    fi

    mkdir -p "${INSTALL_ROOT}"
    if [[ -d "${REPO_DIR}/.git" ]]; then
        echo "Updating Willow in ${REPO_DIR}"
        git -C "${REPO_DIR}" pull --ff-only
    else
        echo "Cloning Willow into ${REPO_DIR}"
        git clone "${REPO_URL}" "${REPO_DIR}"
    fi
}

ensure_uv
checkout_repo

echo "Installing willow from ${REPO_DIR}"
uv tool install --force -e "${REPO_DIR}"

echo "Installed: $(command -v willow || true)"
if ! command -v willow >/dev/null 2>&1; then
    echo "Note: ${BIN_DIR} may not be on PATH. Add this to your shell rc:"
    echo "    export PATH=\"\$HOME/.local/bin:\$PATH\""
fi
