#!/bin/bash
set -e

REPO="himanishpuri/ct"
BINARY_NAME="ct"

# detect os and arch
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)

case $ARCH in
    x86_64) ARCH="amd64" ;;
    aarch64|arm64) ARCH="arm64" ;;
    *) echo "unsupported architecture: $ARCH"; exit 1 ;;
esac

# get latest version from github api
LATEST_TAG=$(curl -s "https://api.github.com/repos/$REPO/releases/latest" | grep '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/')

if [ -z "$LATEST_TAG" ]; then
    echo "could not find latest release for $REPO"
    exit 1
fi

echo "installing $BINARY_NAME $LATEST_TAG for $OS/$ARCH..."

# construct download url (assuming naming convention: ct_Linux_x86_64.tar.gz)
# adjusted to match typical goreleaser output or simple manual uploads
URL="https://github.com/$REPO/releases/download/$LATEST_TAG/${BINARY_NAME}_${OS}_${ARCH}.tar.gz"

TMP_DIR=$(mktemp -d)
curl -sL "$URL" -o "$TMP_DIR/package.tar.gz"

# extract and install
tar -xzf "$TMP_DIR/package.tar.gz" -C "$TMP_DIR"
chmod +x "$TMP_DIR/$BINARY_NAME"

sudo mv "$TMP_DIR/$BINARY_NAME" /usr/local/bin/

# cleanup
rm -rf "$TMP_DIR"

echo "$BINARY_NAME has been installed to /usr/local/bin/$BINARY_NAME"
ct --help
