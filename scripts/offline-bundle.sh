#!/usr/bin/env sh
# Prepare an air-gapped install. Run on a machine WITH internet:
#   scripts/offline-bundle.sh            -> industrial-rag-images.tar.gz
# Copy the repository folder + the tarball to the site, then on the site machine:
#   docker load -i industrial-rag-images.tar.gz
#   docker compose up -d                 (no --build needed; nothing is downloaded)
set -eu
cd "$(dirname "$0")/.."
docker compose build
docker compose pull postgres qdrant redis
IMAGES=$(docker compose config --images | sort -u)
echo "Saving images:"; echo "$IMAGES"
# shellcheck disable=SC2086
docker save $IMAGES | gzip > industrial-rag-images.tar.gz
ls -lh industrial-rag-images.tar.gz
