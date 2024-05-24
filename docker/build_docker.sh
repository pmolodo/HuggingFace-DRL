#!/bin/bash

set -e
set -u

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TAG=pmolodo-huggingface-drl
VERSION=0.1.0

docker build -t ${TAG}:${VERSION} . -f "${THIS_DIR}/Dockerfile"
docker tag ${TAG}:${VERSION} ${TAG}:latest

if [[ ${RELEASE:-False} == "True" ]]; then
  docker push ${TAG}:${VERSION}
  docker push ${TAG}:latest
fi
