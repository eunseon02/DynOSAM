#!/usr/bin/env bash
TAG=$1

if [ -z $TAG ]; then
    echo "Usage: ./docker/build_docker.sh TAG [BASE_IMAGE]"
    exit -1
fi

DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile -t $TAG .
