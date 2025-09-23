#!/usr/bin/env sh

IMAGE_NAME=un3-mapping

docker build --build-arg UNAME=$(whoami) --build-arg UID=$(id -u) --build-arg GID=$(id -g) -t ${IMAGE_NAME} .
