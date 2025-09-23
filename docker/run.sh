#!/usr/bin/env sh
if [ -n "$1" ]; then
	echo "Source dir: $1"
else
	echo "usage ./docker/.run.sh <SOURCE_DIR> <DATA_DIR>"
fi

if [ -n "$2" ]; then
	echo "Data dir: $2"
else
	echo "usage: ./docker/run.sh <SOURCE_DIR> <DATA_DIR>"
	exit -1
fi

IMAGE_NAME=un3-mapping

docker run --gpus "all" \
	-it \
	--rm \
	--env="DISPLAY" \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	--privileged \
	--network host \
	-e DISPLAY=$DISPLAY \
	-e QT_X11_NO_MITSHM=1 \
	-v "$2:/workspace/repository/data/" \
	-v "$1:/workspace/repository/" \
	${IMAGE_NAME} \
	bash -c /bin/bash
