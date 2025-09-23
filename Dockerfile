
FROM	pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel
ARG	UNAME=user
ARG	UID=1000
ARG	GID=1000
RUN	apt update \
	&& apt -y upgrade
RUN	conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia
RUN	apt install -y wget git vim libgl1
RUN	groupadd -g $GID $UNAME
RUN	useradd -m -u $UID -g $GID -o -s /bin/bash $UNAME
RUN	chmod -R 777 /workspace
USER	$UNAME
RUN	pip install --upgrade pip
RUN	pip install open3d scikit-image tqdm natsort typing_extensions==4.15.0 "numpy<1.27"
