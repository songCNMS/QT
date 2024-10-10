FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    net-tools \
    vim \
    virtualenv \
    wget \
    xpra \
    xserver-xorg-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test
RUN apt-get -y update

RUN apt-get install -y libosmesa6-dev libgl1-mesa-glx libglfw3 libgl1-mesa-dev
RUN apt-get install -y git-core
RUN apt install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget
RUN apt install -y python-is-python3
RUN apt-get install python3-pip -y
RUN apt-get install -y g++-11
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/home/aiscuser/.mujoco/mujoco210/bin:/home/aiscuser/.local/bin:${LD_LIBRARY_PATH}
RUN apt-get install patchelf

RUN pip install azureml-mlflow tensorboard
RUN pip install requests==2.23.0
RUN pip install -r requirements.txt
RUN pip install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl
RUN pip install packaging==21.3
RUN pip install "dm_control<=1.0.20" "mujoco<=3.1.6"