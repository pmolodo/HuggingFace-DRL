FROM stablebaselines/rl-baselines3-zoo:2.2.0a1

USER root

RUN apt-get -y update \
    && apt-get -y install \
    git \
    git-lfs \
    xvfb \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER $MAMBA_USER

RUN \
    eval "$(micromamba shell hook --shell bash)" \
    && micromamba activate \
    && pip install \
    rl_zoo3==2.3.0 \
    pyopengl \
    gymnasium[atari] \
    gymnasium[accept-rom-license] \
    pyvirtualdisplay \
    ipython \
    && pip cache purge

ENV CODE_DIR /home/$MAMBA_USER/code
ENV ZOO3_DIR ${CODE_DIR}/rl_zoo3

RUN mkdir -p ${ZOO3_DIR}

WORKDIR ${ZOO3_DIR}
