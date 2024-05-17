FROM nvcr.io/nvidia/pytorch:23.08-py3

RUN \
  apt-get -y -q update && \
  # Prevents debconf from prompting for user input
  # See https://github.com/phusion/baseimage-docker/issues/58
  DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3-full \
    python3-dev \
    python3-venv \
    tmux \
    gcc \
    make

ARG user=dev
ARG uid=999
RUN groupadd -g $uid $user && \
    useradd -m -r -u $uid -g $user $user && \
    chown -R $user:$user /home/$user
ENV USER $user
USER $user
ENV HOME /home/$user

RUN mkdir -p $HOME/workdir
WORKDIR $HOME/workdir
RUN touch YOU_ARE_IN_DOCKER

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH=$HOME/.cargo/bin:$PATH

RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH=$HOME/.local/bin:$PATH

RUN uv venv --system-site-packages $HOME/venv

# Only copy and install from the lockfile first to improve docker caching
COPY --chown=$USER:$USER requirements.txt .
COPY --chown=$USER:$USER dev-requirements.txt .
RUN . $HOME/venv/bin/activate && uv pip install -r requirements.txt -r dev-requirements.txt

USER root
COPY scripts/entrypoint.sh /
RUN chmod 755 /entrypoint.sh
USER dev
ENTRYPOINT ["/entrypoint.sh"]
