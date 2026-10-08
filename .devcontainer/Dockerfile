# Fly Car devcontainer image.
#
# It builds the *same* conda environment as environment.yml, so the container
# and a native conda install behave identically.
#
# The build context is the repository root (see devcontainer.json).
FROM mcr.microsoft.com/devcontainers/base:ubuntu-24.04

# System libraries.
#  - libgl1 / libegl1 / libx*: Irrlicht (the 3D window) links against these
#    even when it is only imported and never opened.
#  - libvulkan1: needed by the VSG visualization module.
#  - xvfb: a virtual screen, handy for trying a 3D window on a machine with no
#    display (`xvfb-run python tutorials/03_vehicle_figure8.py`).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bzip2 ca-certificates curl \
        libgl1 libegl1 libglu1-mesa libvulkan1 \
        libx11-6 libxext6 libxrender1 libxxf86vm1 libxcursor1 libxrandr2 libxinerama1 libxi6 \
        xvfb \
    && rm -rf /var/lib/apt/lists/*

# micromamba: a small, fast, single-file conda replacement.
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xj -C /usr/local bin/micromamba

ENV MAMBA_ROOT_PREFIX=/opt/conda

# Create the environment. Copying only environment.yml first means Docker can
# reuse this (slow) layer until environment.yml itself changes.
COPY environment.yml /tmp/environment.yml
RUN micromamba env create -y -f /tmp/environment.yml \
    && micromamba clean --all --yes \
    && chmod -R a+rX /opt/conda

# Put the environment first on PATH so `python` and `pytest` just work in every
# terminal, with no `conda activate` step.
ENV PATH=/opt/conda/envs/flycar/bin:$PATH

# Let the NVIDIA container runtime expose everything (compute + graphics) when
# a GPU is passed through. Harmless when there is no GPU.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all

# Matplotlib has no window to draw in inside a container by default.
# Override with `MPLBACKEND=TkAgg` etc. if you set up X forwarding.
ENV MPLBACKEND=Agg
