FROM ubuntu:22.04

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV TERM=xterm
# Auto-accept NVIDIA Omniverse EULA
ENV ACCEPT_EULA=Y
ENV OMNI_KIT_ACCEPT_EULA=yes

# System dependencies (from docs/installation.md)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    ca-certificates \
    libglu1-mesa \
    libgl1 \
    libegl1 \
    libxrandr2 \
    libxinerama1 \
    libxcursor1 \
    libxi6 \
    libxext6 \
    libx11-6 \
    libglib2.0-0 \
    libxt6 \
    libvulkan1 \
    vulkan-tools \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /opt/lehome-challenge

# Copy dependency files first (cache layer for expensive uv sync)
COPY pyproject.toml uv.lock ./

# Install all dependencies
RUN uv sync

# Activate venv (Docker equivalent of source .venv/bin/activate)
ENV VIRTUAL_ENV=/opt/lehome-challenge/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install IsaacLab
# flatdict 4.0.1 needs pkg_resources which was removed in setuptools>=70.
# Pin an older setuptools, install flatdict, then let isaaclab.sh proceed.
RUN uv pip install "setuptools<70" wheel \
    && uv pip install flatdict==4.0.1 --no-build-isolation
COPY third_party/IsaacLab ./third_party/IsaacLab
RUN chmod +x ./third_party/IsaacLab/isaaclab.sh \
    && ./third_party/IsaacLab/isaaclab.sh -i none \
    && uv pip install -e ./third_party/IsaacLab/source/isaaclab --no-build-isolation

# Install lehome source package
COPY source/lehome ./source/lehome
RUN uv pip install -e ./source/lehome

# Copy remaining project files
COPY . .

ENV __GLX_VENDOR_LIBRARY_NAME=nvidia
# Expose all NVIDIA driver capabilities (compute, graphics, video, display)
# Without this, Docker only exposes compute+utility, and IsaacSim can't use Vulkan
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV NVIDIA_VISIBLE_DEVICES=all

CMD ["/bin/bash"]
