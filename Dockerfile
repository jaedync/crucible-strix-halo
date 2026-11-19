# Container image of Crucible for gfx1151, built from a pinned release tag.
# The image holds no model: mount an EXL3 pack as a volume (docs/DOCKER.md).
ARG CRUCIBLE_VERSION=v0.38.0
ARG ROCM_IMAGE=docker.io/rocm/dev-ubuntu-24.04:7.2.4

FROM ${ROCM_IMAGE} AS build
ARG CRUCIBLE_VERSION
ARG CRUCIBLE_REPO=https://github.com/jaedync/crucible-strix-halo.git
ARG BUILD_JOBS=1
RUN apt-get update \
 && apt-get install -y --no-install-recommends git make python3 ca-certificates rocm-llvm-dev \
 && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch "${CRUCIBLE_VERSION}" "${CRUCIBLE_REPO}" /opt/crucible
WORKDIR /opt/crucible
RUN make -j "${BUILD_JOBS}" GPU_ARCH=gfx1151 engine \
 && rm -rf build/engine/split .git

FROM ${ROCM_IMAGE}
ARG CRUCIBLE_VERSION
LABEL org.opencontainers.image.title="crucible" \
      org.opencontainers.image.version="${CRUCIBLE_VERSION}" \
      org.opencontainers.image.source="https://github.com/jaedync/crucible-strix-halo" \
      org.opencontainers.image.licenses="Apache-2.0"
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv \
 && rm -rf /var/lib/apt/lists/*
COPY --from=build /opt/crucible /opt/crucible
RUN python3 -m venv /opt/crucible/.venv \
 && /opt/crucible/.venv/bin/pip install --no-cache-dir -r /opt/crucible/requirements-cpu.txt
WORKDIR /opt/crucible
ENV CRUCIBLE_NO_UPDATE_CHECK=1
# The n-gram table stays in host memory, as in the documented run line (-e CRUCIBLE_PLE_HOST=1). The engine default (0)
# puts it in GPU memory, where the 4.05 pack does not fit with its sessions.
ENV CRUCIBLE_PLE_HOST=1
# Only this image sets the switch (docs/DOCKER.md). With it, --host 0.0.0.0 or :: starts with no API key.
ENV CRUCIBLE_CONTAINER_OPEN_BIND=1
EXPOSE 8790
# Use the full docker run line in docs/DOCKER.md for the measured configuration.
# Arguments: the EXL3 pack directory, then the server options.
ENTRYPOINT ["/opt/crucible/.venv/bin/python", "-m", "server"]
