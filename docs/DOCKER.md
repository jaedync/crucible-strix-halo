# Docker

The [Dockerfile](../Dockerfile) builds the engine for gfx1151 from a pinned release tag and installs the HTTP server.
The image holds no model. Mount an EXL3 pack as a volume.

## Tested setup

| Item | Version |
|---|---|
| Machine | Framework Desktop, Ryzen AI Max+ 395, Radeon 8060S (gfx1151), 128 GB |
| Host | Ubuntu 26.04.1 LTS, kernel 7.0.0-42-generic, in-tree amdgpu driver |
| Host ROCm | None. ROCm is only in the image. |
| Image ROCm | 7.2.4 (`rocm/dev-ubuntu-24.04:7.2.4`) |
| Docker | 29.1.3 |
| Crucible | v0.32.0, pack 4.05bpw_h6_ng6 |

Other kernels and ROCm versions are not tested.

## Build

```sh
docker build -t crucible:v0.32.0 .
```

The build needs no GPU. It clones the tag in `CRUCIBLE_VERSION` and runs `make engine` for gfx1151.
These build arguments change the defaults:

| Argument | Default | Content |
|---|---|---|
| `CRUCIBLE_VERSION` | `v0.32.0` | The release tag to build |
| `BUILD_JOBS` | `1` | Parallel HIP compiles. Each compile needs some GB of RAM. |
| `ROCM_IMAGE` | `docker.io/rocm/dev-ubuntu-24.04:7.2.4` | The base image of the build and the run |

Example: `docker build --build-arg BUILD_JOBS=2 -t crucible:v0.32.0 .`

## Model

Download a pack as in the [main README](../README.md#download-a-model). Read the model license note first.

## Run

```sh
docker run --rm \
  --device /dev/kfd --device /dev/dri \
  --group-add "$(getent group video | cut -d: -f3)" \
  --group-add "$(getent group render | cut -d: -f3)" \
  --security-opt seccomp=unconfined \
  -v "$CKPT":/models/pack:ro \
  -p 127.0.0.1:8790:8790 \
  crucible:v0.32.0 \
  /models/pack --host 0.0.0.0 --port 8790 --ctx 32768
```

- `--device /dev/kfd --device /dev/dri` give the container access to the GPU.
- `--group-add` adds the numeric IDs of the host groups `video` and `render`. The names in the image can have other IDs.
- `--security-opt seccomp=unconfined` is the setting that we tested. The ROCm container documentation uses it.
- The arguments after the image name go to `python -m server`. The first argument is the pack directory in the container.
- `--host 0.0.0.0` is necessary in the container, because Docker forwards the port to the container address.
  The server then prints a warning about a non-loopback address. Bind the published port to `127.0.0.1` as above.
  The server has no authentication and no TLS. Do not expose it to a network.

The server binds its port only after the engine is ready. Then `GET /health` answers `200`.
Test a request:

```sh
curl -sS http://127.0.0.1:8790/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"crucible","messages":[{"role":"user","content":"Say hello."}],"temperature":0,"max_tokens":32}'
```

## Updates

The image sets `CRUCIBLE_NO_UPDATE_CHECK=1`, because a container does not update itself.
For a new release, build again with a new `CRUCIBLE_VERSION`. `make update` does not apply to the image.
