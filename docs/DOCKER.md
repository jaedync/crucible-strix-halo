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
docker build -t crucible:v0.34.0 .
```

The build needs no GPU. It clones the tag in `CRUCIBLE_VERSION` and runs `make engine` for gfx1151.
These build arguments change the defaults:

| Argument | Default | Content |
|---|---|---|
| `CRUCIBLE_VERSION` | `v0.34.0` | The release tag to build |
| `BUILD_JOBS` | `1` | Parallel HIP compiles. Each compile needs some GB of RAM. |
| `ROCM_IMAGE` | `docker.io/rocm/dev-ubuntu-24.04:7.2.4` | The base image of the build and the run |

Example: `docker build --build-arg BUILD_JOBS=2 -t crucible:v0.34.0 .`

## Model

Download a pack as in the [main README](../README.md#download-a-model). Read the model license note first.

## Run

This run line gives the configuration of the speed table in the [main README](../README.md): the same words and options
that `./crucible start` sets on a host (the table is `tools/crucible_cli/measured.py`; a test checks this page against it).

```sh
docker run --rm \
  --device /dev/kfd --device /dev/dri \
  --group-add "$(getent group video | cut -d: -f3)" \
  --group-add "$(getent group render | cut -d: -f3)" \
  --security-opt seccomp=unconfined \
  -e CRUCIBLE_PLE_HOST=1 -e CRUCIBLE_COMPACT_STATE=1 -e CRUCIBLE_PLE_FETCH=1 \
  -e CRUCIBLE_DRAFT_HEAD=q2 -e CRUCIBLE_DRAFT_CHAIN=hc8,lead -e CRUCIBLE_PF_LAUNCH_ROWS=16384+128 \
  -v "$CKPT":/models/pack:ro \
  -p 127.0.0.1:8790:8790 \
  crucible:v0.34.0 \
  /models/pack --host 0.0.0.0 --port 8790 --streams 4 --ctx 262144 --prefill 4096 \
  --conversation-cache-mib 9700 --host-reserve-mib 11264 --vision
```

- The image runs `python -m server`, not `./crucible start`. The `-e` words and the options after the image name are the ones
  that `./crucible start` sets. The service also sets `--clamp-max-tokens`, `--default-thinking on` and a state folder; add them if you want them.
- The line is the layout of the speed table: the 4.05 pack, 4 sessions of 262,144 tokens, images on. For less memory, lower `--streams` or `--ctx` and keep the other words.
- Without `--conversation-cache-mib`, the server keeps no conversation state and prefills the whole conversation again on every
  follow-up turn. It prints a notice at start in that case.
- `CRUCIBLE_DRAFT_HEAD` and `CRUCIBLE_DRAFT_CHAIN` change the drafts only. The verify pass checks every draft, so the tokens do not change.
- `--device /dev/kfd --device /dev/dri` give the container access to the GPU.
- `--group-add` adds the numeric IDs of the host groups `video` and `render`. The names in the image can have other IDs.
- `--security-opt seccomp=unconfined` is the setting that we tested. The ROCm container documentation uses it.
- The arguments after the image name go to `python -m server`. The first argument is the pack directory in the container.
- `--host 0.0.0.0` is necessary in the container, because Docker forwards the port to the container address.
- `-p 127.0.0.1:8790:8790` publishes the port to your own machine only. Keep this form.

### The key rule in the image

A key is required for an address that is not loopback, except in the Docker image.
The image sets `CRUCIBLE_CONTAINER_OPEN_BIND=1`. With it, the server accepts `--host 0.0.0.0` (or `--host ::`) with no key,
so the line above works without a key file. Only these two addresses are open. Any other address needs a key file.
The server prints one `WARNING` line at start when it runs with no key in the container.
The switch works only inside a container. The server checks that the file `/.dockerenv` (Docker) or
`/run/.containerenv` (Podman) exists. On a host without one of these files, the switch does nothing, and a start on
`0.0.0.0` with no key exits with code 2. Do not set `CRUCIBLE_CONTAINER_OPEN_BIND=1` on a host.

**Do not use `--network host` with no key.** With `--network host`, the container shares the network of the host.
`--host 0.0.0.0` then opens the port `8790` on every address of the host, and `-p` does not limit it.
Anyone who can reach the host can send requests with no key. Use `-p 127.0.0.1:8790:8790` with the default network,
or make a key file (see below).

A published port on all addresses is an open server. `-p 8790:8790` publishes the port on every address of the host,
and Docker rules can go around the firewall of the host. Anyone who can reach the port can send requests with no key.
Do not use `-p 8790:8790` with no key. Use `-p 127.0.0.1:8790:8790`, or use a key.

### DNS rebinding with no key

**The image has no Host list by default.** With no key, a web page in your browser can reach a published port
through DNS rebinding. This is also true for `-p 127.0.0.1:8790:8790`. Choose one of these two fixes:

- Use a key. Mount a key file and pass `--api-key-file` (see below). A web page does not have the key.
- Set the Host list. Add `-e CRUCIBLE_ALLOWED_HOSTS=localhost,127.0.0.1,[::1]` to the `docker run` line, before the
  image name. The server then refuses a request with any other `Host` name. Add the name of a reverse proxy to the list.

A key is optional in the container. To reach the port from another machine, make a key file and mount it:

```sh
mkdir -p -m 700 ~/.config/crucible
( umask 077; python3 -c 'import secrets; print("crucible-" + secrets.token_urlsafe(32))' > ~/.config/crucible/api-key )
docker run --rm \
  --device /dev/kfd --device /dev/dri \
  --group-add "$(getent group video | cut -d: -f3)" \
  --group-add "$(getent group render | cut -d: -f3)" \
  --security-opt seccomp=unconfined \
  -v "$CKPT":/models/pack:ro \
  -v ~/.config/crucible/api-key:/run/crucible-api-key:ro \
  --user "$(id -u):$(id -g)" \
  -p 8790:8790 \
  crucible:v0.34.0 \
  /models/pack --host 0.0.0.0 --port 8790 --ctx 32768 --api-key-file /run/crucible-api-key
```

- With `--api-key-file`, every request but `GET /health` needs the key, also when the switch is set.
- The server refuses a key file that belongs to another user than the one that runs it (exit code 78). `--user` makes
  the container user the owner of the mounted file. With rootless podman the container user root is your user,
  so omit `--user`.
- The server has no TLS. Put a reverse proxy in front of it (see [Remote access](REMOTE_ACCESS.md)).

The server binds its port only after the engine is ready. Then `GET /health` answers `200`.
Test a request:

```sh
curl -sS http://127.0.0.1:8790/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"crucible","messages":[{"role":"user","content":"Say hello."}],"temperature":0,"max_tokens":32}'
```

## Updates

The image sets `CRUCIBLE_NO_UPDATE_CHECK=1`, because a container does not update itself.
For a new release, build again with a new `CRUCIBLE_VERSION`. `make update` does not apply to the image.
