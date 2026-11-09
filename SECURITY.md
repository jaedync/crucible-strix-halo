# Security

## Network exposure and the API key

The Crucible server listens on `127.0.0.1:8790` by default. It needs no key there.

A key is required for an address that is not loopback, except in the Docker image.
Loopback is 127.0.0.0/8 and `::1`. Give the key as a file with `--api-key-file PATH`.
Without the key file, the server exits with code 2 before it loads the model.
With a key, every route but `GET /health` needs it. The server keeps only a digest of the key and never prints it.

The server has no TLS. To reach it from other machines, use a reverse proxy
(see [Remote access](docs/REMOTE_ACCESS.md)).

## Limits on connections

Each connection must send its request head (the first line and the headers) within 10 seconds.
A server with a key also limits the connections that have not yet sent the key. One address can hold at most 16
such connections for more than 2 seconds. When all 256 slots are full, the server closes the oldest
connection of the busiest address and takes the new one. If every address holds one connection, the server closes
only a connection older than 2 seconds. If none is that old, it refuses the new connection with HTTP 503. An IPv6 `/64`
counts as one address. A client that sent the key is never counted.
A server with no key (the default on `127.0.0.1`) has the 10-second head limit only.

## Host names and DNS rebinding

The service that `./crucible setup` and `./crucible start` run accepts only the `Host` names `localhost`, `127.0.0.1`,
`[::1]` and the bind address. This stops DNS rebinding: a web page in your browser cannot reach the server by an
attacker name that resolves to `127.0.0.1`. For a reverse proxy, add its name with `--allowed-host NAME`
(see [Remote access](docs/REMOTE_ACCESS.md)). A server that you start by hand with `python -m server` accepts any
`Host` name until you give `--allowed-host`.
The Docker image has no Host list by default (see [Docker](#docker)).

## Request-ID header

The server can copy a request ID from a client header into its log. The names `Authorization`, `Proxy-Authorization`,
`x-api-key`, `Cookie` and `Set-Cookie` carry credentials. The server refuses them as the name of this header.

## Docker

The Docker image lets the server listen on all addresses of the container with no key. Docker forwards the port to
the container address, so the server must listen there.

A published port on all addresses is an open server. Publish the port only to `127.0.0.1`
(`-p 127.0.0.1:8790:8790`), or use a key. Docker rules can go around the firewall of the host.
The image has no Host list by default, so with no key a web page in your browser can reach a published port through
DNS rebinding. To stop this, use a key (`--api-key-file`), or set
`-e CRUCIBLE_ALLOWED_HOSTS=localhost,127.0.0.1,[::1]` on the `docker run` line.
A key is optional in the container. See [docs/DOCKER.md](docs/DOCKER.md).

The switch that allows this works only when the file `/.dockerenv` or `/run/.containerenv` exists.
Do not use `--network host` with no key: the container then opens the network of the host.
