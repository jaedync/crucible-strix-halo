## Reach the server from other machines

By default the server listens on `127.0.0.1`. Only programs on the same machine can reach it. This is the safe choice.
Keep it, and put a reverse proxy in front of the server. The proxy adds TLS (https) for the other machines.

The server has no TLS code. Without TLS, a person on the network can read your prompts and your API key.

### Step 1: make an API key

```sh
./crucible setup --yes --api-key
```

This makes the file `~/.config/crucible/api-key` (mode 0600, in a folder with mode 0700) and starts the server with it.
If the server is running, run `./crucible stop` first, so that it restarts with the key.
The key is in that file only. It is not in `config.json`, in `serve.sh` or in any argument.

Without `./crucible`, make a file by hand and start the server with `--api-key-file`:

```sh
mkdir -p -m 700 ~/.config/crucible
( umask 077; python3 -c 'import secrets; print("crucible-" + secrets.token_urlsafe(32))' > ~/.config/crucible/api-key )
python -m server "$CKPT" --engine build/engine/crucible --api-key-file ~/.config/crucible/api-key
```

The path must be absolute. The server stops with exit code 78 if the file is a link, is not yours, can be read by other
users, or does not hold one key of 32 to 256 characters. Run `chmod 600` on the file to fix the mode. The server reads
the file once at start. To change the key, replace the file and restart the server.

Every request needs the key, except `GET /health`. A client sends the key in one of two ways:

```sh
curl -sS http://127.0.0.1:8790/v1/models -H "Authorization: Bearer $(cat ~/.config/crucible/api-key)"
curl -sS http://127.0.0.1:8790/v1/models -H "x-api-key: $(cat ~/.config/crucible/api-key)"
```

Set the key as the API key of your client. OpenAI clients send `Authorization`. Anthropic clients send `x-api-key`.
A request without the key, or with a wrong key, gets `401`.

### Step 2: use a reverse proxy

Do not use `--host 0.0.0.0` and do not use `--bind 0.0.0.0` when a proxy can do the work. The proxy runs on the same
machine and sends the requests to `127.0.0.1:8790`. The proxy passes the `Authorization` header on, and the server checks it.

Do not add basic authentication in the proxy. It uses the same `Authorization` header as the key.

**Caddy.** Caddy gets a certificate by itself for a public name. Put this in the `Caddyfile`:

```
ai.example.org {
    reverse_proxy 127.0.0.1:8790
}
```

Caddy sends event streams (SSE) at once. For a name that only your LAN knows,
add `tls internal` inside the block. Clients then must trust the Caddy root certificate.

**nginx.** Put this in a `server` block that has your certificate (`ssl_certificate` and `ssl_certificate_key`):

```
location / {
    proxy_pass http://127.0.0.1:8790;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_buffering off;
    proxy_read_timeout 3600s;
    proxy_send_timeout 3600s;
    client_max_body_size 20m;
}
```

`proxy_buffering off` lets the tokens arrive as the model makes them. `proxy_read_timeout` keeps a long answer open.
`client_max_body_size` is above the 16 MiB body limit of the server. The nginx default of 1 MB refuses a large prompt or an image.

### Step 3: tell the server the name of the proxy

The service that `./crucible setup` and `./crucible start` run accepts only these `Host` names: `localhost`,
`127.0.0.1`, `[::1]` and the bind address. This stops DNS rebinding. In a DNS rebinding attack, a web page in your
browser uses a name that first points to a public address and then to `127.0.0.1`. The page can then send
requests to a server on your machine. A `Host` list stops this, because the attacker name is not on it.

Caddy and nginx (with `proxy_set_header Host $host`) send the name that the client used, for example
`ai.example.org`. The server then answers `403` for that name, and the text of the answer names this command. Add the name to the
list, and restart the server:

```sh
./crucible setup --yes --api-key --allowed-host ai.example.org
```

Give `--allowed-host` more than once for more names. A name can have a port (`ai.example.org:8443`). The names that you
give replace the names of an earlier `setup`, so give all of them each time. The list does not hold the name
of a server that you start by hand with `python -m server`. That server accepts any `Host` name until you give
`--allowed-host` or set `CRUCIBLE_ALLOWED_HOSTS`.

As an alternative, set the `Host` header in the proxy to `127.0.0.1:8790` (`proxy_set_header Host 127.0.0.1:8790;` in
nginx, `header_up Host 127.0.0.1:8790` inside `reverse_proxy` in Caddy). No name is then necessary in the list.

### Step 4: check

Run these commands on another machine:

```sh
curl -sS https://ai.example.org/health                                   # expect: {"status": "ok"}
curl -sS https://ai.example.org/v1/models                                # expect: 401
curl -sS https://ai.example.org/v1/models -H "Authorization: Bearer KEY" # expect: the model list
```

### If you must listen on the network

`./crucible setup --bind 0.0.0.0` (or `--bind ::`, or a LAN address) makes the key file and starts the server on that
address. The server refuses to start on an address that is not loopback (127.0.0.0/8 and ::1) without a key file. It exits with code 2.
The Docker image is the one exception (see [Docker](DOCKER.md)).
The address `::ffff:127.0.0.1` is not loopback for this rule.

The key is the only protection then, and the traffic has no TLS. Add a firewall rule for the port, and use
`--allowed-host` to limit the `Host` names that the server accepts.
