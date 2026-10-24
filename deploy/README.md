# deploy: charter

Opt-in service templates for a Crucible checkout.

## Interface

- `user-systemd/`: templates of a systemd user timer that runs `make update` once a day, and of a user unit for the
  server. They are off by default. "Updates" in the main README shows the commands that install them.

## Invariants

- No file here runs by itself. A person installs a template on purpose.
- A template holds no path of a machine and no secret. `@CHECKOUT@` is replaced when you install it.
