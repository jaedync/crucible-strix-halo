# user-systemd: the service and the update timer

## The server as a service

Run `./crucible service install`. It writes `~/.config/systemd/user/crucible-server.service`, enables it and starts it.
Do not copy the file by hand. The file `crucible-server.service` in this folder is the template of that command:
the command fills in the paths of your clone. A test keeps the two texts equal.

- `./crucible service status` shows the state, the restarts and why the service is down.
- `./crucible service logs -f` shows the journal.
- `./crucible service uninstall` removes the service.
- Run `loginctl enable-linger "$USER"` once to start the service at boot, before any login.
  If the system refuses it, run `sudo loginctl enable-linger "$USER"`.

## Opt-in automatic updates

Templates for a systemd user timer that runs `./crucible update --yes --unattended` every day. They are off by default. No script and no
make target installs or enables them. "Updates" in the main README shows the commands.

- `crucible-update.service`: runs `./crucible update --yes --unattended` in your clone. The command restarts `crucible-server.service` if it runs, and it rolls back by itself if a step fails. When Kiln's unit `kiln-crucible.service` exists, the timer changes nothing and logs one line: run `./crucible update` by hand. Only an update by hand stops and restarts the Kiln unit.
  A timer from an older template runs `./crucible update --yes` with no `--unattended`. The update sees that it runs in
  `crucible-update.service` (or cannot read `/proc/self/cgroup`) and acts as with `--unattended`. Install the new template
  again to get the flag in the unit.
- `crucible-update.timer`: runs the service daily at 04:00.
