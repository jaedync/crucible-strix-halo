# user-systemd: opt-in automatic updates

Templates for a systemd user timer that runs `make update` every day. They are off by default. No script and no
make target installs or enables them. "Updates" in the main README shows the commands.

- `crucible-update.service`: runs `make update` in your clone, then restarts `crucible-server.service` if it runs.
- `crucible-update.timer`: runs the service daily at 04:00.
- `crucible-server.service`: an optional unit for the server itself.
- A user timer runs only while you are logged in. Run `loginctl enable-linger "$USER"` once to let it run when you are logged out.
- `crucible-server.service` starts `/usr/bin/python3`. Replace it with the `python` of your virtual environment if the server dependencies are there.
