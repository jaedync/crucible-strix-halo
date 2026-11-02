# Updates

Each release has a number `v0.<train>.<fix>`. The release notes are in [CHANGELOG.md](../CHANGELOG.md) and on the
Releases page of this repository on GitHub. The file `VERSION` has the number of your
checkout. The server shows it at start, in `GET /v1/crucible/version`, and in `GET /v1/models`
(`crucible_version`). `build/engine/crucible --version` shows the number of the engine build.

**Watch for releases.** On GitHub, select Watch, then Custom, then Releases. GitHub sends you a message for each
new release.

**Start notice.** The server asks GitHub once in 24 hours if a newer release exists. It sends one plain request and
nothing else. If a newer release exists, the server prints two lines at start:

```text
Crucible v0.31.0 is available (you run v0.30.0). Update: make update. Release notes: <link>
Automatic updates are off unless you turned on the timer. See "Updates" in the README.
```

The notice never changes anything. A failed check prints nothing. To turn the check off, set
`CRUCIBLE_NO_UPDATE_CHECK=1` or start the server with `--no-update-check`.

**Update by hand.** Run this in your clone:

```sh
make update
```

The command gets the newest release tag, builds it, and runs a short self-test. It stops and changes nothing when
you have local changes, or when your checkout is not on the release line. If the build or the self-test fails, the
command restores the previous release by itself. It keeps the previous binary as `build/engine/crucible.prev`.

A kill or a power loss can stop an update half way. The next `make update` or `make rollback` finds the file
`build/update-running` and restores the previous release and its binary first, and it says so. It does not build again:
a build can be stopped again by the same cause, and a restore cannot. The linker writes a temporary file and renames it,
so the binary is never half written: a stopped build leaves the old binary or none. Until that next run, the checkout
can be on the new release and the binary on the old one. After a stopped update, run `make update` before you start
the server.

`make update` does not restart your server. Stop the server and start it again to run the new release.

`make update` leaves your clone on the release tag (a detached HEAD). A later `git pull` then fails. To get back on
a branch, run `git checkout main && git pull`. To follow the releases again, run `make update`.

To undo the last update, run:

```sh
make rollback
```

`make rollback` undoes one update, once, and needs a clean tree. It then holds the release that you left. `make update`
and the timer skip that release until a newer one exists. To install the held release anyway, delete the file
`build/update-hold` and run `make update`.

The self-test checks the CPU part only. To add a short GPU test (a few greedy tokens against the stored token IDs),
stop the server and run `make update CKPT="$CKPT"`. This release stores token IDs for the 3.05 and 4.05 packs. A pack
with no stored token IDs is skipped, and the command says so.

**Automatic updates (off by default).** Nothing in this repository turns them on. To turn them on, run these three
commands in your clone. They install a systemd user timer that runs `make update` once a day at about 04:00:

```sh
mkdir -p ~/.config/systemd/user
for f in crucible-update.service crucible-update.timer; do sed "s|@CHECKOUT@|$PWD|" deploy/user-systemd/$f > ~/.config/systemd/user/$f; done
systemctl --user enable --now crucible-update.timer
```

A user timer runs only while you are logged in. To let it run when you are logged out, run
`loginctl enable-linger "$USER"` once.

The build runs next to your loaded model. A build with one job peaked at 1.4 GB of memory in a test. The service
file limits the build to 4 GB (`MemoryMax`) and tells the kernel to stop the build first if memory runs out
(`OOMScoreAdjust`). In the same test the build took 16 minutes.

The 4 GB limit covers the split build, which is the default. If that build cannot run, the build falls back to the plain
build, which can need more memory. If the plain build hits the limit, the kernel stops it, `make update` restores the
previous release (`OOMPolicy=continue` in the service file lets it do so), and the timer tries again the next night.

To turn them off:

```sh
systemctl --user disable --now crucible-update.timer
```

After a good update, the timer job restarts the user unit `crucible-server.service` if that unit runs. A restart
stops the requests that are in progress. The folder `deploy/user-systemd/` has a template for that unit. In the
template, replace `/usr/bin/python3` with the `python` of your virtual environment if you installed
`requirements-cpu.txt` in one.
Use no restart if you start the server in another way: delete the `CRUCIBLE_RESTART_UNIT` line in the installed
service file. The timer job does not run the GPU self-test.
