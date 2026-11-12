"""./crucible service: the systemd user unit, install, uninstall, status, logs, start and stop.

The unit text comes from one function, render_unit. The file deploy/user-systemd/crucible-server.service is the output of
that function with placeholders, and a test keeps the two equal. The tool never runs sudo: when linger is refused, it prints
the command for the user.
"""
import json
import re
from pathlib import Path

from . import settings as settings_mod
from .files import Sandbox

UNIT = "crucible-server.service"
CHECK_FILE = "last-start-check.json"
OK_PATH = re.compile(r"^[A-Za-z0-9._/@+ -]+$")   # no %, $, quotes, backslash or newline in a unit file
OK_BIND = re.compile(r"^[A-Za-z0-9.:@-]+$")
STATES_OK = ("running", "degraded", "starting", "initializing")
MIN_START_TIMEOUT = 300
DEFAULT_START_TIMEOUT = 600
MAX_START_TIMEOUT = 3600

TEMPLATE = """\
# Written by ./crucible service install (Crucible @VERSION@). Run that command again to change it; do not edit.
[Unit]
Description=Crucible LLM server on @BIND@:@PORT@
Documentation=file://@REPO@/docs/QUICKSTART.md
StartLimitIntervalSec=15min
StartLimitBurst=4

[Service]
Type=notify
NotifyAccess=main
WorkingDirectory=@REPO@
Environment=PYTHONUNBUFFERED=1
ExecStartPre="@VENV_PY@" -I "@REPO@/crucible" doctor --for start --quiet@DOCTOR_ARGS@
ExecStart="@EXEC_START@" --systemd
TimeoutStartSec=@START_TIMEOUT@
TimeoutStopSec=120s
KillMode=control-group
WatchdogSec=30s
WatchdogSignal=SIGKILL
Restart=on-failure
RestartSec=5s
RestartPreventExitStatus=2 78
LimitCORE=0
LimitNOFILE=65536
UMask=0077
NoNewPrivileges=yes
SyslogIdentifier=crucible

[Install]
WantedBy=default.target
"""

PROPS = ("LoadState", "ActiveState", "SubState", "Result", "NRestarts", "ExecMainStatus", "StatusText",
         "UnitFileState", "ActiveEnterTimestamp")


class UnitPathError(ValueError):
    pass


def _path_ok(name, value):
    value = str(value)
    if not OK_PATH.fullmatch(value):
        raise UnitPathError(f"The {name} {value!r} has a character that a unit file cannot hold safely. "
                            "Allowed: letters, digits and . _ / @ + - and the space. Move the folder to such a path.")
    return value


def _arg(value):
    value = _path_ok("argument", value)
    return f'"{value}"' if " " in value else value


def render_unit(*, repo, venv_py, exec_start, bind, port, start_timeout, version, doctor_args=None):
    """The text of the unit. Pure: no file, no program. A value in the form @NAME@ stays as it is (for the template)."""
    if not OK_BIND.fullmatch(str(bind)):
        raise UnitPathError(f"The bind address {bind!r} has a character that a unit file cannot hold safely.")
    if isinstance(start_timeout, str):
        timeout = _path_ok("start timeout", start_timeout)
    else:
        timeout = f"{int(start_timeout)}s"
    values = {
        "VERSION": _path_ok("version", version), "BIND": str(bind), "PORT": _path_ok("port", port),
        "REPO": _path_ok("repo folder", repo), "VENV_PY": _path_ok("venv python", venv_py),
        "EXEC_START": _path_ok("serve script", exec_start), "START_TIMEOUT": timeout,
        "DOCTOR_ARGS": "".join(" " + _arg(a) for a in (doctor_args or [])),
    }
    return re.sub(r"@([A-Z_]+)@", lambda m: values[m.group(1)] if m.group(1) in values else m.group(0), TEMPLATE)


def template_text():
    """The deploy template: the generator output with placeholders."""
    return render_unit(repo="@REPO@", venv_py="@VENV_PY@", exec_start="@EXEC_START@", bind="@BIND@", port="@PORT@",
                       start_timeout="@START_TIMEOUT@", version="@VERSION@")


def start_timeout_for(load_seconds):
    """Three times the load time that setup measured, at least 300 s and at most one hour. Ten minutes when unknown."""
    if not load_seconds:
        return DEFAULT_START_TIMEOUT
    return int(min(MAX_START_TIMEOUT, max(MIN_START_TIMEOUT, round(3 * float(load_seconds)))))


# paths ---------------------------------------------------------------------------------------------------------------
def unit_dir(host):
    xdg = host.env.get("XDG_CONFIG_HOME")
    return (Path(xdg) if xdg else Path(host.home) / ".config") / "systemd/user"


def unit_path(host, unit=UNIT):
    return unit_dir(host) / unit


def installed(host, unit=UNIT):
    return unit_path(host, unit).is_file()


def user_name(host):
    return host.env.get("USER") or host.env.get("LOGNAME") or Path(host.home).name


def check_file_path(host, s, check_file=None):
    if check_file:
        return Path(check_file)
    work = host.env.get("CRUCIBLE_WORK")
    return (Path(work) if work else s.work) / CHECK_FILE


def version_of(s):
    try:
        return (Path(s.repo) / "VERSION").read_text().strip() or "unknown"
    except OSError:
        return "unknown"


# systemctl -----------------------------------------------------------------------------------------------------------
def _ctl(host, *args):
    return host.runner.run(["systemctl", "--user", *args])


def active(host, unit=UNIT):
    """True when the unit is installed and systemd says it is active (the service writes no server.pid)."""
    if not installed(host, unit) or host.runner is None:
        return False
    return _ctl(host, "is-active", unit).rc == 0


def state(host, unit=UNIT):
    """(ActiveState, SubState) of the unit. The sub state is asked only for "activating". ("", "") when it is not installed."""
    if not installed(host, unit) or host.runner is None:
        return "", ""
    lines = _ctl(host, "is-active", unit).out.strip().splitlines()
    active_state = lines[-1].strip() if lines else ""
    if active_state != "activating":
        return active_state, ""
    return active_state, _show(host, unit)[1].get("SubState", "")


def up(host, unit=UNIT):
    """True when the unit runs or loads the model: a restart is needed to change its settings."""
    return state(host, unit)[0] in ("active", "activating", "reloading")


def serving(host, unit=UNIT):
    """True when the server process of the unit exists: "active", or "activating" while the model loads (ExecStart runs).
    Not while ExecStartPre runs (that is the start check itself) and not while systemd waits to restart."""
    active_state, sub = state(host, unit)
    if active_state == "active":
        return True
    return active_state == "activating" and sub not in ("start-pre", "condition") and not sub.startswith("auto-restart")


# A setup run that writes new settings sets this flag first and clears it only after the server was restarted with them.
PENDING_KEY = "restart_pending"


def clear_restart_pending(host, s):
    """Remove the flag from setup-state.json (the file stays when it holds other data)."""
    path = s.work / "setup-state.json"
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return
    if isinstance(data, dict) and data.pop(PENDING_KEY, None) is not None:
        Sandbox([s.work]).write_json(path, data)


def manager_ok(host):
    """True when a systemd user manager answers (its state is running, degraded or still starting)."""
    res = _ctl(host, "is-system-running")
    lines = res.out.strip().splitlines()
    return bool(lines) and lines[-1].strip() in STATES_OK


NO_MANAGER = ("No systemd user manager is running here (a container, for example). "
              "Start the server without the service with: ./crucible start")


def _linger(host):
    res = host.runner.run(["loginctl", "show-user", user_name(host), "-p", "Linger"])
    if "Linger=yes" in res.out:
        return True
    if "Linger=no" in res.out:
        return False
    return None


def _show(host, unit):
    args = ["show", unit]
    for name in PROPS:
        args += ["-p", name]
    res = _ctl(host, *args)
    props = {}
    for line in res.out.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            props[key] = value
    return res, props


def _health(host, s, tries=10):
    code = 0
    for i in range(tries):
        code, _ = host.http.get(f"{s.base_url}/health")
        if code == 200:
            return code
        if i + 1 < tries:
            host.sleep(1)
    return code


def _failure_report(host, unit):
    host.out(_ctl(host, "status", unit, "--no-pager", "-n", "0").out.strip())
    host.out("Last 20 log lines:")
    host.out(host.runner.run(["journalctl", "--user", "-u", unit, "-n", "20", "--no-pager", "-o", "short-iso"]).out.strip())
    host.out("More: ./crucible service logs")


# commands ------------------------------------------------------------------------------------------------------------
def install(host, s, no_start=False, start_timeout=None):
    from . import steps
    say = host.out
    if not (s.work / "config.json").exists():
        say("No configuration yet. Run ./crucible setup first.")
        return 2
    state = {}
    try:
        state = json.loads((s.work / "setup-state.json").read_text())
    except (OSError, ValueError):
        pass
    timeout = int(start_timeout) if start_timeout else start_timeout_for(state.get("load_seconds"))
    serve = s.work / "serve.sh"
    try:
        text = render_unit(repo=s.repo, venv_py=s.venv_python, exec_start=serve, bind=s.bind, port=s.port,
                           start_timeout=timeout, version=version_of(s))
    except UnitPathError as exc:
        say(str(exc))
        return 2
    if not manager_ok(host):
        say(NO_MANAGER)
        return 3
    if (s.work / "server.pid").exists():
        say("A server that ./crucible start made runs without the service. The service takes its place.")
        steps.stop_pidfile(host, s)
    steps.Setup(host, s).step_config([])  # serve.sh passes its arguments on ("$@"), so the unit can add --systemd
    box = Sandbox([unit_dir(host)])
    box.write_text(unit_path(host), text, mode=0o644)
    say(f"Wrote {unit_path(host)}")
    _ctl(host, "daemon-reload")
    res = _ctl(host, "enable", UNIT)
    if res.rc != 0:
        say(f"systemctl enable failed: {res.out.strip()}")
        return 1
    _enable_linger(host)
    if no_start:
        say("The service is enabled and not started (--no-start). Start it with: ./crucible service start")
        return 0
    say("Starting the service. This waits until the model is loaded; the first load can take a few minutes.")
    res = _ctl(host, "start", UNIT)
    if res.rc != 0:
        say("The service did not start.")
        _failure_report(host, UNIT)
        return 1
    if _health(host, s) != 200:
        say(f"The service started, but {s.base_url}/health does not answer 200.")
        _failure_report(host, UNIT)
        return 1
    say(f"The service is running at {s.base_url}/v1")
    say("Check it: ./crucible service status    Logs: ./crucible service logs -f    Remove it: ./crucible service uninstall")
    return 0


def _enable_linger(host):
    if _linger(host) is True:
        return
    res = host.runner.run(["loginctl", "enable-linger"])
    if res.rc == 0:
        host.out("Linger is on: the service starts at boot, without a login.")
        return
    host.out("Linger is off, so the service runs only while you are logged in. To start it at boot, run this once: "
             f"sudo loginctl enable-linger {user_name(host)}")


def uninstall(host, s):
    if not installed(host):
        host.out("The service is not installed. Nothing to remove.")
        return 0
    host.out("Stopping the service. " + settings_mod.stop_note(s))
    _ctl(host, "stop", UNIT)
    _ctl(host, "disable", UNIT)
    Sandbox([unit_dir(host)]).remove(unit_path(host))
    _ctl(host, "daemon-reload")
    _ctl(host, "reset-failed", UNIT)
    host.out("The service is removed. Linger stays on. To turn it off: loginctl disable-linger")
    host.out("Start the server again without the service with: ./crucible start")
    return 0


def start(host, s, restart=False):
    _ctl(host, "reset-failed", UNIT)  # a stop after "start-limit-hit" must not block this start
    res = _ctl(host, "restart" if restart else "start", UNIT)
    if res.rc != 0:
        host.out("The service did not start.")
        _failure_report(host, UNIT)
        return 1
    if _health(host, s) != 200:
        host.out(f"The service started, but {s.base_url}/health does not answer 200.")
        _failure_report(host, UNIT)
        return 1
    if restart:
        clear_restart_pending(host, s)  # the server runs with the files on the disk now
    return 0


def stop(host, s):
    host.out("Stopping the service. " + settings_mod.stop_note(s))
    res = _ctl(host, "stop", UNIT)
    if res.rc != 0:
        host.out(f"systemctl stop failed: {res.out.strip()}")
        return 1
    host.out("Stopped.")
    return 0


def logs(host, follow=False, lines=200, unit=UNIT):
    argv = ["journalctl", "--user", "-u", unit, "-n", str(lines), "-o", "short-iso"]
    argv += ["-f"] if follow else ["--no-pager"]
    try:
        res = host.runner.run(argv, echo=True)
    except KeyboardInterrupt:
        return 0
    if "No journal files were found" in res.out:
        host.out("The user journal is not kept on this machine. Ask the administrator to set Storage=persistent "
                 "in /etc/systemd/journald.conf, or read the system journal.")
    return res.rc


# status --------------------------------------------------------------------------------------------------------------
def read_check(path):
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def explain(props, health, check, installed_=True):
    """One sentence that says why the service is down, or that it runs."""
    if not installed_:
        return "The service is not installed. Run: ./crucible service install"
    active, sub, result = props.get("ActiveState", ""), props.get("SubState", ""), props.get("Result", "")
    bad_check = ""
    if check and check.get("ok") is False and check.get("failed"):
        first = check["failed"][0]
        more = len(check["failed"]) - 1
        bad_check = (f" Last start check: {first.get('detail') or first.get('id')}"
                     + (f" (and {more} more problem{'s' if more > 1 else ''}; see: ./crucible doctor --for start)" if more else "") + ".")
    if active == "active" and sub == "running":
        if health == 200:
            return "The service is running."
        return f"The unit is active, but /health did not answer 200 (it answered {health or 'nothing'})."
    if active == "inactive":
        return "The service is stopped. Start it with: ./crucible service start"
    if result == "start-limit-hit":
        return ("Stopped after 4 failed starts in 15 min." + (bad_check + " Fix it, then run: ./crucible service start"
                if bad_check else " See why with: ./crucible service logs. Then run: ./crucible service start"))
    if result == "watchdog":
        return "The watchdog stopped the server because the engine made no progress." + (
            " systemd starts it again." if active != "failed" else " Run: ./crucible service start")
    if result == "timeout":
        return "The server did not get ready in time (TimeoutStartSec)." + bad_check
    if result == "signal":
        return "The server was stopped by a signal." + bad_check
    if active == "activating":
        what = props.get("StatusText") or "starting"
        return f"The service is starting: {what}." + bad_check
    if result == "exit-code" or active == "failed":
        code = props.get("ExecMainStatus", "")
        return (f"The server exited with code {code}." if code not in ("", "0") else "The service failed.") + bad_check
    return f"The service is {active} ({sub})."


def _int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def status(host, s, as_json=False, unit=UNIT, check_file=None):
    res, props = _show(host, unit)
    present = res.rc == 0 and props.get("LoadState", "not-found") != "not-found"
    active, sub = props.get("ActiveState", ""), props.get("SubState", "")
    health = None
    if present and active == "active":
        health, _ = host.http.get(f"{s.base_url}/health")
    check = read_check(check_file_path(host, s, check_file))
    reason = explain(props, health, check, present)
    data = {
        "unit": unit, "installed": present, "state": active, "substate": sub, "nrestarts": _int(props.get("NRestarts")),
        "result": props.get("Result", ""), "exec_main_status": _int(props.get("ExecMainStatus")),
        "status_text": props.get("StatusText", ""), "enabled": props.get("UnitFileState") == "enabled",
        "linger": _linger(host) if present else None, "health": health, "url": s.base_url,
        "key_required": getattr(s, "api_key_file", None) is not None, "since": props.get("ActiveEnterTimestamp", ""),
        "last_start_check": check, "reason": reason,
    }
    if not present:
        code = 4
    elif active == "active" and sub == "running" and health == 200:
        code = 0
    elif active == "inactive":
        code = 3
    else:
        code = 1
    if as_json:
        host.out(json.dumps(data, indent=2, sort_keys=True))
        return code
    if not present:
        host.out(reason)
        return code
    host.out(f"{unit}: {active} ({sub})" + (f" since {data['since']}" if data["since"] and active == "active" else ""))
    host.out(f"  Restarts: {data['nrestarts']}   Result: {data['result'] or 'none'}   Last exit status: {data['exec_main_status']}")
    host.out(f"  Enabled at login: {'yes' if data['enabled'] else 'no'}   Linger (start at boot): "
             f"{ {True: 'yes', False: 'no', None: 'unknown'}[data['linger']] }")
    host.out(f"  URL: {s.base_url}   API key required: {'yes' if data['key_required'] else 'no'}   /health: "
             f"{health if health is not None else 'not asked'}")
    if data["status_text"]:
        host.out(f"  Server says: {data['status_text']}")
    host.out(reason)
    return code
