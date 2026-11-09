"""The command line: setup, doctor, start, stop."""
import argparse
import os
from pathlib import Path

from . import doctor, keyfile, packs, service, settings as settings_mod, steps, update
from .host import Host
from .settings import UNSET

DESCRIPTION = """Run Crucible on a Strix Halo machine.

  ./crucible setup    check the machine, download the pack, build, start the server, test it
  ./crucible doctor   check the machine and print the fix for each problem
  ./crucible start    start the server again after a stop or a restart
  ./crucible stop     stop the server that this tool started
  ./crucible service  install, remove or check the systemd user service (starts at boot, restarts after a fault)
  ./crucible update   move to the newest release (checks every file first, goes back if the new release does not start)
  ./crucible uninstall  remove the service and the install folder (keeps the models unless you add --remove-models)

The command uses only the Python standard library. It never uses sudo, never installs system packages and
never changes the boot loader or the kernel command line: when a system change is needed, it prints the text to apply.
"""


def _address(text):
    """argparse type for --bind: an IPv4 or IPv6 literal, as the server wants it."""
    try:
        keyfile.address(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not an IPv4 or IPv6 address. Use an address such as "
                                         "127.0.0.1, 0.0.0.0 or ::, not a name.") from None
    return text


def _host_name(text):
    """argparse type for --allowed-host: a host name, an IPv4 address or [IPv6], with an optional port."""
    if not settings_mod.host_name_ok(text):
        raise argparse.ArgumentTypeError(f"{text!r} is not a host name. Use a name such as llm.example.org, "
                                         "llm.example.org:8443 or [::1], with no scheme and no path.")
    return text


def _tag(text):
    """argparse type for --tag: a release name such as v1.2.3."""
    if not update.TAG_RE.fullmatch(text):
        raise argparse.ArgumentTypeError(f"{text!r} is not a release name. Use a name such as v1.2.3.")
    return text


def _common(p):
    p.add_argument("--pack", choices=sorted(packs.PACKS), default=None, help="model pack in bits per weight (default 4.05)")
    p.add_argument("--models-dir", type=Path, default=None, help="folder for the pack (default: models/ in this folder)")
    p.add_argument("--state-dir", type=Path, default=None,
                   help="conversation cache on the disk; a new folder inside your home folder (default ~/.cache/crucible/states)")
    p.add_argument("--no-state-dir", action="store_true", help="keep the conversation cache in memory only")
    p.add_argument("--state-disk-gib", type=int, default=None, help="largest size of the cache on the disk (default 200)")
    p.add_argument("--port", type=int, default=None, help="server port (default 8790)")
    p.add_argument("--bind", type=_address, default=None, metavar="ADDRESS",
                   help="address to listen on (default 127.0.0.1). An address that is not loopback "
                        "(0.0.0.0, ::, a LAN address) makes an API key file and the server then needs the key")
    p.add_argument("--api-key", action="store_true",
                   help="also use an API key on 127.0.0.1 (for a reverse proxy). Setup makes the key file")
    p.add_argument("--allowed-host", action="append", type=_host_name, default=None, metavar="NAME",
                   help="a Host name that the server accepts besides localhost, 127.0.0.1, [::1] and the bind address "
                        "(for a reverse proxy; repeat the flag for more). The names replace the saved list")
    p.add_argument("--streams", type=int, choices=range(1, 9), default=None, help="sessions at the same time (default 4)")
    p.add_argument("--ctx", type=int, default=None, help="tokens per session (default 262144)")


def parser():
    ap = argparse.ArgumentParser(prog="crucible", description=DESCRIPTION, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("setup", help="check, download, build, start, test")
    _common(st)
    st.add_argument("--yes", "-y", action="store_true", help="no questions: use the flags and the defaults (for scripts and agents)")
    st.add_argument("--dry-run", action="store_true", help="print the plan and the checks, change nothing")
    st.add_argument("--smoke", action="store_true", help="after the test request, stop the server and exit")
    st.add_argument("--service", choices=("yes", "no"), default=None,
                    help="install the systemd user service (the question of setup); with --yes and no --service: no service")
    st.add_argument("--jobs", type=int, default=None, help="compile jobs (default 1; more need more memory)")
    st.add_argument("--start-timeout", type=int, default=3600, help="seconds to wait for the server (default 3600)")
    dr = sub.add_parser("doctor", help="check the machine")
    _common(dr)
    dr.add_argument("--json", action="store_true", help="print the result as JSON")
    dr.add_argument("--for", dest="for_", choices=("start",), default=None,
                    help="only the checks that block a start; exit 0 ready, 75 try again later, 78 bad setting (used by the service)")
    dr.add_argument("--quiet", action="store_true", help="with --for start: print one line for each failure and nothing else")
    dr.add_argument("--check-file", type=Path, default=None,
                    help="with --for start: where to write the result (default .crucible/last-start-check.json)")
    up = sub.add_parser("update", help="move to the newest release")
    up.add_argument("--yes", "-y", action="store_true", help="no question (for scripts and agents)")
    up.add_argument("--check", action="store_true", help="only say whether a newer release exists; change nothing")
    up.add_argument("--tag", type=_tag, default=None, metavar="vX.Y.Z", help="install this release, also when it is on hold or older")
    up.add_argument("--source-build", action="store_true", help="build the engine from source instead of the release file")
    up.add_argument("--rollback", action="store_true", help="not a flag: a failed start rolls back by itself (see --tag)")
    un = sub.add_parser("uninstall", help="remove the service and the install folder")
    un.add_argument("--yes", "-y", action="store_true", help="no question (for scripts and agents)")
    un.add_argument("--remove-models", action="store_true", help="also remove the model pack and the conversation cache")
    sa = sub.add_parser("start", help="start the server")
    sa.add_argument("--start-timeout", type=int, default=3600)
    sub.add_parser("stop", help="stop the server")
    sv = sub.add_parser("service", help="the systemd user service")
    ssub = sv.add_subparsers(dest="service_cmd", required=True)
    si = ssub.add_parser("install", help="write the unit, enable it, start it")
    si.add_argument("--no-start", action="store_true", help="write and enable the unit, do not start it")
    si.add_argument("--start-timeout", type=int, default=None, help="seconds that systemd waits for the model load")
    ssub.add_parser("uninstall", help="stop, disable and remove the unit")
    ss = ssub.add_parser("status", help="state, restarts, health and why it is down")
    ss.add_argument("--json", action="store_true", help="print the result as JSON")
    ss.add_argument("--unit", default=service.UNIT, help=f"unit name (default {service.UNIT})")
    ss.add_argument("--check-file", type=Path, default=None, help="the last-start-check.json to read")
    sl = ssub.add_parser("logs", help="the journal of the unit")
    sl.add_argument("-f", "--follow", action="store_true", help="keep printing new lines")
    sl.add_argument("-n", "--lines", type=int, default=200, help="lines to print (default 200)")
    for name in ("start", "stop", "restart"):
        ssub.add_parser(name, help=f"{name} the service")
    return ap


def _settings(args, host, repo, **extra):
    return settings_mod.build(
        repo, host, pack=getattr(args, "pack", None), models_dir=getattr(args, "models_dir", None),
        state_dir=getattr(args, "state_dir", None) or UNSET, no_state_dir=getattr(args, "no_state_dir", False),
        state_disk_gib=getattr(args, "state_disk_gib", None), port=getattr(args, "port", None),
        bind=getattr(args, "bind", None), api_key=getattr(args, "api_key", False),
        allowed_hosts=getattr(args, "allowed_host", None),
        streams=getattr(args, "streams", None), ctx=getattr(args, "ctx", None), jobs=getattr(args, "jobs", None), **extra)


def _ask_settings(args, host, repo):
    """Interactive questions for what the flags did not set."""
    ask = host.prompt
    pack = args.pack
    if pack is None:
        pack = ask("Pack: 4.05 (best quality, 107 GB) or 3.05 (smaller, 85 GB)", packs.DEFAULT)
        while pack not in packs.PACKS:
            pack = ask("Type 4.05 or 3.05", packs.DEFAULT)
        args.pack = pack
    probe = _settings(args, host, repo)
    if args.models_dir is None:
        args.models_dir = Path(ask(f"Folder for the pack ({probe.spec.download_gb} GB)", str(probe.models_dir))).expanduser()
    if args.state_dir is None and not args.no_state_dir:
        answer = ask("Folder for the conversation cache on the disk (type none to keep it in memory)", str(probe.state_dir))
        if answer.strip().lower() == "none":
            args.no_state_dir = True
        else:
            args.state_dir = Path(answer).expanduser()
    if args.port is None:
        args.port = int(ask("Server port", str(probe.port)))


def _validate(s, host):
    if s.state_dir is not None:
        home = Path(os.path.abspath(host.home))
        sd = Path(os.path.abspath(s.state_dir))
        if sd == home or home not in sd.parents:
            host.out(f"The state folder {sd} must be a new folder inside your home folder {home}. "
                     "Use --state-dir ~/.cache/crucible/states or --no-state-dir.")
            return False
        problem = settings_mod.state_folder_problem(sd, owned=settings_mod.state_owned(s.repo, sd))
        if problem:
            host.out(f"The state folder is not accepted: {problem} Crucible removes this folder with "
                     "./crucible uninstall --remove-models, so it must be a new or empty folder. "
                     "Use --state-dir ~/.cache/crucible/states or --no-state-dir.")
            return False
    return True


def _ready(host, s):
    host.out(f"Crucible is ready. Base URL: {s.base_url}/v1")
    host.out(keyfile.client_words(s))
    return 0


def _service(args, host, repo):
    s = settings_mod.build(repo, host)
    cmd = args.service_cmd
    if cmd == "install":
        return service.install(host, s, no_start=args.no_start, start_timeout=args.start_timeout)
    if cmd == "uninstall":
        return service.uninstall(host, s)
    if cmd == "status":
        return service.status(host, s, as_json=args.json, unit=args.unit, check_file=args.check_file)
    if cmd == "logs":
        return service.logs(host, follow=args.follow, lines=args.lines)
    if cmd == "stop":
        return service.stop(host, s)
    return service.start(host, s, restart=cmd == "restart")


QUESTION = "Install Crucible as a service? It starts at boot and restarts after a fault. (y/n)"


def service_answer(args, host, print_hint=True):
    """"yes" or "no" for the service question. --service answers it. --yes, or no terminal at all, gives "no" and a hint
    line. Without a systemd user manager the question is not asked. Enter means yes."""
    args.service_hint = False
    if args.service:
        return args.service
    ask = host.prompt if host.is_tty else host.tty_prompt
    if args.yes or ask is None:
        args.service_hint = True
        if print_hint:
            host.out(steps.HINT)
        return "no"
    if not service.manager_ok(host):
        host.out("This machine has no systemd user manager, so the service is not offered. " + service.NO_MANAGER)
        return "no"
    answer = ask(QUESTION, "y")
    return "no" if answer.strip().lower() in ("n", "no") else "yes"


def main(argv=None, host=None, repo=None):
    args = parser().parse_args(argv)
    host = host or Host.real()
    repo = Path(repo) if repo else Path(__file__).resolve().parents[2]
    if args.cmd == "doctor":
        if args.for_ == "start":
            return doctor.for_start(host, _settings(args, host, repo), quiet=args.quiet, as_json=args.json,
                                    check_file=args.check_file)
        return doctor.main(host, _settings(args, host, repo), as_json=args.json)
    if args.cmd == "service":
        return _service(args, host, repo)
    if args.cmd in ("update", "uninstall"):
        ask = host.prompt if host.is_tty else host.tty_prompt
        run = update.run_update if args.cmd == "update" else update.run_uninstall
        return run(host, repo, args, ask)
    if args.cmd == "stop":
        return steps.stop(host, settings_mod.build(repo, host))
    if args.cmd == "start":
        if not (repo / ".crucible/config.json").exists():
            host.out("No configuration yet. Run ./crucible setup first.")
            return 1
        s = settings_mod.build(repo, host)
        if service.installed(host):
            return 1 if service.start(host, s) else _ready(host, s)
        run = steps.Setup(host, s, start_timeout=args.start_timeout)
        results = doctor.run_checks(host, s)
        bad = [r for r in results if r.status == "fail" and "start" in r.blocks]
        if bad:
            for r in bad:
                host.out(f"FAIL  {r.id}: {r.detail}\n      fix: {r.fix}")
            return 1
        try:
            run.step_start([])
        except steps.StepFailed as exc:
            host.out(f"FAILED: {exc}")
            return 1
        return _ready(host, s)
    # setup
    ask = host.prompt if host.is_tty else host.tty_prompt  # a pipe on stdin (curl | bash): ask on the terminal device
    if not args.yes and ask is None:
        host.out("Setup asks questions, and no terminal is attached. Run it in a terminal, "
                 "or add --yes to use the flags and the defaults (for example: ./crucible setup --yes --pack 4.05).")
        return 2
    if args.smoke and args.service == "yes":
        host.out("--smoke stops the server after the test, so it cannot go with --service yes.")
        return 2
    if ask is not None:
        host.prompt = ask
    if not args.yes:
        host.out("Crucible setup. It checks this machine, downloads the model, builds the engine and starts the server.")
        host.out("It never uses sudo and never changes the boot loader. Press Enter to take a default.")
        _ask_settings(args, host, repo)
    s = _settings(args, host, repo)
    if not _validate(s, host):
        return 2
    wants = "no" if args.smoke else service_answer(args, host, print_hint=False)
    if wants == "yes" and not args.dry_run and not service.manager_ok(host):
        host.out(service.NO_MANAGER)
        return 3
    run = steps.Setup(host, s, smoke=args.smoke, start_timeout=args.start_timeout, service=wants,
                      hint=getattr(args, "service_hint", False) and not args.smoke)
    if args.dry_run:
        run.plan()
        results = doctor.run_checks(host, s)
        host.out(doctor.format_text(results))
        return 0
    if not args.yes:
        answer = host.prompt(f"Download {s.spec.download_gb} GB and build now? (y/n)", "y")
        if answer.strip().lower() not in ("y", "yes"):
            host.out("Stopped. Nothing was changed.")
            return 0
    return run.run()
