"""./crucible setup: resumable steps. Each step checks its own result on the disk, so a re-run continues."""
import json
import time
from pathlib import Path

from . import doctor, kernelparams, keyfile, launch, packs, rocm, runtime as runtime_mod, service, settings as settings_mod
from .files import Sandbox
from .launch import process_env

ORDER = ("check", "venv", "download", "build", "config", "start", "test")
NEEDS = {"download": ("venv",), "start": ("venv", "download", "build", "config"), "test": ("start",), "service": ("test",)}
WHAT = {
    "runtime": "download the private ROCm runtime into .crucible (no root, no system change)",
    "check": "check the machine",
    "venv": "make the Python environment in .venv",
    "download": "download the pack",
    "build": "build the engine (make)",
    "config": "write the configuration in .crucible",
    "start": "start the server",
    "test": "send a test request",
    "service": "install the systemd user service (starts at boot, restarts after a fault)",
}
IMPORTS = "import numpy, tokenizers, huggingface_hub, jinja2, PIL"


HINT = "To run it as a service: ./crucible service install"


class StepFailed(Exception):
    pass


class Setup:
    def __init__(self, host, s, smoke=False, start_timeout=3600, service="no", hint=False, runtime="", kernel_ask=None):
        self.config_changed = False  # step_config sets it: the settings on the disk differ from the ones of the last run
        self.host, self.s, self.smoke, self.start_timeout = host, s, smoke, start_timeout
        self.service, self.hint = service, hint
        self.kernel_ask = kernel_ask  # the prompt for the kernel-parameter change; None: print the change and stop
        self.runtime = runtime  # "yes": download the private ROCm runtime first; "offer": a dry run would offer it; "" or "no": not
        roots = [s.work, s.models_dir] + ([s.state_dir] if s.state_dir else [])
        if s.api_key_file is not None:
            roots.append(s.api_key_file.parent)  # the folder of the key file, and nothing else outside the repo
        self.box = Sandbox(roots)
        self.state_file = s.work / "setup-state.json"
        self.state = self._load()
        self.say = host.out
        self.proc = None
        self.blocked = {}

    # state ------------------------------------------------------------------------------------------
    def _load(self):
        try:
            data = json.loads(self.state_file.read_text())
            if isinstance(data.get("steps"), dict):
                return data
        except (OSError, ValueError):
            pass
        return {"version": 1, "steps": {}}

    def _record(self, step, status, seconds=0.0, detail=""):
        self.state["steps"][step] = {"status": status, "seconds": round(seconds, 1), "detail": detail,
                                     "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.host.clock()))}
        self.state["pack"] = self.s.pack
        self.box.write_json(self.state_file, self.state)

    def _log(self, name):
        return self.box.open_log(self.s.work / name)

    # driver -----------------------------------------------------------------------------------------
    @property
    def order(self):
        return ORDER + (("service",) if self.service == "yes" else ())

    def plan(self):
        self.say("Setup plan:")
        if self.runtime in ("yes", "offer"):
            self.say(f"  {'runtime':<9} {WHAT['runtime']}")
        for step in self.order:
            self.say(f"  {step:<9} {WHAT[step]}")

    def run(self):
        h, s = self.host, self.s
        t_all = h.clock()
        done_before = [k for k in ("venv", "download", "build") if self.state["steps"].get(k, {}).get("status") == "done"]
        if done_before:
            self.say("Resuming: " + ", ".join(done_before) + " finished in an earlier run; each is checked again and skipped when complete.")
        self.say(f"Pack {s.pack} ({s.spec.revision}), server on {s.base_url}, work folder {s.work}")
        results = []
        failed = []
        if self.runtime == "yes":
            self._runtime_step()
        for step in self.order:
            deps = [d for d in NEEDS.get(step, ()) if d in failed or d in self.blocked]
            if step in self.blocked:
                self._record(step, "blocked", detail=self.blocked[step])
                failed.append(step)
                self.say(f"[{step}] blocked: {self.blocked[step]}")
                continue
            if deps:
                self._record(step, "skipped", detail="needs " + ", ".join(deps))
                failed.append(step)
                self.say(f"[{step}] skipped: needs {', '.join(deps)}")
                continue
            self.say(f"[{step}] {WHAT[step]} ...")
            t0 = h.clock()
            try:
                note = getattr(self, "step_" + step)(results)
            except StepFailed as exc:
                self._record(step, "failed", h.clock() - t0, str(exc))
                self.say(f"[{step}] FAILED: {exc}")
                failed.append(step)
                continue
            self._record(step, "done", h.clock() - t0, note or "")
            self.say(f"[{step}] done" + (f": {note}" if note else ""))
            if step == "check":
                self._block(results)
                self._kernel_step(results)
        total = h.clock() - t_all
        if failed:
            self._report_failures(results, failed)
            self.say("Run ./crucible setup again after the fix. Finished steps are not repeated.")
            return 1
        self._final(total)
        return 0

    def _runtime_step(self):
        """The private ROCm runtime. A failure is told here and the run goes on: the check then fails with the usual fix."""
        h = self.host
        self.say(f"[runtime] {WHAT['runtime']} ...")
        t0 = h.clock()
        try:
            note = runtime_mod.install(h, self.s, say=self.say)
        except runtime_mod.RuntimeFailed as exc:
            self._record("runtime", "failed", h.clock() - t0, str(exc))
            self.say(f"[runtime] FAILED: {exc}")
            return
        rt = rocm.private(self.s.work)
        self.s.runtime_lib = rt.lib if rt else None
        self._record("runtime", "done", h.clock() - t0, note)
        self.say(f"[runtime] done: {note}")

    def _kernel_step(self, results):
        """The kernel parameters. Prints the exact change when the GPU memory check failed. Applies it only when this run
        has a prompt and the answer is the word yes. A failure is told and the run goes on. Nothing restarts the machine."""
        if not any(r.id == "gtt" and r.status == "fail" for r in results):
            return
        h = self.host
        p = kernelparams.plan(h)
        if p is None:
            self.say("[kernel] The two parameters are on the kernel command line already. Restart the machine, "
                     "then run ./crucible setup again.")
            return
        self.say("[kernel] The kernel command line needs a change.")
        for line in kernelparams.describe(p, h):
            self.say("  " + line)
        if not p.can_apply or self.kernel_ask is None:
            return
        answer = self.kernel_ask("Apply this change now with sudo? Type yes to apply. Any other answer changes nothing", "")
        if answer.strip().lower() != "yes":
            self.say("[kernel] Nothing was changed.")
            return
        try:
            kernelparams.apply(h, p, self.s.work, self.say)
        except kernelparams.Failed as exc:
            self.say(f"[kernel] FAILED: {exc}")
        except OSError as exc:
            self.say(f"[kernel] FAILED: {exc}")

    def _block(self, results):
        steps = {}
        for r in results:
            if r.status == "fail":
                for step in r.blocks:
                    steps.setdefault(step, []).append(r.id)
        self.blocked = {k: "check failed: " + ", ".join(v) for k, v in steps.items()}

    def _report_failures(self, results, failed):
        fails = [r for r in results if r.status == "fail" and r.blocks]
        if fails:
            self.say("")
            self.say("What to fix:")
            shown = set()
            for r in fails:
                self.say(f"- {r.title}: {r.detail}")
                if r.fix in shown:
                    self.say("    fix: the same as above")
                    continue
                shown.add(r.fix)
                for line in r.fix.split("\n"):
                    self.say("    " + line)
        self.say("")
        self.say("Steps not finished: " + ", ".join(failed))

    def _final(self, total):
        s = self.s
        self.say("")
        if self.smoke:
            self.say(f"Smoke run finished in {total:.0f} s. The server was stopped.")
            return
        self.say("Crucible is ready.")
        self.say(f"  Base URL: {s.base_url}/v1")
        self.say(f'  {keyfile.client_words(s)}')
        service_step = self.state["steps"].get("service", {}).get("status") == "done"
        if service_step or service.installed(self.host):
            self.say("  Service: ./crucible service status     Logs: ./crucible service logs -f")
            self.say("  Stop: ./crucible stop     Start: ./crucible start     Remove the service: ./crucible service uninstall")
            self.say("  Server log: ./crucible service logs")
        else:
            self.say("  Stop the server: ./crucible stop     Start it again: ./crucible start")
            self.say(f"  Server log: {s.work / 'server.log'}")
            if self.hint:
                self.say(HINT)
        self.say(f"  Setup took {total:.0f} s in this run.")

    # steps ------------------------------------------------------------------------------------------
    def step_check(self, results):
        results.extend(doctor.run_checks(self.host, self.s))
        bad = [r for r in results if r.status == "fail"]
        for r in results:
            if r.status in ("fail", "warn"):
                self.say(f"  {r.status.upper()}  {r.id}: {r.detail}")
        return f"{len(results) - len(bad)} of {len(results)} checks without a failure" + (f"; {len(bad)} failed" if bad else "")

    def _venv_ready(self):
        s = self.s
        return s.venv_python.exists() and self.host.runner.run([str(s.venv_python), "-c", IMPORTS]).rc == 0

    def step_venv(self, results):
        h, s = self.host, self.s
        if self._venv_ready():
            return "already complete"
        log = self._log("venv.log")
        if not s.venv_python.exists():
            res = h.runner.run([h.python_exe, "-m", "venv", str(s.venv)], cwd=s.repo, log=log)
            if res.rc != 0:
                raise StepFailed("python -m venv failed. On Debian or Ubuntu install the package python3-venv. Output: "
                                 + res.out[-300:])
        req = s.repo / "requirements-cpu.txt"
        for argv in ([str(s.venv_python), "-m", "pip", "install", "-r", str(req)],
                     [str(s.venv_python), "-m", "pip", "install", "-U", "huggingface_hub"]):
            res = h.runner.run(argv, cwd=s.repo, log=log)
            if res.rc != 0:
                raise StepFailed(f"pip failed (log: {s.work / 'venv.log'}). Check the network, then run setup again. "
                                 + res.out[-300:])
        return "made .venv"

    def step_download(self, results):
        h, s = self.host, self.s
        state, detail = packs.verify(s.ckpt)
        if state == "ok":
            return "the pack is on the disk"
        self.box.mkdir(s.models_dir)
        self.say(f"  Downloading about {s.spec.download_gb} GB to {s.ckpt}. A stopped download continues when you run setup again.")
        hf = s.venv / "bin" / "hf"
        argv = [str(hf), "download", packs.REPO, "--revision", s.spec.revision, "--local-dir", str(s.ckpt)]
        res = h.runner.run(argv, cwd=s.repo, log=self._log("download.log"), echo=True)
        if res.rc != 0:
            raise StepFailed(f"the download stopped (log: {s.work / 'download.log'}). Run setup again: it continues. "
                             + res.out[-200:])
        state, detail = packs.verify(s.ckpt)
        if state != "ok":
            raise StepFailed(detail + ". Run setup again: it continues.")
        return "downloaded"

    def step_build(self, results):
        h, s = self.host, self.s
        if s.engine.exists() and self.state["steps"].get("build", {}).get("status") == "done":
            return "already built (delete build/ to build again)"
        rel = doctor.release_engine(s.repo)
        if rel:
            return f"release engine {rel['tag']} (verified)"
        make = h.which("make") or "make"
        env = {"ROCM_PATH": h.env["ROCM_PATH"]} if h.env.get("ROCM_PATH") else None
        if env:
            env = process_env(h.env, env)
        log = self._log("build.log")
        self.say(f"  Building with one compile job; this takes about 16 minutes. Log: {s.work / 'build.log'}")
        argv = [make, f"-j{s.jobs}"]
        res = h.runner.run(argv, cwd=s.repo, env=env, log=log)
        if res.rc != 0 and "argument facts drift" in res.out:
            self.say("  The build stopped with 'argument facts drift' (another ROCm build). Building again with SPLIT=0 (slower).")
            res = h.runner.run(argv + ["SPLIT=0"], cwd=s.repo, env=env, log=log)
        if res.rc != 0 or not s.engine.exists():
            raise StepFailed(f"make failed (log: {s.work / 'build.log'}). Last lines: " + res.out[-400:])
        return "built build/engine/crucible"

    def step_config(self, results):
        s = self.s
        if s.state_dir is not None:
            problem = settings_mod.state_folder_problem(s.state_dir, owned=settings_mod.state_owned(s.repo, s.state_dir))
            if problem:
                raise StepFailed(f"the state folder is not accepted: {problem} Use a new or empty folder (--state-dir) "
                                 "or --no-state-dir.")
            self.box.mkdir(s.state_dir, mode=0o700)
        self.box.mkdir(s.work)
        if s.state_dir is not None:
            self.box.write_text(settings_mod.state_marker_path(s.repo), settings_mod.state_marker_text(s.state_dir))
        if s.api_key_file is not None:
            self.box.check(s.api_key_file)
            if keyfile.ensure_key(self.host, s.api_key_file):
                self.say(f"  Made the API key file {s.api_key_file}. The key is in that file only.")
            # step_check ran before the key existed: ask the key-file check again, so that a key made now does not block the start.
            results[:] = [r for r in results if r.id != "key-file"] + doctor.key_file_results(s)
            self._block(results)
        config, serve = s.to_json(), launch.serve_script(s, s.venv_python)
        old = self._read(s.work / "config.json"), self._read(s.work / "serve.sh")
        changed = old != (json.dumps(config, indent=2, sort_keys=True) + "\n", serve)
        self.config_changed = changed
        if changed and any(old) and not self.state.get(service.PENDING_KEY):
            # A running server keeps the settings that it started with. Write the flag before the files: a run that stops
            # before the restart (a download, for example) must not lose it. step_start clears it after a good restart.
            self.state[service.PENDING_KEY] = True
            self.box.write_json(self.state_file, self.state)
        self.box.write_json(s.work / "config.json", config)
        self.box.write_text(s.work / "serve.sh", serve, executable=True)
        return f"{s.work}/config.json and serve.sh"

    @staticmethod
    def _read(path):
        try:
            return Path(path).read_text()
        except OSError:
            return None

    def _pid(self):
        try:
            return int((self.s.work / "server.pid").read_text().split()[0])
        except (OSError, ValueError, IndexError):
            return None

    def _save_state(self):
        self.box.write_json(self.state_file, self.state)

    def _settings_applied(self):
        """The server runs with the files on the disk now: forget the restart flag."""
        if self.state.pop(service.PENDING_KEY, None) is not None:
            self._save_state()

    def step_start(self, results):
        h, s = self.host, self.s
        if service.installed(h):
            pending = self.config_changed or bool(self.state.get(service.PENDING_KEY))
            restart = pending and service.up(h)
            if restart:
                self.say("  The settings changed and the service runs: restarting it with systemd (it loads the weights again).")
            else:
                self.say("  The service is installed: starting it with systemd.")
            if service.start(h, s, restart=restart) != 0:
                raise StepFailed("the service did not start. See: ./crucible service logs")
            self._settings_applied()
            return f"ready at {s.base_url} (service)"
        pid = self._pid()
        proc = None
        t_spawn = h.clock()
        if pid is not None and h.runner.alive(pid) and (self.config_changed or self.state.get(service.PENDING_KEY)):
            self.say(f"  The settings changed and the server from an earlier run runs (process {pid}): restarting it "
                     "with the new settings (it loads the weights again).")
            if not h.runner.terminate(pid):
                raise StepFailed(f"the server (process {pid}) did not stop yet. Wait, then run setup again.")
            self.box.remove(s.work / "server.pid")
            pid = None
        if pid is not None and h.runner.alive(pid):
            self.say(f"  The server from an earlier run is still running (process {pid}).")
        else:
            env, argv = launch.server_command(s, s.venv_python)
            proc = h.runner.spawn(argv, cwd=s.repo, env=process_env(h.env, env), log=self._log("server.log"))
            pid = proc.pid
            self.box.write_text(s.work / "server.pid", f"{pid} {int(h.clock())}\n")
            self.say(f"  Server process {pid} started. The first start loads the weights; this can take a few minutes.")
            self._settings_applied()
        deadline = h.clock() + self.start_timeout
        last_note = h.clock()
        while True:
            if proc is not None and proc.poll() is not None:
                self.box.remove(s.work / "server.pid")
                raise StepFailed(f"the server stopped at start. Last lines of {s.work / 'server.log'}:\n{self._tail('server.log')}")
            code, _ = h.http.get(f"{s.base_url}/health")
            if code == 200:
                if proc is not None:
                    self.state["load_seconds"] = round(h.clock() - t_spawn, 1)
                return f"ready at {s.base_url}"
            if proc is None and not h.runner.alive(pid):
                self.box.remove(s.work / "server.pid")
                raise StepFailed(f"the server stopped. See {s.work / 'server.log'}")
            if h.clock() > deadline:
                raise StepFailed(f"the server is not ready after {self.start_timeout} s. See {s.work / 'server.log'}")
            if h.clock() - last_note >= 30:
                self.say("  still loading ...")
                last_note = h.clock()
            h.sleep(2)

    def step_service(self, results):
        code = service.install(self.host, self.s)
        if code != 0:
            raise StepFailed(f"the service did not install (exit code {code}). The server may still run. "
                             "Try again with: ./crucible service install")
        return "installed and running"

    def _tail(self, name, n=15):
        try:
            return "\n".join(self.s.work.joinpath(name).read_text(errors="replace").splitlines()[-n:])
        except OSError:
            return "(no log)"

    def step_test(self, results):
        h, s = self.host, self.s
        body = {"model": "crucible", "messages": [{"role": "user", "content": "Say hello."}], "max_tokens": 2048,
                "temperature": 0, "stream": True}
        res = h.http.chat_stream(f"{s.base_url}/v1/chat/completions", body, key_file=s.api_key_file)
        if not res.get("ok"):
            raise StepFailed("the test request got no token: " + str(res.get("error", "")))
        text = " ".join(res["text"].split())[:80]
        note = f"first token after {res['first_token_s']:.2f} s, {res['tokens']} tokens in {res['total_s']:.2f} s"
        self.say(f"  Reply: {text!r}")
        if self.smoke:
            pid = self._pid()
            if pid is not None:
                h.runner.terminate(pid)
                self.box.remove(s.work / "server.pid")
        return note


def stop(host, s):
    if service.installed(host):
        return service.stop(host, s)
    return stop_pidfile(host, s)


def stop_pidfile(host, s):
    box = Sandbox([s.work])
    try:
        pid = int((s.work / "server.pid").read_text().split()[0])
    except (OSError, ValueError, IndexError):
        host.out("No server is recorded in .crucible/server.pid. Nothing to stop.")
        return 0
    if not host.runner.alive(pid):
        box.remove(s.work / "server.pid")
        host.out(f"The recorded process {pid} is not running. Removed the record.")
        return 0
    host.out(f"Stopping the server (process {pid}). " + settings_mod.stop_note(s))
    if host.runner.terminate(pid):
        box.remove(s.work / "server.pid")
        host.out("Stopped.")
        return 0
    host.out(f"The server (process {pid}) did not stop yet. Wait, then run ./crucible stop again.")
    return 1
