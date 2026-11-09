# Quick start: from a new machine to a running server

This guide sets up Crucible the way we run it every day:

- the **4.05 bpw pack** (the higher-quality pack),
- the n-gram table read from the SSD (`CRUCIBLE_PLE_HOST=1`),
- the conversation cache on the SSD (`--state-dir`), so a later turn or a restarted server continues without a full prefill,
- 4 sessions of 262,144 tokens, image input, and an OpenAI-compatible server on `127.0.0.1:8790`.

The 3.05 bpw pack (faster, lower quality) and the 16-bit n-gram table are options at the end.

| | |
|---|---|
| Machine | Ryzen AI Max+ 395 with Radeon 8060S (gfx1151) and 128 GB of memory. Other machines are not supported. |
| Download | About 107 GB (the 4.05 pack). |
| Disk | About 110 GB for the pack, plus up to 200 GiB (215 GB) for the conversation cache. Use an SSD. |
| Time | One command: the pack download, plus about 1 to 2 minutes (it downloads a 25 MB engine; it does not build). Steps 3 to 6 by hand: the download, plus about 16 minutes for the build. |

## One command

Run this on a machine that has ROCm 7.2.4 and GPU access (step 1 below):

```sh
curl -fsSL https://raw.githubusercontent.com/jaedync/crucible-strix-halo/main/install.sh | bash
```

The script downloads the release, checks its sha256 and starts `./crucible setup`. Setup asks a few questions.
Press Enter to take a default. It downloads the model (about 107 GB) and starts the server. The default install
folder is the folder `crucible` in your home folder, and the default port is 8790.

The one command does not run `make`. It downloads the release engine (about 25 MB), checks its sha256, and puts it in
`build/engine/`. The line `[build] done: release engine v0.34.0 (verified)` means this. So the 16 minutes of step 4
do not apply, and no compiler is needed. In our test, with the pack already on the disk, setup took 65 seconds.
With a new pack, the download time is added, and the first model load takes a few minutes. Add `--source-build` to
build the engine from source instead (about 16 minutes).

What the one command checks for steps 1 and 2:

- The installer checks the tools (`git`, `curl`, `python3`, `gzip`, `sha256sum`) and the libraries of the engine.
  It does not install ROCm, and it does not change the kernel command line.
- Setup then prints `[check] done: N of N checks without a failure`. The checks cover the Linux system, Python 3.12 or
  newer, the ROCm version (7.2.4; another version is a warning), `rocminfo` with `gfx1151`, access to `/dev/kfd` and a
  render node (the result of the `groups` check of step 1), 120 GiB or more of memory, and the GPU memory limit
  (`mem_info_gtt_total` is 130 GB or more, the check of step 2). They also cover free GPU memory, disk space, the state
  folder and the port.
- A check that fails prints what to fix. Setup then does not start the server. It still downloads the pack first,
  so do steps 1 and 2 before the one command.
- The checks do not look at the `pacman` package, the group names, the kernel parameter text or `amd_iommu`. They test
  the result. If every check passes, you do not need to run the check commands of steps 1 and 2 again.
- The checks cannot do the work of steps 1 and 2: install ROCm 7.2.4, and put the two parameters on the kernel
  command line (then restart). Do these by hand first.

Setup also sends one test request ("Say hello.") when the server is ready. This is the same as step 7. It shows in the
server log as one request before yours. With the service, setup stops this first server and lets the service start the
server again, so the model loads twice. The log of each step is in `.crucible/`. The file `.crucible/setup-state.json`
holds the time of each step. The installer log has no times.

The pack goes to the folder `models/` in the install folder (change it with `--models-dir`), in the folder
`Qwen3.8-Flash-Next-exl3-4.05bpw_h6_ng6`. Setup downloads it with the `hf download` command of step 5. If that folder
holds the whole pack already (every file of the index), setup does not download it again.

Setup asks one more question: "Install Crucible as a service?" The default is yes.
A service starts at boot and restarts after a fault.

To answer all questions in advance, add flags after `bash -s --`:

```sh
curl -fsSL https://raw.githubusercontent.com/jaedync/crucible-strix-halo/main/install.sh | bash -s -- --yes --service yes --port 8790
```

- `--yes` takes the defaults and asks nothing.
- `--yes` alone does not install the service. Setup prints a hint line instead.
- `--service yes` or `--service no` answers the service question.
- `--port N` sets the server port. Use it when port 8790 is busy.
- `--dir FOLDER` sets the install folder. The folder must be new or empty. The installer refuses a folder that holds other files.
- `--models-dir FOLDER` names the folder of the model pack. If the pack is there, setup does not download it.
- `--no-state-dir` keeps the conversation cache in memory only. Nothing is saved at a stop. The stop messages then say
  that the cache is in memory only and is lost.

Setup never uses `sudo`. It never changes the boot loader. The installer asks once before it runs `sudo` for a
missing package, and it shows the exact command first.

After the install, the server is ready. Go to step 7 to send a test request (use your port). The steps below are the
same install in single steps, for a person or an agent that does not use the script.

## For an LLM agent that follows this guide

- Do the steps in order. Each step ends with a **Check**. Run it and compare the result.
- If a check fails, you may use only the fallback that the step names. If the fallback also fails, stop. Report the
  step, the command and its full output to the user. Do not try other changes.
- Step 1 needs `sudo`. Get permission before you use it. Use `sudo` nowhere else.
- Do not run any command of step 2. It changes the kernel command line, and an error there can stop the machine from
  starting. Show the user the parameters and the instructions, then wait for a confirmation that the machine has
  restarted. Never restart the machine yourself.
- The "One command" section does the work of steps 3 to 6 in one run. Steps 1 and 2 still come first: do them by
  hand, because the command does not install ROCm and does not change the kernel command line. The setup checks of the
  command test the result of steps 1 and 2 (see "What the one command checks"). If you use the command, you can skip
  steps 3 to 6 and go to step 7. Setup sends its own test request.
- Step 6 runs the server until you stop it. Start it in the background as the step shows, and read its log.

## Step 1: ROCm 7.2.4, the HIP compiler and GPU access

The engine needs the HIP runtime and compiler of ROCm 7.2.4. It uses no ROCm libraries (no rocBLAS, no hipBLAS).
The tested system is Arch Linux with kernel 7.2. Other distributions and other ROCm versions are not tested.

On Arch Linux:

```sh
sudo pacman -Syu --needed base-devel git python hip-runtime-amd rocm-llvm rocminfo
```

`-Syu` also updates the rest of the system (Arch does not support a partial update). If the kernel was updated,
restart before step 2.

On another distribution, install ROCm 7.2.4 with AMD's instructions for your distribution
(https://rocm.docs.amd.com). You need `hipcc` and `rocminfo`.

Your user must be able to open the GPU. Run `groups`. If `render` or `video` is not in the list:

```sh
sudo usermod -aG render,video "$USER"
```

Then log out and log in again (or restart).

**Check:**

```sh
pacman -Q hip-runtime-amd                      # Arch: expect hip-runtime-amd 7.2.4-1
/opt/rocm/bin/hipcc --version | head -1
/opt/rocm/bin/rocminfo | grep -m1 -o gfx1151   # expect: gfx1151
groups                                         # expect: render and video in the list
```

## Step 2: let the GPU use the memory

The GPU of Strix Halo uses the system memory. By default, Linux lets it use only part of it. The 4.05 pack with
4 long sessions needs most of the 128 GB.

First run the check at the end of this step. If it already shows the values, skip this step.

The test machine has these kernel parameters:

```text
amdgpu.gttsize=126976 ttm.pages_limit=32505856
```

The test machine also has `amd_iommu=off`, and the GPU memory (UMA frame buffer) in the firmware setup is set to
512 MB. We did not test without them.

Add the two parameters to the kernel command line. Keep every parameter that is already there (see `cat /proc/cmdline`),
and do not add a parameter twice. Where the line is depends on how your system starts:

- systemd-boot with entry files: the `options` line of your entry in `/boot/loader/entries/`.
- GRUB: `GRUB_CMDLINE_LINUX_DEFAULT` in `/etc/default/grub`, then `sudo grub-mkconfig -o /boot/grub/grub.cfg`.
- A unified kernel image (UKI): usually `/etc/kernel/cmdline`, then build the image again with your tool
  (for example `sudo mkinitcpio -P`).
- Another boot loader: see its documentation.

Keep a working boot entry until the new one starts. An error in the command line can stop the machine from starting.
Then restart.

**Check** (after the restart):

```sh
tr ' ' '\n' < /proc/cmdline | grep -E 'gttsize|pages_limit'
cat /sys/class/drm/card*/device/mem_info_gtt_total 2>/dev/null   # expect one line: about 133000000000
```

The GPU can then use about 124 GiB of the 128 GiB. Do not run a desktop session or other large programs while the
server runs.

## Step 3: the code and the Python environment

Python 3.12 or newer is needed:

```sh
python3 -c 'import sys; print(sys.version_info >= (3, 12))'   # expect: True
```

```sh
git clone https://github.com/jaedync/crucible-strix-halo.git
cd crucible-strix-halo
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-cpu.txt
python -m pip install -U huggingface_hub
```

Run every later step in this folder, with the environment active. In a new terminal, run `cd crucible-strix-halo`
and `. .venv/bin/activate` again.

**Check:**

```sh
python -m server --version    # expect: crucible 0.32.0 or newer
hf version
```

## Step 4: build the engine

```sh
make
```

The build uses one compile job and takes about 16 minutes. More jobs (`make -j2`) need more memory; see
[BUILD](BUILD.md).

Fallback: if `make` stops with `argument facts drift` (another ROCm build), run `make SPLIT=0`. It builds with plain
`hipcc`, which takes longer. Please report the drift as an issue.

**Check:**

```sh
test -x build/engine/crucible && echo ok   # expect: ok
```

## Step 5: download the 4.05 pack

Read the [model license note](THIRD_PARTY.md#model-weights) first. The weights use the Qwen Community License 1.0.

Check the free disk space and memory first:

```sh
df -h .        # need 110 GB free here (the pack), plus 215 GB where the cache goes if you use step 6 as written
free -g        # expect: total 120 or more
```

```sh
PACK=4.05bpw_h6_ng6
CKPT="$PWD/models/Qwen3.8-Flash-Next-exl3-$PACK"
hf download turboderp/Qwen3.8-Flash-Next-exl3 --revision "$PACK" --local-dir "$CKPT"
```

The download is about 107 GB. Fallback: if it stops, run the same command again. It continues.
Keep every file of the pack: the tokenizer files, all weight shards, `ngram_embedding.safetensors` and the MTP files.
Do not change, move or download the pack again while the server runs: the engine reads the n-gram table from the file.

`PACK` and `CKPT` exist only in this terminal. In a new terminal, run the first two lines again.

**Check:**

```sh
ls "$CKPT"/ngram_embedding.safetensors "$CKPT"/config.json "$CKPT"/tokenizer.json
du -sh --apparent-size "$CKPT"    # expect: 100G
```

## Step 6: start the server

```sh
mkdir -p ~/.cache/crucible/states
CRUCIBLE_PLE_HOST=1 CRUCIBLE_COMPACT_STATE=1 CRUCIBLE_PLE_FETCH=1 \
CRUCIBLE_DRAFT_HEAD=q2 CRUCIBLE_DRAFT_CHAIN=hc8,lead CRUCIBLE_PF_LAUNCH_ROWS=16384+128 \
nohup python -m server "$CKPT" --engine build/engine/crucible \
  --streams 4 --ctx 262144 --prefill 4096 \
  --conversation-cache-mib 9700 --host-reserve-mib 11264 \
  --state-dir ~/.cache/crucible/states --state-disk-gib 200 \
  --vision --clamp-max-tokens --default-thinking on --max-body-mib 16 \
  > server.log 2>&1 &
```

The server is ready when `server.log` shows `Crucible is ready at http://127.0.0.1:8790`. The first start loads the
weights and can take a few minutes. To stop the server, run `kill %1` in the same terminal. From another terminal, find its process ID with
`pgrep -af 'python -m server'` and stop it with `kill`.

What each part does:

| Part | Effect |
|---|---|
| `CRUCIBLE_PLE_HOST=1` | The n-gram table (39 GB) stays on the SSD. The engine reads the rows that it needs; the page cache keeps the frequent ones. |
| `CRUCIBLE_COMPACT_STATE=1` `CRUCIBLE_PLE_FETCH=1` | Compact sequence storage, and read hints for the n-gram table. |
| `CRUCIBLE_DRAFT_HEAD=q2` `CRUCIBLE_DRAFT_CHAIN=hc8,lead` | Faster drafts. They change the drafts only; the verify pass checks every draft. |
| `CRUCIBLE_PF_LAUNCH_ROWS=16384+128` | Prefill launches of 16,384 rows. This is the engine default; it is named here to match the speed table. |
| `--streams 4 --ctx 262144` | 4 sessions at the same time, each up to 262,144 tokens. |
| `--conversation-cache-mib 9700` | Memory for the states of earlier turns, so a later turn pays only for its new tokens. Without it (the default of `python -m server` is 0) every turn prefills the whole conversation again. |
| `--state-dir ... --state-disk-gib 200` | States that leave the memory cache go to the SSD (at most 200 GiB), also at a stop. The folder must be a new folder inside your home folder and hold nothing else. A guard limits the writes to 300 GiB per day. |
| `--vision` | Accepts images in requests. |
| `--clamp-max-tokens` | Lowers `max_tokens` to the room that is left instead of refusing the request. |
| `--default-thinking on` | The model thinks when a request does not say. This is the model's own default. |

If the server stops with an out-of-memory message, lower `--streams` or `--ctx`. The server needs no key on
`127.0.0.1`, and it has no TLS. Keep the default address, or read "Other machines and the API key" below
([security](../SECURITY.md)). Once a day at start, the server asks GitHub if
a newer release exists; `--no-update-check` turns this off.

**Check:**

```sh
grep -m1 'Crucible is ready at' server.log
curl -sS http://127.0.0.1:8790/health    # expect: {"status": "ok"}
```

## Run as a service

The service is a systemd user unit. It runs the server as your user. It starts at boot and restarts after a fault.
Use it instead of the manual start of step 6. If a server from step 6 runs, stop it first: two servers cannot use one port.
Run the commands in the folder of the install (the folder `crucible` in your home folder after the one command, or the folder of step 3).

Install it:

```sh
./crucible service install
```

The one command (with `--service yes`) does this for you. It also turns on linger (see "Start at boot").

`./crucible service install` needs the configuration that `./crucible setup` writes. If you did the steps by hand and never ran `./crucible setup`, the command prints "No configuration yet. Run ./crucible setup first." Run `./crucible setup`, then run the install again.

The command writes `~/.config/systemd/user/crucible-server.service`, enables it and starts it. It waits until the
model is loaded. The first load takes a few minutes. Do not copy the unit file by hand.

Before each start, the unit runs `./crucible doctor --for start`. If the GPU memory is in use or the port is busy,
the check waits up to 2 minutes. If a setting is wrong (for example the model folder is missing), the start stops with
a clear reason.

Commands:

```sh
./crucible service status       # state, restarts, health, and why it is down
./crucible service status --json
./crucible service logs -f      # the journal of the unit
./crucible service restart
./crucible service stop
./crucible service uninstall    # stops and removes the unit
```

After 4 failed starts in 15 minutes, systemd stops the restarts. `service status` then names the failed check. Fix the
cause, then run `./crucible service start`.

When the service is installed, `./crucible start` and `./crucible stop` use the service.

### Start at boot

A user service runs only while you are logged in, unless "linger" is on. `./crucible service install` (and the one
command with `--service yes`) turns linger on if it is off. If linger is on already, it does nothing. Check it with
`./crucible service status` (the line "Linger (start at boot)").

If the system refuses it, the tool prints a line for you. You can also run this once yourself:

```sh
loginctl enable-linger "$USER"
```

If that is refused too, run `sudo loginctl enable-linger "$USER"`. This tool never runs `sudo`.
`./crucible service uninstall` leaves linger on. To turn it off: `loginctl disable-linger`.

### Remove Crucible

To remove the service and the install folder, run this in the install folder:

```sh
./crucible uninstall --yes
```

The model packs stay on the disk. This holds for every pack folder in the models folder (4.05 and 3.05), also
when the models folder is inside the install folder. Add `--remove-models` to delete them too (about 107 GB for the
4.05 pack).

### Other machines and the API key

On `127.0.0.1` the server needs no key. To reach it from other machines, see [Remote access](REMOTE_ACCESS.md).
A server on another address needs an API key file. Setup makes it.

## Step 7: send a test request

With the one command, setup has sent one test request already (the reply "Hello!" is in its output). Send yours as
well to see the full reply.

```sh
curl -sS http://127.0.0.1:8790/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"crucible","messages":[{"role":"user","content":"Say hello."}],"max_tokens":2048}'
```

**Check:** the reply is JSON, and `choices[0].message.content` holds a short answer. With thinking on, the reply
also holds the model's reasoning. A `max_tokens` below 1,280 turns thinking off for that request.

To use Crucible from a client, set the base URL to `http://127.0.0.1:8790/v1`, the model to `crucible`, and any
API key. More in [OpenAI clients](OPENAI_CLIENTS.md).

To watch the engine live (requests, rates, memory, engine switch), see
[Kiln](https://github.com/jaedync/crucible-kiln), a dashboard and OpenAI-compatible gateway for Crucible.

The default prefill launch rows (`CRUCIBLE_PF_LAUNCH_ROWS=16384+128`) fit 4 sessions of 262K with `--vision` on the 4.05 pack (stage arena 2,174 MiB); `8192+128` uses 1,439 MiB and gives a slower cold prefill.

## Option: the 3.05 bpw pack

The 3.05 pack is about 85 GB and decodes faster, at a lower quality ([Quality](QUALITY.md)). Use
`PACK=3.05bpw_h5_ng5` in step 5 (the `du` check then shows about 80G). Step 6 is the same for this pack.

## Option: the 16-bit n-gram table (experimental)

The 4.05 pack stores its n-gram table at 6 bits. A 16-bit copy from the original weights gave a 5.7% lower
KL divergence in our test. Decode speed changed between -1.3% (4 sessions) and +2.2% (code with speculation).

- It costs a 104.3 GB download and 102.4 GB of disk (207 GB during the build). Put `--out` on a large SSD, not in
  the repo folder.
- Stop the server first: the build needs 15.6 GB of free memory and takes about 26 minutes.
- The file does not fit in memory with the model, so the engine reads it from the SSD all the time.
- The steps are in [NGRAM16](NGRAM16.md). Then add
  `CRUCIBLE_PLE_BF16=/path/to/ngram16/ngram_embedding_bf16.safetensors` to the start command of step 6.
- A start with this setting does not reuse the cached conversation states of a start without it.

## When a step fails

| Problem | What to do |
|---|---|
| `hipcc: not found` or no `gfx1151` in `rocminfo` | Step 1 is not complete. Check that `/opt/rocm/bin` exists, and that `groups` shows `render` and `video` (log in again after `usermod`). |
| `make` stops with `argument facts drift` | Run `make SPLIT=0`. |
| The server stops at start with an out-of-memory message | Do the check of step 2. Close other programs that use memory. Try `--streams 2` or `--ctx 131072`. |
| The download stops | Run the same `hf download` command again. |
| `Address already in use` | Another server uses port 8790. Stop it, or add `--port 8791`. |
| `--state-dir` refuses the folder | The folder must be a new, empty folder inside your home folder. |
| `"$CKPT"` is empty | You are in a new terminal. Run the `PACK=` and `CKPT=` lines of step 5 again. |

Not tested: a new installation of Arch Linux or another distribution from zero. The commands of steps 3 to 7 are
the ones that we run on the test machine. Please report a step that does not work for you as an issue.
