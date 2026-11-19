#!/usr/bin/env bash
# Crucible installer.
#
#   curl -fsSL https://raw.githubusercontent.com/jaedync/crucible-strix-halo/main/install.sh | bash
#   curl -fsSL https://raw.githubusercontent.com/jaedync/crucible-strix-halo/main/install.sh | bash -s -- --yes --pack 4.05
#
# What it does, in order:
#   1. checks this system and the tools it needs (git, curl, python3, gzip, sha256sum)
#   2. downloads SHA256SUMS and the manifest of the pinned release tag into a temp folder
#   3. downloads the release engine (gfx1151, built with ROCm 7.2.4; it runs on 7.2.4 and on 10.1) into the same folder
#   4. checks the sha256 of every download. It runs nothing before this check passes
#   5. checks that the libraries of the engine exist on this machine. A missing package is shown as an exact command,
#      and the installer asks once before it runs sudo
#   6. clones the source at the pinned tag (a normal git checkout, so updates work)
#   7. puts the engine in build/engine/ and hands over to ./crucible setup
# It never touches the boot loader, the kernel command line or any file outside the install folder, the install log and
# the folders of a package install that you approve. The whole script is functions plus one last line, so a download
# that stops half way runs nothing.
#
# Environment (for mirrors and tests): CRUCIBLE_TAG, CRUCIBLE_RELEASE_BASE (https:// or file://), CRUCIBLE_REPO_URL,
# CRUCIBLE_DIR, CRUCIBLE_SUMS_SHA256, CRUCIBLE_TTY, CRUCIBLE_OS_RELEASE, CRUCIBLE_PROC, CRUCIBLE_SYS, ROCM_PATH.

# PIN_SUMS_SHA256: the sha256 of SHA256SUMS of release PIN_TAG. The release step sets it in one commit on main after the
# release (tools/release_pin.py), so a tag's own copy has it empty. Scope of the pin:
#   it makes a replaced release file visible as a commit on main (who replaces SHA256SUMS must change this line too);
#   it does not protect against a person who can push to main.
# An empty pin gives a WARN line; a pin that is present and does not match stops the install (exit 4).
PIN_TAG="v0.38.0"
PIN_SUMS_SHA256=""
REPO_URL_DEFAULT="https://github.com/jaedync/crucible-strix-halo.git"
RELEASE_BASE_DEFAULT="https://github.com/jaedync/crucible-strix-halo/releases/download"
GPU_ARCH_NAME="gfx1151"
ROCM_SERIES="7.2"
TOTAL_STEPS=7
RAW_URL="https://raw.githubusercontent.com/jaedync/crucible-strix-halo/main/install.sh"
PIPE_CHAR="|"

# The options of ./crucible setup of release PIN_TAG and the values that each one takes. install.sh checks the options with
# this list before any download. tests/test_install_sh.py keeps it equal to tests/data/setup_options/<PIN_TAG>.txt; the
# release step writes both with `python3 -I tools/setup_options.py record <tag>` on the release commit.
SETUP_OPTIONS="
-h flag
--help flag
--pack choice:3.05,4.05,4.18
--models-dir value
--state-dir value
--no-state-dir flag
--state-disk-gib int
--state-write-gib-day int:0-65536
--no-disk-check flag
--port int
--bind value
--api-key flag
--no-api-key flag
--allowed-host value
--streams int:1-8
--ctx int
--layout choice:default,long,two-long
--yes flag
-y flag
--dry-run flag
--smoke flag
--service choice:yes,no
--private-runtime choice:yes,no
--ngram16 choice:yes,no
--abliterated choice:yes,no
--download-anyway flag
--jobs int
--start-timeout int
--allow-rocm10 flag
"
# The disk size of each pack in GB (tools/crucible_cli/packs.py) and the smallest memory in KiB (tools/crucible_cli/doctor.py).
PACK_DISK_GB_405=110
PACK_DISK_GB_305=90
PACK_DISK_GB_418=125
MEM_MIN_KB=125829120
GTT_MIN_BYTES=130000000000
# The 96 GB class (tools/crucible_cli/machine.py): from 88 GiB, the 3.05 pack with 2 sessions; the GTT limit within 3 GiB
# of MemTotal less 1 GiB.
MEM_MID_KB=92274688

usage() {
  command_text "$(usage_text)"
  printf '\n'
}

usage_text() {
  cat <<'EOF'
Crucible installer

Usage:
  install.sh [install] [options] [setup options]   install Crucible and run ./crucible setup
  install.sh status                                show what is installed
  install.sh update [options]                      update to the newest release
  install.sh uninstall [--yes] [--remove-models]   remove Crucible (asks once)
  install.sh engine                                fetch the release engine of the checked out tag (used by make update)

Options:
  --dir DIR         install folder (default $HOME/crucible)
  --tag vX.Y.Z      install this release (alias: --version); overrides CRUCIBLE_TAG
  --source-build    do not download the engine (the manifest is still checked); ./crucible setup builds it from source (about 16 minutes with the ROCm 7.2.4 compiler)
  --dry-run         show the plan and run the read-only machine checks; change nothing (exit 6: this machine cannot run Crucible)
  --yes, -y         no questions in ./crucible setup (the sudo question is never answered for you)
  --remove-models   with uninstall: also remove the model pack and the state folder
  -h, --help        show this text
Every other option goes to ./crucible setup (for example --pack 4.05 or --state-dir DIR). The installer checks these options
before any download: use the full names (./crucible setup --help lists them).

Steps: check the system, download SHA256SUMS and the manifest, download the engine, check every sha256, check the libraries,
get the source at the pinned tag, put the engine in place and run setup. Nothing runs before the sha256 check passes.
The installer asks once before it runs sudo, and shows the exact command first. It never changes the boot settings.
The log is $XDG_STATE_HOME/crucible/install.log (default ~/.local/state/crucible/install.log).
Exit codes: 0 done, 2 bad use or bad folder, 3 a package is needed (run the shown command, then run this again),
4 a download or a check failed, 5 the release engine cannot run on this machine, 6 this machine cannot run Crucible
(the dry run and step 1 check the GPU, the memory, the GPU memory limit, the HIP runtime and the disk; nothing is downloaded).
EOF
}

# ---- output and logging -------------------------------------------------------------------------------------------

log_line() {
  [ -n "${LOG:-}" ] && [ -w "$(dirname "$LOG")" ] && printf '%s\n' "$*" >>"$LOG"
  return 0
}

command_text() {
  # No ${text//pat/rep}: with bash 5.2 patsub_replacement, an & in the replacement is the matched text, and that drops the
  # backslash that %q puts before &. Cut and join the text instead.
  local launcher rest="$*" out=""
  printf -v launcher '%q' "${DIR:-${CRUCIBLE_DIR:-$HOME/crucible}}/crucible"
  while [[ $rest == *./crucible* ]]; do
    out+="${rest%%./crucible*}$launcher"
    rest=${rest#*./crucible}
  done
  printf '%s' "$out$rest"
}

say() {
  local text
  text=$(command_text "$*")
  printf '%s\n' "$text"
  log_line "$text"
}

warn() {
  local text
  text=$(command_text "$*")
  printf '%s\n' "$text" >&2
  log_line "$text"
}

die() {
  local code=$1
  shift
  local text
  text=$(command_text "$*")
  printf 'install: %s\n' "$text" >&2
  log_line "install: $text"
  [ -n "${LOG:-}" ] && printf 'Log: %s\n' "$LOG" >&2
  exit "$code"
}

step() {
  say "[$1/$TOTAL_STEPS] $2"
}

cleanup() {
  [ -n "${WORK:-}" ] && [ -d "$WORK" ] && rm -rf -- "$WORK"
  [ -n "${CLONE_TMP:-}" ] && [ -d "$CLONE_TMP" ] && rm -rf -- "$CLONE_TMP"
  return 0
}

# ---- the terminal -------------------------------------------------------------------------------------------------

tty_file() {
  printf '%s' "${CRUCIBLE_TTY:-/dev/tty}"
}

have_tty() {
  local t
  t=$(tty_file)
  { : <"$t"; } 2>/dev/null
}

# ask_yn TEXT: one question, read from the terminal (stdin is the script when you pipe it). Anything but y or yes is no.
ask_yn() {
  local answer=""
  have_tty || return 1
  printf '%s (y/N) ' "$1"
  IFS= read -r answer <"$(tty_file)" || answer=""
  case "$answer" in
    y | Y | yes | YES | Yes) return 0 ;;
  esac
  return 1
}

# ---- arguments ----------------------------------------------------------------------------------------------------

parse_args() {
  CMD=install
  IN_CLONE=0
  DIR_ARG=""
  TAG_ARG=""
  DRY_RUN=0
  SOURCE_BUILD=0
  YES=0
  REMOVE_MODELS=0
  PASS=()
  local seen_cmd=0
  while [ $# -gt 0 ]; do
    case "$1" in
      install | status | update | uninstall | engine)
        if [ "$seen_cmd" = 0 ]; then
          CMD=$1
          seen_cmd=1
        else
          PASS+=("$1")
        fi
        ;;
      --dir)
        [ $# -ge 2 ] || die 2 "--dir needs a folder."
        DIR_ARG=$2
        shift
        ;;
      --dir=*) DIR_ARG=${1#--dir=} ;;
      --tag | --version)
        if [ $# -lt 2 ] || [ -z "$2" ]; then die 2 "$1 needs a release name, such as v0.37.2."; fi
        TAG_ARG=$2
        shift
        ;;
      --tag=* | --version=*)
        TAG_ARG=${1#*=}
        [ -n "$TAG_ARG" ] || die 2 "The release name must not be empty."
        ;;
      --dry-run) DRY_RUN=1 ;;
      --source-build) SOURCE_BUILD=1 ;;
      --remove-models) REMOVE_MODELS=1 ;;
      --yes | -y)
        YES=1
        PASS+=("--yes")
        ;;
      -h | --help) CMD=help ;;
      --)
        shift
        PASS+=("$@")
        break
        ;;
      *) PASS+=("$1") ;;
    esac
    shift
  done
  if [ "$CMD" = update ] && [ -n "$TAG_ARG" ]; then
    PASS+=("--tag" "$TAG_ARG")
  fi
}

# ---- the system ---------------------------------------------------------------------------------------------------

check_platform() {
  [ "$(uname -s)" = "Linux" ] || die 2 "Crucible runs on Linux only. This system is $(uname -s)."
  [ "$(uname -m)" = "x86_64" ] || die 2 "Crucible runs on x86-64 only. This machine is $(uname -m)."
}

os_family() {
  local file id like text
  file=${CRUCIBLE_OS_RELEASE:-/etc/os-release}
  id=$(sed -n 's/^ID=//p' "$file" 2>/dev/null | head -n 1 | tr -d '"')
  like=$(sed -n 's/^ID_LIKE=//p' "$file" 2>/dev/null | head -n 1 | tr -d '"')
  text=" $id $like "
  case "$text" in
    *" arch "* | *" manjaro "*) echo arch ;;
    *" debian "* | *" ubuntu "*) echo debian ;;
    *" fedora "* | *" rhel "* | *" centos "*) echo fedora ;;
    *) echo unknown ;;
  esac
}

# pkg_for FAMILY NAME: the package that holds a library (by soname) or a tool. Empty when not known.
pkg_for() {
  local family=$1 name=$2
  case "$family:$name" in
    arch:libelf.so.1) echo libelf ;;
    debian:libelf.so.1) echo libelf1t64 ;;
    fedora:libelf.so.1) echo elfutils-libelf ;;
    arch:libdrm.so.2 | arch:libdrm_amdgpu.so.1) echo libdrm ;;
    debian:libdrm.so.2) echo libdrm2 ;;
    debian:libdrm_amdgpu.so.1) echo libdrm-amdgpu1 ;;
    fedora:libdrm.so.2 | fedora:libdrm_amdgpu.so.1) echo libdrm ;;
    arch:libnuma.so.1) echo numactl ;;
    debian:libnuma.so.1) echo libnuma1 ;;
    fedora:libnuma.so.1) echo numactl-libs ;;
    arch:libz.so.1) echo zlib ;;
    debian:libz.so.1) echo zlib1g ;;
    fedora:libz.so.1) echo zlib-ng-compat ;;
    arch:libzstd.so.1) echo zstd ;;
    debian:libzstd.so.1) echo libzstd1 ;;
    fedora:libzstd.so.1) echo libzstd ;;
    arch:libfmt.so.12) echo fmt ;;
    fedora:libfmt.so.12) echo fmt ;;
    arch:libglog.so.2) echo google-glog ;;
    debian:libglog.so.1) echo libgoogle-glog0v6t64 ;;
    fedora:libglog.so.2) echo glog ;;
    arch:libstdc++.so.6) echo gcc-libs ;;
    debian:libstdc++.so.6) echo libstdc++6 ;;
    fedora:libstdc++.so.6) echo libstdc++ ;;
    arch:tool-git | debian:tool-git | fedora:tool-git) echo git ;;
    debian:tool-python3-venv) echo python3-venv ;;
    arch:tool-python3) echo python ;;
    debian:tool-python3) echo "python3 python3-venv" ;;
    fedora:tool-python3) echo python3 ;;
    arch:tool-gzip | debian:tool-gzip | fedora:tool-gzip) echo gzip ;;
    arch:tool-curl | debian:tool-curl | fedora:tool-curl) echo curl ;;
    arch:tool-sha256sum | debian:tool-sha256sum | fedora:tool-sha256sum) echo coreutils ;;
    arch:tool-make) echo "make gcc" ;;
    debian:tool-make) echo build-essential ;;
    fedora:tool-make) echo "make gcc-c++" ;;
  esac
}

# ensure_packages FAMILY GROUP...: each GROUP is a string of package names. Show the exact commands, ask once, run them.
# The installer asks once per run: a second call (a package that only a later check finds) shows its command and stops.
# Exits 3 when the user does not run the commands.
SUDO_ASKED=0

ensure_packages() {
  local family=$1
  shift
  local groups=() g all="" update=() verb=() sudo_cmd=()
  for g in "$@"; do
    [ -n "$g" ] || continue
    groups+=("$g")
    all="$all $g"
  done
  [ ${#groups[@]} -gt 0 ] || return 0
  all=${all# }
  case "$family" in
    arch) verb=(pacman -S --needed --noconfirm) ;;
    debian)
      update=(apt-get update)
      verb=(apt-get install -y)
      ;;
    fedora) verb=(dnf install -y) ;;
    *) die 3 "This system is not known. Install these packages with its package manager, then run this again: $all" ;;
  esac
  if command -v sudo >/dev/null 2>&1; then
    sudo_cmd=(sudo)
  else
    local plain=""
    [ ${#update[@]} -gt 0 ] && plain="${update[*]} && "
    for g in "${groups[@]}"; do plain="$plain${verb[*]} $g && "; done
    die 3 "sudo is not installed. As root, run: ${plain% && }"
  fi
  say ""
  say "These packages are missing: $all"
  say "The command to install them:"
  [ ${#update[@]} -gt 0 ] && say "  $(printf '%q ' "${sudo_cmd[@]}" "${update[@]}")"
  for g in "${groups[@]}"; do
    # shellcheck disable=SC2086
    say "  $(printf '%q ' "${sudo_cmd[@]}" "${verb[@]}" $g)"
  done
  if ! have_tty; then
    die 3 "No terminal is attached, so the installer does not run sudo. Run the command above yourself, then run the installer again."
  fi
  if [ "$SUDO_ASKED" = 1 ]; then
    die 3 "The installer asks for sudo once per run, and it asked already. Run the command above yourself, then run the installer again."
  fi
  SUDO_ASKED=1
  if ! ask_yn "Run it now? sudo may ask for your password."; then
    die 3 "Nothing was installed. Run the command above yourself, then run the installer again."
  fi
  if [ ${#update[@]} -gt 0 ]; then
    "${sudo_cmd[@]}" "${update[@]}" </dev/null || die 3 "The command failed: ${update[*]}"
  fi
  for g in "${groups[@]}"; do
    # shellcheck disable=SC2086
    "${sudo_cmd[@]}" "${verb[@]}" $g </dev/null || die 3 "The command failed: ${verb[*]} $g"
  done
}

# PENDING_PKGS: packages of missing tools. They are installed with the next package question (before the unpack, when the
# download can go on without them), so that a missing tool and a missing library give one question.
PENDING_PKGS=()

# venv_works: the python that setup will use can make a venv. A stock Ubuntu has python3 but not python3-venv.
venv_works() {
  local py tmp rc
  py=$(pick_python)
  tmp=$(mktemp -d "${TMPDIR:-/tmp}/crucible-venv.XXXXXX") || return 0
  "${py:-python3}" -m venv "$tmp/venv" >/dev/null 2>&1
  rc=$?
  rm -rf -- "$tmp"
  return $rc
}

# check_tools [venv]: with "venv", a python3 that cannot make a venv counts as a missing tool (python3-venv).
# Only Debian and Ubuntu have a package for it; elsewhere venv comes with python, so setup reports the problem.
check_tools() {
  local family missing=() pkgs=() tool p now=0
  family=$(os_family)
  for tool in git curl python3 gzip sha256sum; do
    command -v "$tool" >/dev/null 2>&1 || missing+=("$tool")
  done
  if [ "${1:-}" = venv ] && command -v python3 >/dev/null 2>&1 && [ -n "$(pkg_for "$family" tool-python3-venv)" ] && ! venv_works; then
    missing+=(python3-venv)
  fi
  [ ${#missing[@]} -eq 0 ] && return 0
  for tool in "${missing[@]}"; do
    p=$(pkg_for "$family" "tool-$tool")
    [ -n "$p" ] || die 3 "The tool $tool is missing, and its package is not known for this system. Install it, then run this again."
    pkgs+=("$p")
    # The download needs curl and the check needs sha256sum: those two cannot wait for the one question.
    case "$tool" in curl | sha256sum) now=1 ;; esac
  done
  if [ "$now" = 1 ]; then
    ensure_packages "$family" "${pkgs[*]}"
  else
    PENDING_PKGS=("${pkgs[@]}")
  fi
}

# settle_packages: install the packages of missing tools that are still pending (one question).
settle_packages() {
  [ ${#PENDING_PKGS[@]} -gt 0 ] || return 0
  ensure_packages "$(os_family)" "${PENDING_PKGS[*]}"
  PENDING_PKGS=()
}

python_note() {
  local v
  v=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null) || return 0
  if ! ver_ge "$v" "3.12"; then
    warn "Note: python3 is $v. ./crucible needs Python 3.12 or newer. If python3.12 is installed, the installer uses it."
  fi
}

ver_ge() {
  [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n 1)" = "$2" ]
}

pick_python() {
  local c
  if python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)' 2>/dev/null; then
    return 0
  fi
  for c in python3.14 python3.13 python3.12; do
    if command -v "$c" >/dev/null 2>&1; then
      printf '%s' "$c"
      return 0
    fi
  done
}

# ---- the options of ./crucible setup ------------------------------------------------------------------------------

setup_opt_kind() {
  awk -v o="$1" '$1 == o { print $2; exit }' <<<"$SETUP_OPTIONS"
}

# check_value NAME KIND VALUE: stops (exit 2) when VALUE is not a value of option NAME.
check_value() {
  local name=$1 kind=$2 val=$3 list lo hi
  case "$kind" in
    choice:*)
      list=${kind#choice:}
      case ",$list," in
        *",$val,"*) return 0 ;;
      esac
      die 2 "$name must be one of ${list//,/, }, not '$val'. Nothing was downloaded."
      ;;
    int | int:*)
      [[ $val =~ ^[0-9]+$ ]] || die 2 "$name needs a number, not '$val'. Nothing was downloaded."
      if [ "$kind" != int ]; then
        lo=${kind#int:}
        hi=${lo#*-}
        lo=${lo%-*}
        { [ "$val" -ge "$lo" ] && [ "$val" -le "$hi" ]; } || die 2 "$name must be from $lo to $hi, not $val. Nothing was downloaded."
      fi
      ;;
    value) [ -n "$val" ] || die 2 "$name needs a value. Nothing was downloaded." ;;
  esac
  return 0
}

# check_setup_args: every option that goes to ./crucible setup is one of its options with a good value. It runs before any
# download, so that a bad option does not stop the install after the downloads. Use the full option names.
check_setup_args() {
  local i=0 n=${#PASS[@]} a name val kind has
  while [ "$i" -lt "$n" ]; do
    a=${PASS[$i]}
    has=0
    case "$a" in
      --*=*)
        name=${a%%=*}
        val=${a#*=}
        has=1
        ;;
      -*) name=$a ;;
      *) die 2 "'$a' is not an option of ./crucible setup. Nothing was downloaded. See: install.sh --help" ;;
    esac
    kind=$(setup_opt_kind "$name")
    [ -n "$kind" ] || die 2 "$name is not an option of ./crucible setup (use the full option name). Nothing was downloaded. See: install.sh --help"
    if [ "$kind" = flag ]; then
      [ "$has" = 0 ] || die 2 "$name takes no value. Nothing was downloaded."
    else
      if [ "$has" = 0 ]; then
        i=$((i + 1))
        [ "$i" -lt "$n" ] || die 2 "$name needs a value. Nothing was downloaded."
        val=${PASS[$i]}
        case "$val" in -*) die 2 "$name needs a value, not '$val'. Nothing was downloaded." ;; esac
      fi
      check_value "$name" "$kind" "$val"
    fi
    i=$((i + 1))
  done
}

# setup_arg NAME: the last value of a setup option in PASS (empty when it is not given). setup_flag NAME: 0 when it is given.
setup_arg() {
  local i=0 n=${#PASS[@]} a out=""
  while [ "$i" -lt "$n" ]; do
    a=${PASS[$i]}
    case "$a" in
      "$1="*) out=${a#*=} ;;
      "$1") [ $((i + 1)) -lt "$n" ] && out=${PASS[$((i + 1))]} ;;
    esac
    i=$((i + 1))
  done
  printf '%s' "$out"
}

setup_flag() {
  local a
  for a in "${PASS[@]}"; do [ "$a" = "$1" ] && return 0; done
  return 1
}

# ---- the read-only machine checks ---------------------------------------------------------------------------------

# free_kib PATH: free KiB on the file system of PATH (the nearest folder that exists).
free_kib() {
  local p=$1
  while [ ! -e "$p" ] && [ "$p" != / ]; do p=$(dirname "$p"); done
  df -Pk "$p" 2>/dev/null | awk 'NR == 2 { print $4 }'
}

# gpu_names: the gfx names of the GPUs that the kernel reports (kfd topology), else of rocminfo.
gpu_names() {
  local f v names=""
  for f in "$SYS_DIR"/class/kfd/kfd/topology/nodes/*/properties; do
    [ -r "$f" ] || continue
    v=$(awk '$1 == "gfx_target_version" { print $2; exit }' "$f")
    case "$v" in '' | 0) continue ;; esac
    [[ $v =~ ^[0-9]+$ ]] || continue
    names="$names gfx$((v / 10000))$(printf '%x' $(((v / 100) % 100)))$(printf '%x' $((v % 100)))"
  done
  if [ -z "$names" ] && [ -x "$ROCM_DIR/bin/rocminfo" ]; then
    names=$("$ROCM_DIR/bin/rocminfo" 2>/dev/null | grep -oE '\bgfx[0-9a-f]+\b' | sort -u | tr '\n' ' ')
  fi
  printf '%s' "$names" | tr ' ' '\n' | sed '/^$/d' | sort -u | tr '\n' ' ' | sed 's/ $//'
}

check_line() {
  say "  $(printf '%-4s' "$1")  $2"
}

# machine_check: the read-only checks (no download, no root, no change). Sets CANNOT (the reasons that this machine cannot
# run Crucible) and FIXES (the problems that have a fix). The kernel line, the runtime and a full disk have a fix.
# The UMA frame buffer advice of the 96 GB memory line (machine.uma_advice in tools/crucible_cli/machine.py): only when
# the GPU has more than 1 GiB of VRAM. A 128 GB machine with a large frame buffer has a MemTotal of the 96 GB class.
uma_advice() {
  local f v best=0
  for f in "$SYS_DIR"/class/drm/card*/device/mem_info_vram_total; do
    [ -r "$f" ] || continue
    v=$(tr -dc '0-9' <"$f")
    if [ -n "$v" ] && [ "$v" -gt "$best" ]; then best=$v; fi
  done
  [ "$best" -gt 1073741824 ] || return 0
  printf ' The BIOS gives the GPU %s GiB of dedicated memory (UMA frame buffer). Set it to the smallest value (512 MB): the GPU uses the system memory through the GTT. On a 128 GB machine, that gives the 128 GB defaults.' "$(( (best + 536870912) / 1073741824 ))"
}

machine_check() {
  CANNOT=()
  FIXES=()
  local names mem gtt f best=0 pack need models free state gib want
  say "Machine checks (read only):"
  names=$(gpu_names)
  case " $names " in
    *" $GPU_ARCH_NAME "*) check_line PASS "GPU: $GPU_ARCH_NAME (Radeon 8060S)" ;;
    "  ")
      check_line FAIL "GPU: no AMD GPU found. Crucible runs on $GPU_ARCH_NAME only (Ryzen AI Max+ 395, Radeon 8060S). On such a machine, check that the amdgpu driver is loaded: lsmod | grep amdgpu"
      CANNOT+=("no $GPU_ARCH_NAME GPU")
      ;;
    *)
      check_line FAIL "GPU: found $names; Crucible runs on $GPU_ARCH_NAME only (Ryzen AI Max+ 395, Radeon 8060S)"
      CANNOT+=("the GPU is $names, not $GPU_ARCH_NAME")
      ;;
  esac
  mem=$(awk '$1 == "MemTotal:" { print $2; exit }' "$PROC_DIR/meminfo" 2>/dev/null)
  local mid=0 gtt_min=$GTT_MIN_BYTES
  pack=$(setup_arg --pack)
  if [ -n "$mem" ] && [ "$mem" -ge "$MEM_MIN_KB" ]; then
    check_line PASS "memory: $((mem / 1048576)) GiB"
  elif [ -n "$mem" ] && [ "$mem" -ge "$MEM_MID_KB" ]; then
    mid=1
    gtt_min=$(( (mem / 1048576 - 4) * 1073741824 ))
    if [ "$pack" = 4.05 ] || [ "$pack" = 4.18 ]; then
      check_line FAIL "memory: $((mem / 1048576)) GiB; the $pack pack needs a 128 GB machine. Use --pack 3.05"
      CANNOT+=("the $pack pack does not fit $((mem / 1048576)) GiB")
    else
      check_line PASS "memory: $((mem / 1048576)) GiB (a 96 GB machine: the 3.05 pack with 2 sessions of 262,144 tokens; if the free memory at start is less than this class expects, the engine starts a smaller conversation cache or fewer sessions, and as the last step a shorter --ctx).$(uma_advice)"
    fi
  else
    local memtext=unknown
    [ -n "$mem" ] && memtext="$((mem / 1048576)) GiB"
    check_line FAIL "memory: $memtext; Crucible needs a 96 GB or 128 GB machine (88 GiB or more)"
    CANNOT+=("the memory is $memtext, not 88 GiB or more")
  fi
  for f in "$SYS_DIR"/class/drm/card*/device/mem_info_gtt_total; do
    [ -r "$f" ] || continue
    gtt=$(tr -dc '0-9' <"$f")
    [ -n "$gtt" ] && [ "$gtt" -gt "$best" ] && best=$gtt
  done
  if [ "$best" -ge "$gtt_min" ]; then
    check_line PASS "GPU memory limit: $((best / 1000000000)) GB"
  elif [ "$best" -gt 0 ]; then
    local gtt_need="about 133 GB"
    [ "$mid" = 1 ] && gtt_need="$(( (mem / 1048576 - 1) * 1024 )) MiB on this 96 GB machine"
    check_line FAIL "GPU memory limit: $((best / 1000000000)) GB; it needs $gtt_need. Fix: two kernel parameters (./crucible setup shows the exact change, and applies it only after you type yes)"
    FIXES+=("the kernel parameters")
  else
    check_line WARN "GPU memory limit: not found in $SYS_DIR/class/drm (the amdgpu driver is not loaded?)"
  fi
  if ls "$ROCM_DIR"/lib/libamdhip64.so* >/dev/null 2>&1; then
    check_line PASS "HIP runtime: found under $ROCM_DIR"
  else
    check_line FAIL "HIP runtime: none under $ROCM_DIR. Fix: install the current ROCm (10.1 tested), or let ./crucible setup download a private runtime (--private-runtime yes)"
    FIXES+=("the HIP runtime")
  fi
  if [ -z "$pack" ]; then
    pack=4.05
    [ "$mid" = 1 ] && pack=3.05
  fi
  need=$PACK_DISK_GB_405
  [ "$pack" = 3.05 ] && need=$PACK_DISK_GB_305
  [ "$pack" = 4.18 ] && need=$PACK_DISK_GB_418
  models=$(setup_arg --models-dir)
  [ -n "$models" ] || models=$DIR/models
  case "$models" in \~/*) models=$HOME/${models#\~/} ;; esac
  free=$(free_kib "$models")
  if [ -z "$free" ]; then
    check_line WARN "disk for the pack: the free space of $models is not known"
  elif [ "$((free * 1024 / 1000000000))" -ge "$need" ]; then
    check_line PASS "disk for the pack: $((free * 1024 / 1000000000)) GB free for the $pack pack (about $need GB)"
  else
    check_line FAIL "disk for the pack: $((free * 1024 / 1000000000)) GB free in $models; the $pack pack needs about $need GB. Fix: free the space, or use --models-dir DIR"
    FIXES+=("the disk space for the pack")
  fi
  if setup_flag --no-state-dir; then
    check_line SKIP "disk for the conversation cache: in memory only (--no-state-dir)"
  else
    state=$(setup_arg --state-dir)
    [ -n "$state" ] || state=$HOME/.cache/crucible/states
    case "$state" in \~/*) state=$HOME/${state#\~/} ;; esac
    gib=$(setup_arg --state-disk-gib)
    [ -n "$gib" ] || gib=200
    free=$(free_kib "$state")
    want=$((gib * 1048576))
    if [ -n "$free" ] && [ "$free" -ge "$want" ]; then
      check_line PASS "disk for the conversation cache: $((free / 1048576)) GiB free, limit $gib GiB"
    elif [ -n "$free" ]; then
      check_line WARN "disk for the conversation cache: $((free / 1048576)) GiB free, limit $gib GiB. Lower the limit (--state-disk-gib N) or use --no-state-dir"
    fi
  fi
  if [ ${#CANNOT[@]} -gt 0 ]; then
    say "This machine cannot run Crucible: $(IFS=';'; printf '%s' "${CANNOT[*]}" | sed 's/;/; /g')."
  elif [ ${#FIXES[@]} -gt 0 ]; then
    say "This machine can run Crucible after the fixes above ($(IFS=';'; printf '%s' "${FIXES[*]}" | sed 's/;/; /g'))."
  else
    say "This machine can run Crucible."
  fi
}

# ---- settings -----------------------------------------------------------------------------------------------------

init_settings() {
  TAG=${TAG_ARG:-${CRUCIBLE_TAG:-$PIN_TAG}}
  [[ $TAG =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]] || die 2 "The release tag must look like v0.34.0, not '$TAG'."
  REPO_URL=${CRUCIBLE_REPO_URL:-$REPO_URL_DEFAULT}
  RELEASE_BASE=${CRUCIBLE_RELEASE_BASE:-$RELEASE_BASE_DEFAULT/$TAG}
  case "$RELEASE_BASE" in
    https://* | file://*) ;;
    *) die 2 "CRUCIBLE_RELEASE_BASE must start with https:// or file://." ;;
  esac
  # The built-in pin is for PIN_TAG only; CRUCIBLE_SUMS_SHA256 (a mirror, a test) pins any tag.
  SUMS_PIN=${CRUCIBLE_SUMS_SHA256:-}
  if [ -z "$SUMS_PIN" ] && [ "$TAG" = "$PIN_TAG" ]; then
    SUMS_PIN=$PIN_SUMS_SHA256
  fi
  STEM="crucible-engine-${TAG}-${GPU_ARCH_NAME}-rocm${ROCM_SERIES}-linux-x86_64"
  ASSET_GZ="$STEM.gz"
  ASSET_NEEDS="$STEM.needs"
  ROCM_DIR=${ROCM_PATH:-/opt/rocm}
  PROC_DIR=${CRUCIBLE_PROC:-/proc}
  SYS_DIR=${CRUCIBLE_SYS:-/sys}
  STATE_BASE=${XDG_STATE_HOME:-$HOME/.local/state}
  LOG_DIR="$STATE_BASE/crucible"
  LOG_PATH="$LOG_DIR/install.log"
  if [ -n "$DIR_ARG" ]; then
    DIR=$DIR_ARG
  elif [ -n "${CRUCIBLE_DIR:-}" ]; then
    DIR=$CRUCIBLE_DIR
  else
    DIR=$HOME/crucible
    # status, update and uninstall from the copy of this script inside an install
    local here="" self=${BASH_SOURCE[0]:-}
    here=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd -P) || here=""
    if [ -n "$here" ] && [ -f "$here/.git/crucible-install" ] && [ "$CMD" != install ]; then
      DIR=$here
    fi
    # install from the install.sh of a clone (the no-pipe way): use that clone, never clone again. A pipe has no script file.
    if [ "$CMD" = install ] && [ -n "$self" ] && [ -f "$self" ] && [ -n "$here" ] && is_clone "$here"; then
      DIR=$here
      IN_CLONE=1
    fi
  fi
  case "$DIR" in
    /*) ;;
    *) DIR="$PWD/$DIR" ;;
  esac
  DIR=${DIR%/}
  case "$DIR" in
    "" | "$HOME" | /usr | /etc | /var | /opt | /bin | /lib | /boot | /root | /home | /tmp | /srv | /sys | /proc | /dev)
      die 2 "The install folder '$DIR' is a system or home folder. Use a new folder, for example --dir \$HOME/crucible."
      ;;
  esac
}

# is_clone DIR: 0 when DIR is a git checkout of Crucible that holds this script (a clone, or a worktree of one).
is_clone() {
  [ -e "$1/.git" ] && [ -f "$1/crucible" ] && [ -f "$1/VERSION" ] && [ -f "$1/install.sh" ]
}

start_log() {
  mkdir -p "$LOG_DIR" 2>/dev/null && LOG=$LOG_PATH
  [ -n "${LOG:-}" ] && log_line "---- $(date -u +%Y-%m-%dT%H:%M:%SZ) install.sh $CMD tag=$TAG dir=$DIR"
  return 0
}

# ---- the install folder -------------------------------------------------------------------------------------------

# dir_state: fresh | same | other | foreign
dir_state() {
  if [ ! -e "$DIR" ]; then
    echo fresh
    return
  fi
  if [ -d "$DIR" ] && [ -z "$(ls -A "$DIR" 2>/dev/null)" ]; then
    echo fresh
    return
  fi
  if [ -d "$DIR/.git" ] && [ -f "$DIR/crucible" ] && [ -f "$DIR/VERSION" ]; then
    local head tagc
    head=$(git -C "$DIR" rev-parse HEAD 2>/dev/null) || head=""
    tagc=$(git -C "$DIR" rev-parse "refs/tags/$TAG^{commit}" 2>/dev/null) || tagc=""
    if [ -n "$head" ] && [ "$head" = "$tagc" ]; then
      echo same
    else
      echo other
    fi
    return
  fi
  echo foreign
}

json_get() {
  sed -n "s/.*\"$2\": *\"\\([^\"]*\\)\".*/\\1/p" "$1" 2>/dev/null | head -n 1
}

file_sha256() {
  sha256sum "$1" | cut -d ' ' -f 1
}

# 0 when DIR holds the release engine of TAG as installed by this script
engine_record_ok() {
  local rec="$DIR/build/engine/release.json" eng="$DIR/build/engine/crucible"
  [ -f "$rec" ] && [ -f "$eng" ] && [ ! -L "$eng" ] || return 1
  [ "$(json_get "$rec" tag)" = "$TAG" ] || return 1
  [ "$(json_get "$rec" engine_sha256)" = "$(file_sha256 "$eng")" ]
}

# ---- download and checks ------------------------------------------------------------------------------------------

fetch() {
  local name=$1 url="$RELEASE_BASE/$1" opts=(-fsSL)
  [ -t 2 ] && opts=(-fL --progress-bar)
  # https or a local file at the start; a redirect may go to https only (never to http, never to a file).
  opts+=(--proto '=https,file' --proto-redir '=https')
  log_line "download $url"
  if ! curl "${opts[@]}" --retry 3 --connect-timeout 20 -o "$WORK/$name.part" "$url" </dev/null 2>>"${LOG:-/dev/null}"; then
    rm -f -- "$WORK/$name.part"
    die 4 "The download of $url failed. Nothing was installed. Check the network and run this again. To build the engine from source instead, run again with --source-build."
  fi
  mv -f -- "$WORK/$name.part" "$WORK/$name"
}

sums_hash() {
  awk -v n="$1" '{ f = $2; sub(/^\*/, "", f); if (f == n) print $1 }' "$WORK/SHA256SUMS"
}

verify_download() {
  local name=$1 want have count
  count=$(sums_hash "$name" | sort -u | wc -l)
  [ "$count" -ge 1 ] || die 4 "$name is not listed in SHA256SUMS. Nothing was installed."
  [ "$count" -eq 1 ] || die 4 "SHA256SUMS lists $name with different hashes. Nothing was installed."
  want=$(sums_hash "$name" | head -n 1)
  [[ $want =~ ^[0-9a-f]{64}$ ]] || die 4 "SHA256SUMS has a bad line for $name. Nothing was installed."
  have=$(file_sha256 "$WORK/$name")
  [ "$want" = "$have" ] || die 4 "The sha256 of $name does not match SHA256SUMS (expected $want, got $have). The download is bad or partial. Nothing was installed."
  say "  sha256 ok: $name"
}

man_get() {
  awk -v k="$1" '$1 == k { print $2; exit }' "$WORK/$ASSET_NEEDS"
}

check_manifest() {
  [ "$(man_get format)" = "1" ] || die 4 "The manifest has an unknown format. Nothing was installed."
  [ "$(man_get tag)" = "$TAG" ] || die 4 "The manifest is for another release ($(man_get tag)), not $TAG. Nothing was installed."
  [ "$(man_get arch)" = "$GPU_ARCH_NAME" ] || die 4 "The manifest is for another GPU ($(man_get arch)). Nothing was installed."
  [ "$(man_get rocm)" = "$ROCM_SERIES" ] || die 4 "The manifest is for ROCm $(man_get rocm), not $ROCM_SERIES. Nothing was installed."
  MAN_COMMIT=$(man_get commit)
  MAN_SHA=$(man_get binary_sha256)
  GLIBC_MIN=$(man_get glibc_min)
  GLIBCXX_MIN=$(man_get glibcxx_min)
  [[ $MAN_COMMIT =~ ^[0-9a-f]{40}$ ]] || die 4 "The manifest has a bad commit. Nothing was installed."
  [[ $MAN_SHA =~ ^[0-9a-f]{64}$ ]] || die 4 "The manifest has a bad engine hash. Nothing was installed."
  [[ $GLIBC_MIN =~ ^([0-9]+(\.[0-9]+)*)?$ ]] || die 4 "The manifest has a bad glibc version. Nothing was installed."
  [[ $GLIBCXX_MIN =~ ^([0-9]+(\.[0-9]+)*)?$ ]] || die 4 "The manifest has a bad libstdc++ version. Nothing was installed."
  NEEDS=()
  local lib
  while read -r lib; do
    [[ $lib =~ ^[A-Za-z0-9._+-]+$ ]] || die 4 "The manifest names a library with a bad name. Nothing was installed."
    NEEDS+=("$lib")
  done < <(awk '$1 == "need" { print $2 }' "$WORK/$ASSET_NEEDS")
}

unpack_engine() {
  gzip -dc "$WORK/$ASSET_GZ" >"$WORK/engine" 2>>"${LOG:-/dev/null}" || die 4 "The engine file could not be unpacked. Nothing was installed."
  [ "$(file_sha256 "$WORK/engine")" = "$MAN_SHA" ] || die 4 "The unpacked engine does not match the sha256 in the manifest. Nothing was installed."
  chmod 700 "$WORK/engine"
}

# ---- the libraries of the target ------------------------------------------------------------------------------------

ldconfig_list() {
  local exe
  exe=$(command -v ldconfig 2>/dev/null || true)
  [ -n "$exe" ] || for exe in /sbin/ldconfig /usr/sbin/ldconfig; do [ -x "$exe" ] && break; done
  [ -x "$exe" ] && "$exe" -p 2>/dev/null
  return 0
}

# lib_present NAME: the ldconfig list is the truth for the system folders. Folders are scanned only when ldconfig gives no list.
lib_present() {
  local name=$1 d dirs
  printf '%s\n' "$LDLIST" | grep -q "[[:space:]]$(printf '%s' "$name" | sed 's/[.+]/\\&/g') " && return 0
  dirs=("$ROCM_DIR/lib" "$ROCM_DIR/lib64")
  [ -z "$LDLIST" ] && dirs+=(/usr/lib /usr/lib64 /lib64 /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu)
  for d in "${dirs[@]}"; do
    [ -e "$d/$name" ] && return 0
  done
  return 1
}

# Fills ENGINE_USABLE (1 or 0), ENGINE_WHY and MISSING (library names). A missing library is not a reason for the source
# build: ROCm needs it too. It is a package to install.
check_target() {
  local pre=${1:-}
  LDD_USED=0
  ENGINE_USABLE=1
  ENGINE_WHY=""
  MISSING=()
  LDLIST=$(ldconfig_list)
  local have lib out libcxx
  have=$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{ print $2 }')
  if [ -n "$have" ] && [ -n "$GLIBC_MIN" ] && ! ver_ge "$have" "$GLIBC_MIN"; then
    ENGINE_USABLE=0
    ENGINE_WHY="This system has glibc $have. The release engine needs glibc $GLIBC_MIN or newer."
    return 0
  fi
  libcxx=$(printf '%s\n' "$LDLIST" | awk '$1 == "libstdc++.so.6" { print $NF; exit }')
  if [ -n "$libcxx" ] && [ -f "$libcxx" ] && [ -n "$GLIBCXX_MIN" ]; then
    local pat
    pat="GLIBCXX_$(printf '%s' "$GLIBCXX_MIN" | sed 's/\./\\./g')"
    if ! LC_ALL=C grep -aqE "${pat}([^0-9]|\$)" "$libcxx"; then
      ENGINE_USABLE=0
      ENGINE_WHY="The libstdc++ of this system is older than GLIBCXX_$GLIBCXX_MIN, which the release engine needs."
      return 0
    fi
  fi
  [ -n "$LDLIST" ] && [ -z "$libcxx" ] && MISSING+=("libstdc++.so.6")
  # With ROCm on this machine, ldd shows the real list of the engine, and ROCm of this machine decides what it needs. It runs
  # after the sha256 check, never before. Then ldd is the check, and the manifest list is not used: the ROCm of the build
  # host may need libraries that the ROCm of this machine does not need. Before the unpack ("pre") there is no engine file
  # yet, and without ROCm ldd cannot show the libraries of ROCm: then the manifest list (the libraries that the engine itself
  # names) is the check.
  out=""
  if [ "$pre" != pre ] && [ -d "$ROCM_DIR/lib" ]; then
    out=$(LD_LIBRARY_PATH="$ROCM_DIR/lib:$ROCM_DIR/lib64" ldd "$WORK/engine" 2>/dev/null) || out=""
    grep -q '=>' <<<"$out" && LDD_USED=1
  fi
  if [ "$LDD_USED" = 1 ]; then
    while read -r lib; do
      case "$lib" in
        libamdhip* | libhsa* | libamd_comgr* | libhiprtc* | libroc*) ;;
        *) printf '%s\n' "${MISSING[@]:-}" | grep -qxF "$lib" || MISSING+=("$lib") ;;
      esac
    done < <(printf '%s\n' "$out" | awk '/=> not found/ { print $1 }')
  else
    for lib in "${NEEDS[@]}"; do
      lib_present "$lib" || MISSING+=("$lib")
    done
  fi
  return 0
}

# hip_runtime_note: one line about the HIP runtime of this machine. It never stops the install and never asks for an older
# version. The latest ROCm (major 10 or later) is current, 7.2.4 is tested, any other version is named with the tested ones.
# The version comes from $ROCM_DIR/.info/version, else from the file name of libamdhip64 (7.16.26385 is 10.1, 7.2.53211 is 7.2.4).
hip_runtime_note() {
  local ver="" f name major
  if [ -r "$ROCM_DIR/.info/version" ]; then
    ver=$(head -n 1 "$ROCM_DIR/.info/version" 2>/dev/null | tr -d '[:space:]')
  fi
  if [ -z "$ver" ]; then
    for f in "$ROCM_DIR"/lib/libamdhip64.so.*; do
      [ -e "$f" ] || continue
      name=${f##*/}
      case "${name#libamdhip64.so.}" in
        7.16.26385) ver=10.1 ;;
        7.2.53211) ver=7.2.4 ;;
      esac
      [ -n "$ver" ] && break
    done
  fi
  if [ -z "$ver" ]; then
    if ls "$ROCM_DIR"/lib/libamdhip64.so.* >/dev/null 2>&1; then
      say "  HIP runtime: found under $ROCM_DIR, version not known. Tested: 10.1 and 7.2.4."
    else
      say "  HIP runtime: none found under $ROCM_DIR. Install the current ROCm (10.1 tested), or let ./crucible setup offer a private runtime."
    fi
    return 0
  fi
  major=${ver%%.*}
  case "$major" in
    '' | *[!0-9]*) say "  HIP runtime: ROCm $ver. Tested: 10.1 and 7.2.4."; return 0 ;;
  esac
  if [ "$major" -ge 10 ]; then
    say "  HIP runtime: ROCm $ver (current)."
  elif [ "${ver%%-*}" = "7.2.4" ]; then
    say "  HIP runtime: ROCm $ver (tested)."
  else
    say "  HIP runtime: ROCm $ver. Tested: 10.1 and 7.2.4."
  fi
}

# lib_packages: LIB_PKGS = the packages for MISSING. Stops (exit 3) when a library has no known package.
lib_packages() {
  local family lib p unknown=()
  family=$(os_family)
  LIB_PKGS=()
  for lib in "${MISSING[@]}"; do
    p=$(pkg_for "$family" "$lib")
    if [ -n "$p" ]; then LIB_PKGS+=("$p"); else unknown+=("$lib"); fi
  done
  if [ ${#unknown[@]} -gt 0 ]; then
    if [ ${#LIB_PKGS[@]} -gt 0 ]; then
      say "These libraries have a known package: ${MISSING[*]}"
    fi
    die 3 "No package is known on this system for: ${unknown[*]}. These libraries come with ROCm or your system. Install them (the ROCm install page for your release lists them), then run this again."
  fi
}

install_missing_libraries() {
  lib_packages
  ensure_packages "$(os_family)" "${LIB_PKGS[*]}"
}

# ---- steps 2 to 5: the engine --------------------------------------------------------------------------------------

# fetch_lists: SHA256SUMS and the manifest (the file that names the commit of the tag).
fetch_lists() {
  WORK=$(mktemp -d "${TMPDIR:-/tmp}/crucible-install.XXXXXX") || die 4 "Could not make a temp folder."
  step 2 "Download the release list and the manifest (release $TAG)"
  fetch SHA256SUMS
  fetch "$ASSET_NEEDS"
  if [ -z "$SUMS_PIN" ]; then
    warn "WARN no pin for SHA256SUMS of $TAG: the check finds a damaged download, not a replaced release file."
  elif [ "$(file_sha256 "$WORK/SHA256SUMS")" != "$SUMS_PIN" ]; then
    die 4 "SHA256SUMS does not match the hash that is pinned in this installer. Nothing was installed."
  fi
}

# get_pin: with --source-build, no engine is downloaded. The manifest still pins the source: get_source compares the commit
# of the tag with the commit in the manifest, so that a tag that was moved is refused.
get_pin() {
  fetch_lists
  verify_download "$ASSET_NEEDS"
  check_manifest
}

get_engine() {
  fetch_lists
  step 3 "Download the engine ($ASSET_GZ)"
  fetch "$ASSET_GZ"
  step 4 "Check the sha256 of every download"
  verify_download "$ASSET_GZ"
  verify_download "$ASSET_NEEDS"
  check_manifest
  if [ ${#PENDING_PKGS[@]} -gt 0 ]; then
    # A tool is missing (gzip, for the unpack). Ask one question for the tool and for the libraries of the manifest.
    check_target pre
    local groups=("${PENDING_PKGS[*]}")
    if [ "$ENGINE_USABLE" = 1 ] && [ ${#MISSING[@]} -gt 0 ]; then
      say "  Missing: ${MISSING[*]}"
      lib_packages
      groups+=("${LIB_PKGS[*]}")
    fi
    ensure_packages "$(os_family)" "${groups[@]}"
    PENDING_PKGS=()
  fi
  unpack_engine
  say "  engine sha256 ok after unpacking"
  step 5 "Check the libraries of the engine on this machine"
  check_target
  if [ "$ENGINE_USABLE" = 1 ] && [ ${#MISSING[@]} -gt 0 ]; then
    say "  Missing: ${MISSING[*]}"
    install_missing_libraries
    check_target
    [ ${#MISSING[@]} -eq 0 ] || die 3 "These libraries are still missing: ${MISSING[*]}. Install them, then run this again."
  fi
  if [ "$ENGINE_USABLE" = 1 ]; then
    if [ "${LDD_USED:-0}" = 1 ]; then
      say "  Libraries: ldd finds every library of the engine."
    else
      say "  Libraries: all ${#NEEDS[@]} needed libraries are present (besides ROCm and the base system)."
    fi
  else
    say "  $ENGINE_WHY"
  fi
  hip_runtime_note
}

place_engine() {
  local eng="$DIR/build/engine"
  mkdir -p "$eng" || die 4 "Could not make $eng."
  if ! { cp -- "$WORK/engine" "$eng/crucible.tmp" && chmod 755 "$eng/crucible.tmp" && mv -f -- "$eng/crucible.tmp" "$eng/crucible"; }; then
    die 4 "Could not write the engine to $eng."
  fi
  printf '{"format": 1, "tag": "%s", "commit": "%s", "engine_sha256": "%s", "arch": "%s", "rocm": "%s", "asset": "%s", "glibc_min": "%s", "glibcxx_min": "%s"}\n' \
    "$TAG" "$MAN_COMMIT" "$MAN_SHA" "$GPU_ARCH_NAME" "$ROCM_SERIES" "$ASSET_GZ" "$GLIBC_MIN" "$GLIBCXX_MIN" >"$eng/release.json.tmp" ||
    die 4 "Could not write $eng/release.json."
  mv -f -- "$eng/release.json.tmp" "$eng/release.json" || die 4 "Could not write $eng/release.json."
}

# ---- step 6: the source --------------------------------------------------------------------------------------------

get_source() {
  local state=$1 commit parent head
  if [ "$state" = same ]; then
    say "  The source is at $TAG already."
    return 0
  fi
  parent=$(dirname "$DIR")
  mkdir -p "$parent" || die 4 "Could not make $parent."
  CLONE_TMP=$(mktemp -d "$parent/.crucible-install.XXXXXX") || die 4 "Could not make a folder in $parent."
  log_line "git clone $REPO_URL $TAG"
  git -c advice.detachedHead=false clone --quiet --branch "$TAG" "$REPO_URL" "$CLONE_TMP/src" </dev/null >>"${LOG:-/dev/null}" 2>&1 ||
    die 4 "git clone of $TAG failed. Nothing was installed. Check the network and the log."
  commit=$(git -C "$CLONE_TMP/src" rev-parse "refs/tags/$TAG^{commit}" 2>/dev/null) || die 4 "The clone has no tag $TAG."
  # git clone --branch takes a branch before a tag of the same name. What is checked out must be the tag commit.
  head=$(git -C "$CLONE_TMP/src" rev-parse HEAD 2>/dev/null) || head=""
  [ "$head" = "$commit" ] || die 4 "The files that were cloned are not the commit of tag $TAG ($commit). The repository has a branch with the name of the tag. Nothing was installed."
  if [ -n "${MAN_COMMIT:-}" ] && [ "$commit" != "$MAN_COMMIT" ]; then
    die 4 "The commit of tag $TAG ($commit) is not the commit in the release manifest ($MAN_COMMIT). Nothing was installed."
  fi
  [ -f "$CLONE_TMP/src/crucible" ] || die 4 "The clone has no ./crucible command. Nothing was installed."
  [ -d "$DIR" ] && rmdir "$DIR" 2>/dev/null
  mv -- "$CLONE_TMP/src" "$DIR" || die 4 "Could not move the source to $DIR."
  rmdir "$CLONE_TMP" 2>/dev/null
  printf 'installed by install.sh, tag %s\n' "$TAG" >"$DIR/.git/crucible-install"
  say "  Source at $TAG ($commit) in $DIR"
}

# ---- hand over -------------------------------------------------------------------------------------------------------

hand_over() {
  local py=""
  cd "$DIR" || die 4 "Cannot enter $DIR."
  py=$(pick_python)
  say ""
  say "[7/$TOTAL_STEPS] Run ./crucible setup in $DIR"
  say "Install log: ${LOG:-none}"
  cleanup
  local cmd=(./crucible)
  [ -n "$py" ] && cmd=("$py" ./crucible)
  if [ ! -t 0 ] && have_tty; then
    exec "${cmd[@]}" setup "${PASS[@]}" <"$(tty_file)"
  fi
  exec "${cmd[@]}" setup "${PASS[@]}"
}

# ---- commands --------------------------------------------------------------------------------------------------------

cmd_install() {
  check_platform
  local state
  state=$(dir_state)
  # A clone that is not at the release tag: setup builds the engine from this source, and nothing is downloaded.
  local own_source=0 ref=""
  if [ "$IN_CLONE" = 1 ] && [ "$state" = other ]; then
    own_source=1
    ref=$(git -C "$DIR" describe --tags --always --dirty 2>/dev/null || echo unknown)
  fi
  case "$state" in
    foreign) die 2 "$DIR exists and is not empty, and it is not a Crucible install. Choose another folder with --dir, or move it away. Nothing was changed." ;;
    other) [ "$own_source" = 1 ] || die 2 "$DIR is a Crucible checkout at another release than $TAG. To move it to the newest release, run: $DIR/install.sh update" ;;
  esac
  check_setup_args
  if [ "$DRY_RUN" = 1 ]; then
    say "Plan (nothing is changed):"
    say "  release:        $TAG"
    say "  install folder: $DIR ($state)"
    say "  files:          $RELEASE_BASE/{SHA256SUMS,$ASSET_NEEDS,$ASSET_GZ}"
    say "  SHA256SUMS pin: $([ -n "$SUMS_PIN" ] && echo "${SUMS_PIN:0:16}..." || echo "none (a WARN line; the check finds a damaged download only)")"
    if [ "$IN_CLONE" = 1 ]; then
      say "  source:         this clone ($DIR)$([ "$own_source" = 1 ] && echo " at $ref, not at $TAG")"
    else
      say "  source:         $REPO_URL at tag $TAG"
    fi
    say "  engine:         $([ "$SOURCE_BUILD" = 1 ] || [ "$own_source" = 1 ] && echo 'built from source by ./crucible setup' || echo 'release engine, checked by sha256')"
    say "  then:           ./crucible setup ${PASS[*]:-}"
    say ""
    machine_check
    [ ${#CANNOT[@]} -eq 0 ] || return 6
    return 0
  fi
  start_log
  trap cleanup EXIT
  step 1 "Check this system"
  machine_check
  [ ${#CANNOT[@]} -eq 0 ] || die 6 "Nothing was downloaded or changed."
  check_tools venv
  python_note
  say "  Linux x86-64, install folder $DIR"
  if [ "$own_source" = 1 ]; then
    for n in 2 3 4 5 6; do step "$n" "Skipped: this clone is at $ref, not at release $TAG"; done
    say "  Setup builds the engine from this source (no download of an engine). Nothing is cloned."
    settle_packages
    hand_over
  fi
  if [ "$state" = same ] && [ "$SOURCE_BUILD" = 0 ] && engine_record_ok; then
    say "Crucible $TAG is already installed in $DIR (engine checked). Going on with setup."
    settle_packages
    hand_over
  fi
  if [ "$SOURCE_BUILD" = 1 ]; then
    get_pin
    for n in 3 4 5; do step "$n" "Skipped (--source-build): setup builds the engine from source"; done
    USE_ENGINE=0
  else
    get_engine
    USE_ENGINE=$ENGINE_USABLE
  fi
  settle_packages
  step 6 "Get the source at $TAG"
  get_source "$state"
  if [ "$USE_ENGINE" = 1 ]; then
    if [ -f "$DIR/build/engine/crucible" ] && [ ! -f "$DIR/build/engine/release.json" ]; then
      say "  An engine build is in $DIR/build/engine already. It is kept; setup uses it."
    else
      place_engine
      say "  Release engine placed in $DIR/build/engine/crucible"
    fi
  elif [ "$SOURCE_BUILD" = 0 ]; then
    say "  Setup will build the engine from source (about 16 minutes with the ROCm 7.2.4 compiler), because the release engine cannot run here."
  fi
  hand_over
}

# engine: the release engine for the tag that is checked out in DIR (used by tools/release_make.sh)
cmd_engine() {
  check_platform
  if [ ! -f "$DIR/crucible" ] || [ ! -d "$DIR/.git" ]; then die 2 "$DIR is not a Crucible checkout."; fi
  [ "$(dir_state)" = same ] || die 5 "$DIR is not at the release $TAG, so there is no release engine for it."
  start_log
  trap cleanup EXIT
  check_tools
  get_engine
  settle_packages
  [ "$ENGINE_USABLE" = 1 ] || die 5 "$ENGINE_WHY"
  place_engine
  say "Release engine $TAG placed in $DIR/build/engine/crucible"
}

is_install() {
  [ -f "$DIR/crucible" ] && [ -d "$DIR/.git" ]
}

cmd_status() {
  if ! is_install; then
    say "Crucible is not installed in $DIR. To install it, run: curl -fsSL $RAW_URL $PIPE_CHAR bash"
    return 1
  fi
  local ref port unit setup
  ref=$(git -C "$DIR" describe --tags --exact-match HEAD 2>/dev/null || git -C "$DIR" rev-parse --short HEAD 2>/dev/null)
  say "Crucible in $DIR"
  say "  Version:  $ref"
  if engine_record_ok_any; then
    say "  Engine:   release engine $(json_get "$DIR/build/engine/release.json" tag) (sha256 checked)"
  elif [ -f "$DIR/build/engine/crucible" ]; then
    say "  Engine:   source build (or an engine that changed after the install)"
  else
    say "  Engine:   none yet (setup builds it or installs it)"
  fi
  if [ -f "$DIR/.crucible/setup-state.json" ] && command -v python3 >/dev/null 2>&1; then
    setup=$(python3 -I -c 'import json,sys
d = json.load(open(sys.argv[1])).get("steps", {})
n = sum(1 for v in d.values() if v.get("status") == "done")
print("%d of 7 steps done" % n + "".join("; %s: %s" % (k, v.get("status")) for k, v in d.items() if v.get("status") != "done"))' "$DIR/.crucible/setup-state.json" 2>/dev/null) || setup=""
    [ -n "$setup" ] && say "  Setup:    $setup"
  else
    say "  Setup:    not run yet (run: $DIR/crucible setup)"
  fi
  unit="crucible-server.service"
  if command -v systemctl >/dev/null 2>&1 && [ -f "$HOME/.config/systemd/user/$unit" ]; then
    say "  Service:  $(systemctl --user is-active "$unit" 2>/dev/null || true) (unit $unit, $(systemctl --user is-enabled "$unit" 2>/dev/null || echo 'not enabled'))"
  else
    say "  Service:  none"
  fi
  port=8790
  if [ -f "$DIR/.crucible/config.json" ] && command -v python3 >/dev/null 2>&1; then
    port=$(python3 -I -c 'import json,sys; print(int(json.load(open(sys.argv[1])).get("port", 8790)))' "$DIR/.crucible/config.json" 2>/dev/null || echo 8790)
  fi
  if curl -fsS --max-time 3 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
    say "  Server:   answers on port $port"
  else
    say "  Server:   not running on port $port"
  fi
  say "  Logs:     install: $LOG_PATH; setup: $DIR/.crucible/ (server.log, build.log)"
  return 0
}

engine_record_ok_any() {
  local saved=$TAG
  TAG=$(json_get "$DIR/build/engine/release.json" tag)
  [ -n "$TAG" ] && engine_record_ok
  local rc=$?
  TAG=$saved
  return $rc
}

cmd_update() {
  is_install || { say "Crucible is not installed in $DIR."; return 1; }
  cd "$DIR" || return 1
  if [ -f tools/crucible_cli/update.py ]; then
    exec ./crucible update "${PASS[@]}"
  fi
  if [ -n "$TAG_ARG" ]; then
    die 2 "This install is too old to choose a release: it has no tools/crucible_cli/update.py, so --tag $TAG_ARG cannot be used. Nothing was changed. Run the update without --tag first (it installs the newest release). Then run it again with --tag $TAG_ARG."
  fi
  local args=() a
  for a in "${PASS[@]}"; do [ "$a" = "--rollback" ] && args+=("$a"); done
  if [ -f "$HOME/.config/systemd/user/crucible-server.service" ]; then
    export CRUCIBLE_RESTART_UNIT=${CRUCIBLE_RESTART_UNIT:-crucible-server.service}
  fi
  export CRUCIBLE_MAKE=${CRUCIBLE_MAKE:-tools/release_make.sh}
  exec bash tools/update.sh "${args[@]}"
}

# size_of PATH: a short size text
size_of() {
  du -sh -- "$1" 2>/dev/null | cut -f 1
}

# safe_remove_target PATH: 0 when PATH is a folder below HOME that is safe to remove
safe_remove_target() {
  local p home
  p=$(realpath -m -- "$1" 2>/dev/null) || return 1
  home=$(realpath -m -- "$HOME")
  case "$p" in
    "$home"/*/* | "$home"/.[!.]*) ;;
    "$home"/*) ;;
    *) return 1 ;;
  esac
  [ "$p" != "$home" ] && [ "$p" != "/" ]
}

# The pack folders that ./crucible setup makes in the models folder (a test keeps this list equal to packs.PACKS).
PACK_FOLDERS=(Qwen3.8-Flash-Next-exl3-4.05bpw_h6_ng6 Qwen3.8-Flash-Next-exl3-3.05bpw_h5_ng5 Qwen3.8-Flash-Next-exl3-4.18bpw_h6_ng8)

# top_name PATH REAL: the top-level name below REAL that holds PATH (both are resolved). Empty when PATH is not inside REAL.
top_name() {
  local rp rel
  rp=$(realpath -m -- "$1")
  case "$rp" in
    "$2"/*)
      rel=${rp#"$2"/}
      printf '%s' "${rel%%/*}"
      ;;
  esac
}

# holds_install PATH REAL: 0 when PATH is REAL or holds REAL (it could not stay when the install goes)
holds_install() {
  local rp
  rp=$(realpath -m -- "$1")
  [ "$rp" = "$2" ] && return 0
  case "$2" in "$rp"/*) return 0 ;; esac
  return 1
}

cmd_uninstall() {
  if ! is_install || [ ! -f "$DIR/.git/crucible-install" ]; then
    die 2 "$DIR is not an install made by install.sh, so uninstall leaves it alone."
  fi
  # The install has the uninstall command (the same code as ./crucible uninstall). Use it.
  if [ -f "$DIR/tools/crucible_cli/update.py" ]; then
    local py="" cmd=(./crucible) args=("${PASS[@]}")
    cd "$DIR" || die 4 "Cannot enter $DIR."
    py=$(pick_python)
    [ -n "$py" ] && cmd=("$py" ./crucible)
    [ "$REMOVE_MODELS" = 1 ] && args+=(--remove-models)
    if [ ! -t 0 ] && have_tty; then
      exec "${cmd[@]}" uninstall "${args[@]}" <"$(tty_file)"
    fi
    exec "${cmd[@]}" uninstall "${args[@]}"
  fi
  uninstall_old_install
}

# An install from before ./crucible uninstall existed. It removes the install folder, and with --remove-models only the
# pack folders that setup makes (never the whole models folder, never the state folder).
uninstall_old_install() {
  local models="" state="" cfg="$DIR/.crucible/config.json" real parent name pack p
  real=$(realpath -m -- "$DIR")
  parent=$(dirname "$real")
  name=$(basename "$real")
  safe_remove_target "$real" || die 2 "Refusing to remove $real."
  if [ -e "$parent/.$name.update.json" ]; then
    die 1 "An update was cut off in $real. Nothing was removed. Run $real/install.sh update to finish or undo it, then run this again."
  fi
  if command -v flock >/dev/null 2>&1; then
    exec 9<"$parent" || die 1 "Cannot open $parent. Nothing was removed."
    flock -n 9 || die 1 "An update is running. Nothing was removed."
  fi
  if [ -f "$cfg" ] && command -v python3 >/dev/null 2>&1; then
    models=$(python3 -I -c 'import json,sys; v = json.load(open(sys.argv[1])).get("models_dir"); print(v or "")' "$cfg" 2>/dev/null || true)
    state=$(python3 -I -c 'import json,sys; v = json.load(open(sys.argv[1])).get("state_dir"); print(v or "")' "$cfg" 2>/dev/null || true)
  fi
  [ -z "$models" ] && [ -d "$DIR/models" ] && models="$DIR/models"
  local packs=()
  if [ -n "$models" ]; then
    for name in "${PACK_FOLDERS[@]}"; do
      pack="$models/$name"
      if [ -e "$pack" ] || [ -L "$pack" ]; then packs+=("$pack"); fi
    done
  fi
  local state_there=0
  if [ -n "$state" ] && { [ -e "$state" ] || [ -L "$state" ]; }; then state_there=1; fi
  # What must stay inside the install folder: the packs (unless --remove-models), the models folder (when it is inside),
  # and the state folder. Found by resolved path, so a link or a models folder that is the install folder hides nothing.
  local keep_items=() keep_names=() item n
  if [ "$REMOVE_MODELS" = 0 ]; then
    keep_items+=("${packs[@]}")
    if [ -n "$models" ] && [ -n "$(top_name "$models" "$real")" ]; then keep_items+=("$models"); fi
  fi
  [ "$state_there" = 1 ] && keep_items+=("$state")
  for item in "${keep_items[@]}"; do
    if holds_install "$item" "$real"; then
      die 2 "$item is the install folder or holds it, so it cannot be kept. Nothing was removed. Move it out of the install folder, then run this again."
    fi
    n=$(top_name "$item" "$real")
    [ -n "$n" ] && keep_names+=("$n")
  done
  if [ "$REMOVE_MODELS" = 1 ]; then
    for pack in "${packs[@]}"; do
      if holds_install "$pack" "$real"; then
        die 2 "$pack holds the install folder, so it cannot be removed. Nothing was removed."
      fi
    done
  fi
  say "Uninstall plan:"
  say "  Remove:  $DIR ($(size_of "$DIR"))"
  [ -f "$HOME/.config/systemd/user/crucible-server.service" ] && say "  Remove:  the service crucible-server.service"
  if [ -n "$models" ]; then
    if [ "$REMOVE_MODELS" = 1 ]; then
      for pack in "${packs[@]}"; do say "  Remove:  $pack ($(size_of "$pack"))"; done
      say "  Keep:    every other file in $models"
    else
      for pack in "${packs[@]}"; do say "  Keep:    $pack ($(size_of "$pack")); add --remove-models to remove it"; done
      say "  Keep:    every other file in $models"
    fi
  fi
  [ "$state_there" = 1 ] && say "  Keep:    $state (the state folder is never removed by this command)"
  [ -e "$HOME/.config/crucible" ] && say "  Keep:    $HOME/.config/crucible (the API key)"
  say "  Keep:    the install log $LOG_PATH"
  if [ "$YES" = 0 ]; then
    have_tty || die 2 "Uninstall asks one question, and no terminal is attached. Add --yes."
    ask_yn "Remove these now?" || { say "Nothing was removed."; return 0; }
  fi
  if [ -x "$DIR/crucible" ]; then
    (cd "$DIR" && ./crucible service uninstall </dev/null) >>"${LOG:-/dev/null}" 2>&1 || warn "The service could not be removed. Look at: ./crucible service status"
    (cd "$DIR" && ./crucible stop </dev/null) >>"${LOG:-/dev/null}" 2>&1 ||
      die 1 "The server did not stop. Nothing was removed. Wait a minute, then run this again."
  fi
  if [ "$REMOVE_MODELS" = 1 ]; then
    for pack in "${packs[@]}"; do
      if safe_remove_target "$pack"; then
        rm -rf -- "$pack"
        say "Removed $pack."
      else
        warn "Not removed: $pack is outside your home folder or is too broad. Remove it yourself if you want."
      fi
    done
    # The models folder stays when other files are in it and it is inside the install folder.
    if [ -n "$models" ] && [ -n "$(top_name "$models" "$real")" ] && [ -d "$models" ] && [ -n "$(ls -A "$models" 2>/dev/null)" ]; then
      keep_names+=("$(top_name "$models" "$real")")
    fi
  fi
  local entry base skip k last=()
  for entry in "$real"/* "$real"/.[!.]* "$real"/..?*; do
    [ -e "$entry" ] || [ -L "$entry" ] || continue
    base=$(basename "$entry")
    skip=0
    for k in "${keep_names[@]}"; do [ "$k" = "$base" ] && skip=1; done
    [ "$skip" = 1 ] && continue
    case "$base" in crucible | .git) last+=("$entry") ;; *) rm -rf -- "$entry" ;; esac
  done
  for entry in "${last[@]}"; do rm -rf -- "$entry"; done
  if [ ${#keep_names[@]} -gt 0 ]; then
    say "Removed the install; kept ${keep_names[*]} inside $real."
  else
    rmdir "$real" 2>/dev/null || true
    say "Removed $real."
  fi
  return 0
}

main() {
  set -u
  LOG=""
  parse_args "$@"
  case "$CMD" in
    help)
      usage
      return 0
      ;;
  esac
  case "$CMD" in
    status) ;;
    *)
      [ "$(id -u 2>/dev/null)" != 0 ] ||
        die 2 "Do not run the installer as root (or with sudo). Run it as your own user. It asks for sudo itself only for system packages."
      ;;
  esac
  init_settings
  case "$CMD" in
    install) cmd_install ;;
    engine) cmd_engine ;;
    status)
      LOG=""
      cmd_status
      ;;
    update) cmd_update ;;
    uninstall) cmd_uninstall ;;
  esac
}

if true; then main "$@"; fi
