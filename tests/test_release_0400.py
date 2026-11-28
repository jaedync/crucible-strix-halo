"""v0.40.0 release constants: VERSION, the engine version text, the installer pin, the setup option record of the pin
and the Docker build tag say 0.40.0. The setup options did not change after v0.39.0. The release engine stays a ROCm
7.2.4 build under the rocm7.2 name, and the release table keeps the twin and exact-resume words."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from crucible_cli import update  # noqa: E402
import release_engine  # noqa: E402


def test_the_version_is_0_40_0():
    assert (ROOT / "VERSION").read_text().strip() == "0.40.0"
    assert '#define ENGINE_BUILD_VERSION "0.40.0"' in (ROOT / "engine/cli/version.h").read_text()
    assert re.search(r'^PIN_TAG="v0\.40\.0"$', (ROOT / "install.sh").read_text(), re.M)


def test_the_pin_has_its_setup_option_record_equal_to_v0_39_0():
    record = (ROOT / "tests/data/setup_options/v0.40.0.txt").read_text()
    assert "--pack choice:3.05,4.05,4.18" in record.splitlines()
    assert record == (ROOT / "tests/data/setup_options/v0.39.0.txt").read_text()
    assert "--pack choice:3.05,4.05,4.18\n" in (ROOT / "install.sh").read_text()


def test_the_docker_build_and_guide_name_v0_40_0():
    assert re.search(r"^ARG CRUCIBLE_VERSION=v0\.40\.0$", (ROOT / "Dockerfile").read_text(), re.M)
    guide = (ROOT / "docs/DOCKER.md").read_text()
    assert re.findall(r"crucible:v\d+\.\d+\.\d+", guide) and set(re.findall(r"crucible:v\d+\.\d+\.\d+", guide)) == {"crucible:v0.40.0"}
    assert "| `CRUCIBLE_VERSION` | `v0.40.0` |" in guide


def test_the_release_engine_is_the_rocm_7_2_series():
    assert re.search(r'^ROCM_SERIES="7\.2"$', (ROOT / "install.sh").read_text(), re.M)
    assert release_engine.ROCM_SERIES == "7.2"
    assert update.ROCM_SERIES == "7.2"
    assert update.SERIES == ("7.2", "10.1")
    assert update.asset_stem("v0.40.0") == "crucible-engine-v0.40.0-gfx1151-rocm7.2-linux-x86_64"


def test_the_installer_header_names_the_7_2_4_build():
    text = (ROOT / "install.sh").read_text()
    assert "built with ROCm 7.2.4" in text
    assert "built with ROCm 10.1" not in text


def test_the_release_table_sets_the_twin_and_exact_resume_words():
    # tools/crucible_cli/measured.py: ./crucible start, the service, serve.sh, Docker and the standing
    # eval read this table. The Docker run line of the guide sets the same two words.
    from crucible_cli import measured
    assert measured.ENV["CRUCIBLE_EXACT_RESUME"] == "1" and measured.ENV["CRUCIBLE_TWIN"] == "1"
    assert "-e CRUCIBLE_EXACT_RESUME=1 -e CRUCIBLE_TWIN=1" in (ROOT / "docs/DOCKER.md").read_text()
