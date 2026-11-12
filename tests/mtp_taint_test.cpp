// Leak guard (WP part 1): the taint logic of the native mtp_dump (bench/mtp_taint.h), CPU only.
//   * the canary value (computed from a seed, like server/leak_canary.py) has the pinned SHA-256;
//   * mtp_dump refuses to start when a diagnostic switch is set in the environment;
//   * a list is public only with the exact public header line, else tainted (unknown provenance is tainted);
//   * a tainted dump gets the canary in every .stk trailer and as the first line of every .ids;
//   * a public dump gets PROVENANCE.json.
#include "../bench/mtp_taint.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;
using namespace mtp_taint;

static std::string Slurp(const fs::path& p) {
  std::ifstream f(p, std::ios::binary);
  std::stringstream s;
  s << f.rdbuf();
  return s.str();
}
// A public list: the public line with the digest of the rows, then the rows.
static std::string Pub(const std::string& rows) { return mtp_taint::PublicLine(rows) + "\n" + rows; }

static void Put(const fs::path& p, const std::string& text) {
  std::ofstream f(p, std::ios::binary);
  f << text;
}

int main() {
  // Same pin as server/tests: the digest of the canary value (the value itself is not in any file).
  assert(Sha256Hex(Canary()) == "8294a84ecb9b205fa42ee9d696d20eda7f78fda87255e89506b0cf5f3299a6aa");
  assert(Sha256Hex("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  assert(Sha256Hex("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  assert(Sha256Hex(std::string(1000, 'a')) == "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3");

  // The environment: every diagnostic switch of the registry, and the four capture names, stops the start.
  for (const char* n : {"CRUCIBLE_QSA_CAPTURE_AT", "CRUCIBLE_QSA_CAPTURE_DIR", "CRUCIBLE_QSA_CAPTURE_INDEX_ONLY", "CRUCIBLE_ROUTE_DUMP"})
    assert(!DiagnosticSwitches().empty() && std::string(n).size() && [&] {
      for (const auto& d : DiagnosticSwitches()) if (d == n) return true;
      return false;
    }());
  assert(DiagnosticSwitchInEnvironment() == nullptr);
  for (const auto& name : DiagnosticSwitches()) {
    setenv(name.c_str(), "", 1);  // set but empty counts as set
    const char* hit = DiagnosticSwitchInEnvironment();
    assert(hit && name == hit);
    unsetenv(name.c_str());
  }
  for (const crucible::SwitchInfo& s : crucible::kSwitches) {
    if (s.kind != crucible::SwitchKind::kDiagnostic) continue;
    bool listed = false;
    for (const auto& d : DiagnosticSwitches()) listed |= d == s.name;
    assert(listed);
  }
  assert(DiagnosticSwitchInEnvironment() == nullptr);
  setenv("CRUCIBLE_EXACT", "1", 1);  // not a diagnostic switch
  assert(DiagnosticSwitchInEnvironment() == nullptr);
  unsetenv("CRUCIBLE_EXACT");

  char tmpl[] = "/tmp/crucible-mtp-taint-XXXXXX";
  assert(mkdtemp(tmpl));
  const fs::path root(tmpl);
  const std::string c = Canary();
  Put(root / "tainted.list", "#" + c + "\nseq 1,2,3\n");
  Put(root / "public.list", Pub("seq 1,2,3\n"));
  Put(root / "bareheader.list", "#CRUCIBLE-PUBLIC-SOURCE-v1\nseq 1,2,3\n");   // the old line, no digest
  Put(root / "bare.list", "seq 1,2,3\n");
  Put(root / "empty.list", "");
  Put(root / "crlf.list", std::string(mtp_taint::PublicLine("seq 1,2,3\r\n")) + "\r\nseq 1,2,3\r\n");
  Put(root / "late.list", "seq 1,2,3\n#CRUCIBLE-PUBLIC-SOURCE-v1\n");
  Put(root / "space.list", mtp_taint::PublicLine("seq 1,2,3\n") + " \nseq 1,2,3\n");
  Put(root / "appended.list", Pub("seq 1,2,3\n") + "other 7,8,9\n");   // H7 digest: rows added later
  Put(root / "changed.list", Pub("seq 1,2,3\n").replace(Pub("seq 1,2,3\n").size() - 4, 1, "4"));
  assert(ListProvenance((root / "tainted.list").string()) == Provenance::kTainted);
  assert(ListProvenance((root / "public.list").string()) == Provenance::kPublic);
  assert(ListProvenance((root / "bare.list").string()) == Provenance::kTainted);   // unknown provenance
  assert(ListProvenance((root / "empty.list").string()) == Provenance::kTainted);
  assert(ListProvenance((root / "missing.list").string()) == Provenance::kTainted);
  assert(ListProvenance((root / "crlf.list").string()) == Provenance::kTainted);   // not the exact line: fail closed
  assert(ListProvenance((root / "late.list").string()) == Provenance::kTainted);   // only the first line counts
  assert(ListProvenance((root / "space.list").string()) == Provenance::kTainted);
  assert(ListProvenance((root / "bareheader.list").string()) == Provenance::kTainted);
  assert(ListProvenance((root / "appended.list").string()) == Provenance::kTainted);
  assert(ListProvenance((root / "changed.list").string()) == Provenance::kTainted);

  // Outputs.
  const std::string rows = "ROWS";
  assert(WriteStk((root / "a.stk").string(), rows.data(), rows.size(), Provenance::kTainted));
  assert(Slurp(root / "a.stk") == rows + c);
  assert(WriteStk((root / "b.stk").string(), rows.data(), rows.size(), Provenance::kPublic));
  assert(Slurp(root / "b.stk") == rows);
  assert(WriteIds((root / "a.ids").string(), {4, 5, 6}, Provenance::kTainted));
  assert(Slurp(root / "a.ids") == "#" + c + "\n4,5,6\n");
  assert(WriteIds((root / "b.ids").string(), {4, 5, 6}, Provenance::kPublic));
  assert(Slurp(root / "b.ids") == "4,5,6\n");
  assert(!WriteStk((root / "no" / "dir.stk").string(), rows.data(), rows.size(), Provenance::kTainted));

  // The public marker: only for a public run, with the digest of the folder.
  fs::create_directory(root / "out");
  Put(root / "out" / "x.ids", "1,2\n");
  const std::vector<std::pair<std::string, std::vector<int>>> x{{"x", {1, 2}}};
  assert(FinishFolder((root / "out").string(), Provenance::kTainted, x));
  assert(!fs::exists(root / "out" / "PROVENANCE.json"));
  assert(FinishFolder((root / "out").string(), Provenance::kPublic, x));
  const std::string marker = Slurp(root / "out" / "PROVENANCE.json");
  assert(marker.rfind("{\"v\": 1, \"source\": \"public\", \"digest\": \"", 0) == 0);
  fs::create_directory(root / "out" / "sub");   // a sub folder is not an output of this list: no marker
  assert(!FinishFolder((root / "out").string(), Provenance::kPublic, x));
  fs::remove(root / "out" / "sub");

  // H8: a file that this list did not make stops the marker.
  Put(root / "out" / "other.stk", "old");
  fs::remove(root / "out" / "PROVENANCE.json");
  assert(!FinishFolder((root / "out").string(), Provenance::kPublic, x));
  assert(!fs::exists(root / "out" / "PROVENANCE.json"));

  // H7: a canary line or any other `#` line after the public line makes the list tainted.
  Put(root / "merged.list", Pub("seq 1,2\n#" + c + "\nb 7,8\n"));
  Put(root / "hash.list", Pub("seq 1,2\n#note\n"));
  Put(root / "inline.list", Pub("seq 1,2\nx " + c + "\n"));
  assert(ListProvenance((root / "merged.list").string()) == Provenance::kTainted);
  assert(ListProvenance((root / "hash.list").string()) == Provenance::kTainted);
  assert(ListProvenance((root / "inline.list").string()) == Provenance::kTainted);

  // H8 stale ids: an .ids file that differs from the list row stops the marker.
  fs::remove(root / "out" / "other.stk");
  Put(root / "out" / "x.ids", "9,9\n");
  assert(!FinishFolder((root / "out").string(), Provenance::kPublic, x));
  Put(root / "out" / "x.ids", "1,2\n");
  assert(FinishFolder((root / "out").string(), Provenance::kPublic, x));

  // --check output cannot carry the canary: a tainted run with --check is refused.
  assert(CheckOutputAllowed(Provenance::kPublic, true));
  assert(CheckOutputAllowed(Provenance::kTainted, false));
  assert(!CheckOutputAllowed(Provenance::kTainted, true));

  fs::remove_all(root);
  std::puts("PASS: mtp_dump taint logic");
}
