// The state folder rule, engine copy (engine/runtime/state_root.h), against the shared table
// tests/data/state_root_rule.tsv. tests/test_state_root_rule.py runs the same table against the host copy
// (tools/crucible_cli/state_root.py): the two give the same code for the same path.
#include "../engine/runtime/state_root.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <unistd.h>

using namespace crucible;
namespace fs = std::filesystem;

namespace {

const std::string kHex16 = "0123456789abcdef";
const std::string kIdent = kHex16 + "-" + kHex16 + "-" + kHex16;  // an identity folder name

struct Row { std::string name, setup, path, uid, gid, code; };

std::vector<std::string> Split(const std::string& text, char sep) {
  std::vector<std::string> out;
  std::string part;
  std::istringstream in(text);
  while (std::getline(in, part, sep)) out.push_back(part);
  return out;
}

std::vector<Row> Rows(const fs::path& table) {
  std::ifstream in(table);
  assert(in && "the table tests/data/state_root_rule.tsv");
  std::vector<Row> rows;
  for (std::string line; std::getline(in, line);) {
    if (line.empty() || line[0] == '#') continue;
    const auto f = Split(line, '\t');
    assert(f.size() == 6);
    rows.push_back({f[0], f[1], f[2], f[3], f[4], f[5]});
  }
  return rows;
}

fs::path FreshDir(const fs::path& base, const std::string& prefix) {
  std::string text = (base / (prefix + "XXXXXX")).string();
  assert(::mkdtemp(text.data()) != nullptr);
  ::chmod(text.c_str(), 0700);
  return text;
}

bool StickyRootTmp() {
  struct stat info{};
  if (::lstat("/tmp", &info) != 0) return false;
  const auto owners = StateRootTrustedOwners(::geteuid());
  return info.st_uid != ::geteuid() && owners.count(info.st_uid) && (info.st_mode & S_ISVTX);
}

struct Case {
  fs::path t, h, s;
  explicit Case(const fs::path& base) : t(FreshDir(base, "case-")), h(t / "home") { fs::create_directory(h); ::chmod(h.c_str(), 0700); }
  std::string Sub(const std::string& text) {
    std::string out;
    const auto parts = Split(text, '/');
    for (size_t i = 0; i < parts.size(); ++i) {
      std::string part = parts[i];
      if (i == 0 && part == "T") part = t.string();
      else if (i == 0 && part == "H") part = h.string();
      else if (i == 0 && part == "S") {
        if (s.empty()) s = FreshDir("/tmp", "state-root-");
        part = s.string();
      } else if (part == "ID") part = kIdent;
      out += (i ? "/" : "") + part;
    }
    if (!text.empty() && text.back() == '/') out += "/";
    return out;
  }
  void Build(const std::string& setup) {
    if (setup == "-") return;
    for (const auto& step : Split(setup, ';')) {
      const auto a = Split(step, ' ');
      if (a[0] == "dir") {
        const auto p = Sub(a[1]);
        assert(::mkdir(p.c_str(), 0700) == 0);
        assert(::chmod(p.c_str(), mode_t(std::stoul(a[2], nullptr, 8))) == 0);
      } else if (a[0] == "file") {
        std::ofstream(Sub(a[1])) << "x";
      } else if (a[0] == "link") {
        fs::create_symlink(Sub(a[2]), Sub(a[1]));
      } else {
        assert(false && "unknown setup step");
      }
    }
  }
  ~Case() {
    for (const auto& root : {t, s}) {
      if (root.empty()) continue;
      std::error_code error;
      for (auto it = fs::recursive_directory_iterator(root, error); !error && it != fs::recursive_directory_iterator(); it.increment(error))
        if (it->is_directory(error) && !it->is_symlink(error)) ::chmod(it->path().c_str(), 0700);
      fs::remove_all(root, error);
    }
  }
};

void TableRows(const fs::path& table) {
  const fs::path base = FreshDir(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp", "state_root_test.");
  // A gate that cannot run the rows fails: a skip would let the run pass with no row checked.
  const auto probe = CheckStateRoot(base / "probe", nullptr, ::geteuid(), ::getegid());
  if (!probe.Accepted()) {
    std::fprintf(stderr, "state_root_test: FAIL, the temp folder fails the rule itself, so no row can run: %s\n",
                 probe.message.c_str());
    fs::remove_all(base);
    std::exit(1);
  }
  const auto rows = Rows(table);
  assert(rows.size() > 40);
  const bool sticky_tmp = StickyRootTmp();
  int failed = 0, run = 0;
  for (const auto& row : rows) {
    if (row.path.rfind("S", 0) == 0 && !sticky_tmp) continue;
    Case c(base);
    c.Build(row.setup);
    struct stat info{};
    assert(::lstat(c.t.c_str(), &info) == 0);
    const uid_t uid = row.uid == "self" ? ::geteuid() : ::geteuid() + 1;
    const gid_t gid = row.gid == "self" ? info.st_gid : info.st_gid + 1;
    const std::string home = c.h.string();
    const auto verdict = CheckStateRoot(c.Sub(row.path), home.c_str(), uid, gid);
    ++run;
    if (verdict.code != row.code || verdict.message.empty() != verdict.Accepted()) {
      std::fprintf(stderr, "state_root_test: row %s: want %s, got %s (%s)\n", row.name.c_str(), row.code.c_str(),
                   verdict.code.c_str(), verdict.message.c_str());
      ++failed;
    }
  }
  fs::remove_all(base);
  std::printf("state_root_test: %d rows, %d failed\n", run, failed);
  assert(failed == 0 && run > 40);
}

// The messages are the host copy's sentences (tools/crucible_cli/state_root.py).
void Messages() {
  const fs::path base = FreshDir(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp", "state_root_msg.");
  fs::create_directory(base / "shared");
  ::chmod((base / "shared").c_str(), 0777);
  const auto verdict = CheckStateRoot(base / "shared" / "states", nullptr, ::geteuid(), ::getegid());
  const std::string want = "other users can write " + (base / "shared").string() + " (mode 0777, no sticky bit), so they "
                           "could move or replace " + (base / "shared" / "states").string() + ".";
  assert(verdict.message == want);
  assert(CheckStateRoot("/etc/x", nullptr, ::geteuid(), ::getegid()).message == "/etc/x is a system folder.");
  // The real path: parent links resolved, the last name kept.
  fs::create_directory(base / "fast");
  fs::create_symlink(base / "fast", base / "link");
  const auto linked = CheckStateRoot(base / "link" / "states", nullptr, ::geteuid(), ::getegid());
  assert(linked.Accepted() && linked.path == base / "fast" / "states" && !linked.exists);
  // A parent link to a folder that is not there (a second disk that is not mounted): the host copy's sentence.
  fs::create_symlink(base / "gone", base / "dangling");
  const auto dangling = CheckStateRoot(base / "dangling" / "states", nullptr, ::geteuid(), ::getegid());
  assert(dangling.message == (base / "dangling").string() + " is a link to a folder that is not there (" +
                                 (base / "gone").string() + ").");
  ::chmod((base / "shared").c_str(), 0700);
  fs::remove_all(base);
}

}  // namespace

int main(int argc, char** argv) {
  const fs::path table = argc > 1 ? fs::path(argv[1]) : fs::path(__FILE__).parent_path() / "data" / "state_root_rule.tsv";
  TableRows(table);
  Messages();
  std::printf("state_root_test: ok\n");
  return 0;
}
