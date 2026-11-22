// OpenPrivateDirectory (engine/runtime/state_private_dir.h, review cache-folder-r4 L-A): each name opens from the fd of
// its parent with O_NOFOLLOW, each parent gets the parent rule, and only a folder of this user gets the mode change.
// A test cannot chown, so "another user" is a rule with another uid (or without this uid among the trusted owners).
#include "../engine/runtime/state_private_dir.h"
#include <cassert>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

namespace fs = std::filesystem;
using crucible::OpenPrivateDirectory;
using crucible::PrivateDirRule;

namespace {

mode_t Mode(const fs::path& path) {
  struct stat info{};
  assert(::lstat(path.c_str(), &info) == 0);
  return info.st_mode & 07777;
}

void SetMode(const fs::path& path, mode_t mode) { assert(::chmod(path.c_str(), mode) == 0); }

fs::path Base() {
  const char* tmp = std::getenv("TMPDIR");
  const fs::path base = fs::canonical(tmp && *tmp ? tmp : "/tmp") / ("state_private_dir_test." + std::to_string(::getpid()));
  fs::remove_all(base);
  fs::create_directories(base);
  SetMode(base, 0700);
  return base;
}

// A rule as the engine's (this process) but for another user: this user's folders belong to "someone else".
PrivateDirRule OtherUser() {
  auto rule = PrivateDirRule::Process();
  rule.uid = rule.uid + 1;
  return rule;
}

std::string LinkMessage(const fs::path& link, const fs::path& path) {
  return link.string() + " is a link, and the engine does not follow a link while it makes " + path.string() +
         " (another user can make one in a shared folder). Remove the link, or use a folder that only you can change.";
}

void MakesMissingNames(const fs::path& base) {
  const auto path = base / "new" / "deeper" / "states";
  auto result = OpenPrivateDirectory(path, PrivateDirRule::Process(), true);
  assert(result.Ok() && result.message.empty());
  for (const auto& folder : {base / "new", base / "new" / "deeper", path}) assert(Mode(folder) == 0700);
  // The fd is the folder: a file made through it is in `path`.
  const int fd = ::openat(result.fd.Get(), "probe", O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
  assert(fd >= 0);
  ::close(fd);
  assert(fs::exists(path / "probe"));
  // An existing folder of this user with another mode gets 0700; an existing parent keeps its mode.
  SetMode(path, 0755);
  SetMode(base / "new", 0750);
  assert(OpenPrivateDirectory(path, PrivateDirRule::Process(), true).Ok());
  assert(Mode(path) == 0700 && Mode(base / "new") == 0750);
}

// The folder of another user: refused, its mode stays, no fchmod.
void OtherUsersFolderStays(const fs::path& base) {
  const auto path = base / "theirs";
  fs::create_directories(path);
  SetMode(path, 0755);
  const auto result = OpenPrivateDirectory(path, OtherUser(), true);
  assert(!result.Ok());
  assert(result.message == path.string() + " belongs to another user (uid " + std::to_string(::geteuid()) +
                               "), so the engine does not change it. Use a folder of your own.");
  assert(Mode(path) == 0755);
}

// A parent of an owner that the rule does not trust (as a folder that another user made in /tmp): refused before the
// next name is made.
void UntrustedParentStops(const fs::path& base) {
  fs::create_directories(base / "made-by-them");
  auto rule = PrivateDirRule::Process();
  rule.owners = {0};
  const auto path = base / "made-by-them" / "states";
  const auto result = OpenPrivateDirectory(path, rule, true);
  assert(!result.Ok());
  // `base` itself belongs to this user, so the walk stops there, the first parent that the rule does not trust.
  assert(result.message.find(" belongs to another user, who could move or replace " + path.string() + ".") !=
         std::string::npos);
  assert(!fs::exists(path));
}

// A parent that other users can write without the sticky bit: refused; with the sticky bit (as /tmp): accepted.
void OpenParent(const fs::path& base) {
  const auto open = base / "open";
  fs::create_directories(open);
  SetMode(open, 0777);
  const auto path = open / "states";
  const auto refused = OpenPrivateDirectory(path, PrivateDirRule::Process(), true);
  assert(!refused.Ok() && refused.message == "other users can write " + open.string() +
                                                 " (mode 0777, no sticky bit), so they could move or replace " +
                                                 path.string() + ".");
  assert(!fs::exists(path));
  SetMode(open, 01777);
  assert(OpenPrivateDirectory(path, PrivateDirRule::Process(), true).Ok() && Mode(path) == 0700);
  SetMode(open, 0700);
}

// A link at any depth stops the walk, and the link target keeps its mode and files (the race of L-A).
void LinksStop(const fs::path& base) {
  const auto target = base / "target" / "states";
  fs::create_directories(target);
  SetMode(target, 0755);
  std::ofstream(target / "keep.txt") << "keep";
  fs::create_symlink(base / "target", base / "swapped");
  const auto through = base / "swapped" / "states";
  const auto parent = OpenPrivateDirectory(through, PrivateDirRule::Process(), true);
  assert(!parent.Ok() && parent.message == LinkMessage(base / "swapped", through));
  fs::create_symlink(target, base / "last");
  const auto last = OpenPrivateDirectory(base / "last", PrivateDirRule::Process(), true);
  assert(!last.Ok() && last.message == LinkMessage(base / "last", base / "last"));
  // A dangling link: the walk does not make its target.
  fs::create_symlink(base / "nowhere", base / "dangling");
  const auto dangling = OpenPrivateDirectory(base / "dangling" / "states", PrivateDirRule::Process(), true);
  assert(!dangling.Ok() && dangling.message == LinkMessage(base / "dangling", base / "dangling" / "states"));
  assert(!fs::exists(base / "nowhere"));
  assert(Mode(target) == 0755 && fs::exists(target / "keep.txt"));
}

// Review state-dir-race M1: a parent that this user can search but not read (as a root-owned /home at mode 0711)
// does not stop the walk. Trunk needed only the search bit of a parent (lstat, then mkdir by path), so the walk must
// not need more. A last folder of this user without the read bit (mode 0300) gets 0700, as on trunk.
void SearchOnlyParent(const fs::path& base) {
  const auto xonly = base / "xonly";
  fs::create_directories(xonly / "old");
  SetMode(xonly, 0300);
  const auto path = xonly / "new" / "states";
  const auto made = OpenPrivateDirectory(path, PrivateDirRule::Process(), true);
  assert(made.Ok() && made.message.empty());
  assert(Mode(xonly / "new") == 0700 && Mode(path) == 0700 && Mode(xonly) == 0300);
  // The fd can make a file and list the folder (it is not an O_PATH fd).
  const int fd = ::openat(made.fd.Get(), "probe", O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
  assert(fd >= 0);
  ::close(fd);
  assert(fs::exists(path / "probe"));
  const int list = ::openat(made.fd.Get(), ".", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  assert(list >= 0);
  ::close(list);
  // Without `make`: the existing folder opens through the same parent.
  assert(OpenPrivateDirectory(path, PrivateDirRule::Process(), false).Ok());
  // The last folder of this user at mode 0300 (no read bit): it gets 0700, and its fd works.
  SetMode(xonly / "old", 0300);
  const auto old = OpenPrivateDirectory(xonly / "old", PrivateDirRule::Process(), true);
  assert(old.Ok() && Mode(xonly / "old") == 0700);
  const int probe = ::openat(old.fd.Get(), "probe", O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
  assert(probe >= 0);
  ::close(probe);
  SetMode(xonly, 0700);
}

// Each folder that the walk makes gets 0700 exactly, whatever the umask cuts from the mkdirat mode: 0277 leaves 0500
// (no write bit), 0377 leaves 0400 (no search bit, so the mode change goes through /proc/self/fd).
void MadeFoldersGet0700WhateverTheUmask(const fs::path& base) {
  for (const mode_t mask : {mode_t(0277), mode_t(0377)}) {
    const auto top = base / ("masked" + std::to_string(mask));
    const mode_t old = ::umask(mask);
    const auto result = OpenPrivateDirectory(top / "states", PrivateDirRule::Process(), true);
    ::umask(old);
    assert(result.Ok() && Mode(top) == 0700 && Mode(top / "states") == 0700);
    const int fd = ::openat(result.fd.Get(), "probe", O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    assert(fd >= 0);
    ::close(fd);
  }
}

// Review state-dir-race L1: a folder that the walk made but that belongs to another user (another user made the name
// between the mkdirat and the openat; here a rule with another uid stands in for that user). The walk stops at that
// name: it changes nothing in it and makes nothing below it.
void MadeFolderOfAnotherUserStops(const fs::path& base) {
  const auto rule = OtherUser();  // its trusted owners still hold this euid, so the parent rule accepts `base`
  const auto path = base / "came" / "states";
  const auto result = OpenPrivateDirectory(path, rule, true);
  assert(!result.Ok());
  assert(result.message == (base / "came").string() +
                               " belongs to another user (it came while the engine made it). Use a folder that only "
                               "you can change.");
  assert(fs::exists(base / "came") && !fs::exists(path));
}

void NotAFolder(const fs::path& base) {
  std::ofstream(base / "file") << "x";
  const auto result = OpenPrivateDirectory(base / "file" / "states", PrivateDirRule::Process(), true);
  assert(!result.Ok() && result.message == (base / "file").string() + " is not a folder.");
}

// Without `make`, a missing name is "missing" and nothing is made.
void NoMake(const fs::path& base) {
  const auto path = base / "absent" / "states";
  const auto result = OpenPrivateDirectory(path, PrivateDirRule::Process(), false);
  assert(!result.Ok() && result.missing && result.message == (base / "absent").string() + " is not there.");
  assert(!fs::exists(base / "absent"));
}

void BadPaths(const fs::path& base) {
  const auto relative = OpenPrivateDirectory("states", PrivateDirRule::Process(), true);
  assert(!relative.Ok() && relative.message == "states is not an absolute path.");
  const auto dots = OpenPrivateDirectory(base / ".." / "states", PrivateDirRule::Process(), true);
  assert(!dots.Ok() && dots.message == (base / ".." / "states").string() + " has a '..' name.");
}

}  // namespace

int main() {
  const fs::path base = Base();
  MakesMissingNames(base);
  OtherUsersFolderStays(base);
  UntrustedParentStops(base);
  OpenParent(base);
  LinksStop(base);
  SearchOnlyParent(base);
  MadeFoldersGet0700WhateverTheUmask(base);
  MadeFolderOfAnotherUserStops(base);
  NotAFolder(base);
  NoMake(base);
  BadPaths(base);
  fs::remove_all(base);
  std::puts("state_private_dir_test passed");
  return 0;
}
