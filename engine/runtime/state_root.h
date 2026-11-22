// Where the conversation state folder (--state-dir) can be: the engine copy of the rule.
// tools/crucible_cli/state_root.py is the host copy (setup, doctor, uninstall). Both give the same code for the same
// path: tests/data/state_root_rule.tsv tests the two. docs/CONVERSATION_CACHE.md "Where the state folder can be" gives
// the reason for each part. The rule does not need HOME: a folder that only this user (or root) can change is accepted.
#pragma once
#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <system_error>
#include <vector>
#include <sys/stat.h>
#include <unistd.h>

namespace crucible {

// Three groups of 16 lower-case hex digits joined by '-' (StateDiskIdentity::directory).
inline bool StateIdentityDirectoryName(const std::string& name) {
  if (name.size() != 50) return false;
  for (size_t i = 0; i < name.size(); ++i) {
    const char c = name[i];
    if (i % 17 == 16 ? c != '-' : !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
  }
  return true;
}

struct StateRootVerdict {
  std::string code = "ok";      // "ok" or the refusal word (the table codes)
  std::string message;          // empty when ok, else one sentence for the user
  std::filesystem::path path;   // the real path: links in the parent folders resolved, the last name kept
  bool exists = false;          // the folder is there now
  bool Accepted() const { return code == "ok"; }
};

// The owners that may hold a parent folder: root and this user. In a user namespace (bwrap, a rootless container) the
// host's root shows as the overflow user, so that one counts as root there.
inline std::set<uid_t> StateRootTrustedOwners(uid_t uid) {
  std::set<uid_t> owners{0, uid};
  std::ifstream map("/proc/self/uid_map");
  std::string a, b, c;
  if (map >> a >> b >> c && !(a == "0" && b == "0" && c == "4294967295")) {
    std::ifstream overflow("/proc/sys/kernel/overflowuid");
    unsigned long value = 65534;
    overflow >> value;
    owners.insert(uid_t(value));
  }
  return owners;
}

inline bool StateRootSystem(const std::string& real) {
  static const char* const kTrees[] = {"/etc", "/usr", "/proc", "/sys", "/dev", "/boot", "/efi", "/bin", "/sbin",
                                       "/lib", "/lib32", "/lib64", "/var/lib"};
  static const char* const kTops[] = {"/", "/run", "/tmp", "/var", "/var/tmp", "/home", "/mnt", "/media", "/opt",
                                      "/srv", "/root"};
  for (const char* top : kTops)
    if (real == top) return true;
  for (const char* tree : kTrees) {
    const std::string t = tree;
    if (real == t || real.rfind(t + "/", 0) == 0) return true;
  }
  return false;
}

// The parent folder with its links resolved up to the first name that stat() cannot reach (missing, a link to a missing
// folder, or not readable); the names from there are kept as they are, so the parent check sees them. The host copy
// (state_root.py _resolve_parent) does the same steps.
inline std::filesystem::path StateRootResolveParent(const std::filesystem::path& parent) {
  std::filesystem::path prefix, rest;
  bool reached = true;
  for (const auto& part : parent) {
    struct stat info{};
    if (reached && ::stat((prefix / part).c_str(), &info) == 0) {
      prefix /= part;
    } else {
      reached = false;
      rest /= part;
    }
  }
  std::error_code error;
  const auto real = std::filesystem::canonical(prefix, error);
  if (error) return parent;  // changed between the two calls: the parent check sees the change
  return rest.empty() ? real : real / rest;
}

// The path with its parent links resolved and the last name kept (as os.path.normpath, then the parent resolved).
inline std::filesystem::path StateRootRealPath(const std::filesystem::path& root) {
  auto norm = root.lexically_normal();
  if (norm.filename().empty() && norm.has_relative_path()) norm = norm.parent_path();
  if (norm == norm.root_path()) return norm;
  return StateRootResolveParent(norm.parent_path()) / norm.filename();
}

namespace state_root_detail {

inline StateRootVerdict Refuse(const char* code, std::string message, const std::filesystem::path& path, bool exists = false) {
  StateRootVerdict verdict;
  verdict.code = code;
  verdict.message = std::move(message);
  verdict.path = path;
  verdict.exists = exists;
  return verdict;
}

inline std::string Mode(mode_t mode) {
  char text[8];
  std::snprintf(text, sizeof text, "%04o", unsigned(mode & 07777));
  return text;
}

// A link in a parent folder after the resolve: its target is not there (a disk that is not mounted), or it changed.
inline StateRootVerdict LinkProblem(const std::filesystem::path& link, const std::filesystem::path& real) {
  struct stat info{};
  if (::stat(link.c_str(), &info) == 0) return Refuse("link", link.string() + " is a link (it changed after the check).", real);
  const int why = errno;
  std::error_code error;
  const auto target = std::filesystem::read_symlink(link, error).string();
  if (why == ENOENT || why == ENOTDIR)
    return Refuse("link", link.string() + " is a link to a folder that is not there (" + target + ").", real);
  return Refuse("link", link.string() + " is a link that cannot be followed (" + target + ": " + std::strerror(why) + ").",
                real);
}

}  // namespace state_root_detail

// The verdict on one parent folder `current` of `real` (its lstat or fstat in `info`, not a link): a folder that only
// a trusted owner can change, or one with the sticky bit. state_private_dir.h checks each opened parent with it too.
inline StateRootVerdict StateRootParentFolder(const std::filesystem::path& current, const std::filesystem::path& real,
                                              const struct stat& info, const std::set<uid_t>& owners, gid_t gid) {
  using state_root_detail::Refuse;
  if (!S_ISDIR(info.st_mode)) return Refuse("not-dir", current.string() + " is not a folder.", real);
  if (!owners.count(info.st_uid))
    return Refuse("parent-owner", current.string() + " belongs to another user, who could move or replace " +
                                      real.string() + ".", real);
  const mode_t open_bits = info.st_mode & (S_IWOTH | (info.st_gid != gid ? S_IWGRP : 0));
  if (open_bits && !(info.st_mode & S_ISVTX))
    return Refuse("parent-open", "other users can write " + current.string() + " (mode " +
                                     state_root_detail::Mode(info.st_mode) + ", no sticky bit), so they could move or "
                                     "replace " + real.string() + ".", real);
  StateRootVerdict ok;
  ok.path = real;
  return ok;
}

namespace state_root_detail {

// The first parent folder that another user could change, or an ok verdict. Missing parents are new (made 0700).
inline StateRootVerdict ParentProblem(const std::filesystem::path& real, uid_t uid, gid_t gid) {
  const auto owners = StateRootTrustedOwners(uid);
  std::filesystem::path current;
  const auto parent = real.parent_path();
  for (const auto& part : parent) {
    current /= part;
    struct stat info{};
    if (::lstat(current.c_str(), &info) != 0) {
      if (errno == ENOENT) break;
      return Refuse("unreadable", current.string() + " cannot be read (" + std::strerror(errno) + ").", real);
    }
    if (S_ISLNK(info.st_mode)) return LinkProblem(current, real);
    auto folder = StateRootParentFolder(current, real, info, owners, gid);
    if (!folder.Accepted()) return folder;
  }
  StateRootVerdict ok;
  ok.path = real;
  return ok;
}

inline std::string Names(const std::vector<std::string>& names) {
  std::string out;
  for (size_t i = 0; i < names.size() && i < 3; ++i) out += (i ? ", " : "") + names[i];
  return out;
}

// The folder holds only what the engine writes there (write-ledger and the identity folders), all of this user.
inline StateRootVerdict ContentProblem(const std::filesystem::path& real, uid_t uid) {
  std::error_code error;
  std::vector<std::string> names;
  for (const auto& item : std::filesystem::directory_iterator(real, error)) names.push_back(item.path().filename());
  if (error)
    return Refuse("contents-unreadable", "the list of " + real.string() + " cannot be read (" + error.message() + ").", real,
                  true);
  std::sort(names.begin(), names.end());
  std::vector<std::string> foreign, other_owner;
  for (const auto& name : names) {
    struct stat info{};
    if (::lstat((real / name).c_str(), &info) != 0) { foreign.push_back(name); continue; }
    if (name != "write-ledger" && !(StateIdentityDirectoryName(name) && S_ISDIR(info.st_mode))) foreign.push_back(name);
    else if (info.st_uid != uid) other_owner.push_back(name);
  }
  if (!foreign.empty())
    return Refuse("foreign", real.string() + " holds files that Crucible did not make (" + Names(foreign) +
                                 (foreign.size() > 3 ? ", ..." : "") + ").", real, true);
  if (!other_owner.empty())
    return Refuse("foreign", real.string() + " holds files of another user (" + Names(other_owner) + ").", real, true);
  StateRootVerdict ok;
  ok.path = real;
  ok.exists = true;
  return ok;
}

}  // namespace state_root_detail

// The verdict on `root` as the state folder. `home` (may be null) refuses the home folder itself.
inline StateRootVerdict CheckStateRoot(const std::filesystem::path& root, const char* home, uid_t uid, gid_t gid) {
  using state_root_detail::Refuse;
  if (!root.is_absolute()) return Refuse("relative", root.string() + " is not an absolute path.", root);
  const auto real = StateRootRealPath(root);
  if (StateRootSystem(real.string())) return Refuse("system", real.string() + " is a system folder.", real);
  if (home && *home) {
    std::error_code error;
    const auto real_home = std::filesystem::weakly_canonical(std::filesystem::path(home), error);
    if (!error && real == real_home)
      return Refuse("home", real.string() + " is your home folder. Use a folder of its own, for example "
                                            "~/.cache/crucible/states.", real);
  }
  auto parents = state_root_detail::ParentProblem(real, uid, gid);
  if (!parents.Accepted()) return parents;
  struct stat info{};
  if (::lstat(real.c_str(), &info) != 0) {
    if (errno == ENOENT) return parents;
    return Refuse("unreadable", real.string() + " cannot be read (" + std::strerror(errno) + ").", real, true);
  }
  if (S_ISLNK(info.st_mode)) return Refuse("link", real.string() + " is a link.", real, true);
  if (!S_ISDIR(info.st_mode)) return Refuse("not-dir", real.string() + " is not a folder.", real, true);
  if (info.st_uid != uid) return Refuse("owner", real.string() + " belongs to another user.", real, true);
  if (info.st_mode & S_ISVTX)
    return Refuse("shared", real.string() + " is a shared folder (it has the sticky bit).", real, true);
  return state_root_detail::ContentProblem(real, uid);
}

}  // namespace crucible
