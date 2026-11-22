// A private folder of this user (mode 0700), made or opened one name at a time from "/" (review cache-folder-r4 L-A).
// open(2) with O_NOFOLLOW checks only the last name. In a shared sticky folder another user can make a missing parent
// (the `x` of /tmp/x/states) a link between the rule check (state_root.h) and the mkdir. A path open then follows that
// link, and the mode change hits another folder of this user with the same last name. Here each name opens with
// openat(O_NOFOLLOW | O_DIRECTORY) from the fd of its parent, and each fd gets the parent rule (fstat), so a link at
// any depth stops the call and no fchmod reaches a folder that the walk did not check. Only the fd gets the fchmod.
// Each name opens with O_PATH, so a parent needs only its search bit, as on trunk (review state-dir-race M1).
#pragma once
#include "state_root.h"
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <set>
#include <string>
#include <utility>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace crucible {

// An owned file descriptor (-1: none).
class StateFd {
 public:
  StateFd() = default;
  explicit StateFd(int fd) : fd_(fd) {}
  StateFd(StateFd&& other) noexcept : fd_(std::exchange(other.fd_, -1)) {}
  StateFd& operator=(StateFd&& other) noexcept {
    if (this != &other) Reset(std::exchange(other.fd_, -1));
    return *this;
  }
  StateFd(const StateFd&) = delete;
  StateFd& operator=(const StateFd&) = delete;
  ~StateFd() { Reset(-1); }
  int Get() const { return fd_; }
  bool Valid() const { return fd_ >= 0; }

 private:
  void Reset(int fd) {
    if (fd_ >= 0) ::close(fd_);
    fd_ = fd;
  }
  int fd_ = -1;
};

// Who the walk trusts: `uid` owns the last folder; `owners` may own a parent; group `gid` may write a parent.
struct PrivateDirRule {
  uid_t uid;
  gid_t gid;
  std::set<uid_t> owners;
  static PrivateDirRule Process() { return {::geteuid(), ::getegid(), StateRootTrustedOwners(::geteuid())}; }
};

// The open folder, or the one sentence for the user that says why not. `missing`: a name was not there (make false).
struct PrivateDirResult {
  StateFd fd;
  std::string message;
  bool missing = false;
  bool Ok() const { return fd.Valid(); }
};

namespace private_dir_detail {

inline PrivateDirResult Fail(std::string message, bool missing = false) {
  PrivateDirResult result;
  result.message = std::move(message);
  result.missing = missing;
  return result;
}

// Why `name` in the folder `at` did not open as a folder (errno `why` of the openat).
inline PrivateDirResult OpenProblem(int at, const std::string& name, const std::filesystem::path& shown,
                                    const std::filesystem::path& path, int why) {
  struct stat info{};
  if (::fstatat(at, name.c_str(), &info, AT_SYMLINK_NOFOLLOW) == 0) {
    if (S_ISLNK(info.st_mode))
      return Fail(shown.string() + " is a link, and the engine does not follow a link while it makes " + path.string() +
                  " (another user can make one in a shared folder). Remove the link, or use a folder that only you "
                  "can change.");
    if (!S_ISDIR(info.st_mode)) return Fail(shown.string() + " is not a folder.");
  }
  return Fail(shown.string() + " cannot be opened (" + std::strerror(why) + ").");
}

// The readable fd (O_RDONLY) of the folder of this user that the O_PATH fd `at` holds, with mode 0700 when `set_mode`.
// The walk opens each name with O_PATH, which needs only the search bit of its parent (review state-dir-race M1: a
// root-owned /home at 0711 has no read bit for this user). fchmod refuses an O_PATH fd, so the mode change goes to a
// new fd of "." (that needs the read and search bits of the folder itself). When the folder has no such bit (for
// example mode 0300), chmod of /proc/self/fd/N changes the inode of `at`: that name always resolves to the inode of the
// fd, so no other folder can take its place.
inline PrivateDirResult Readable(const StateFd& at, const std::filesystem::path& shown, bool set_mode) {
  constexpr int kRead = O_RDONLY | O_DIRECTORY | O_CLOEXEC;
  StateFd own(::openat(at.Get(), ".", kRead));
  if (set_mode && own.Valid() && ::fchmod(own.Get(), 0700) != 0)
    return Fail("cannot set mode 0700 on " + shown.string() + ": " + std::strerror(errno) + ".");
  if (!own.Valid() && errno == EACCES && set_mode) {
    const std::string self = "/proc/self/fd/" + std::to_string(at.Get());
    if (::chmod(self.c_str(), 0700) != 0)
      return Fail("cannot set mode 0700 on " + shown.string() + ": " + std::strerror(errno) + ".");
    own = StateFd(::openat(at.Get(), ".", kRead));
  }
  if (!own.Valid()) return Fail(shown.string() + " cannot be opened (" + std::strerror(errno) + ").");
  PrivateDirResult result;
  result.fd = std::move(own);
  return result;
}

}  // namespace private_dir_detail

// Opens the folder `path` (absolute, no "." or ".." names) and returns its fd. With `make`, each missing name is made
// with mode 0700. Each parent must pass StateRootParentFolder. The last folder must belong to `rule.uid`; its mode
// becomes 0700 (fchmod of the fd). Another user's folder is never changed.
inline PrivateDirResult OpenPrivateDirectory(const std::filesystem::path& path, const PrivateDirRule& rule, bool make) {
  using private_dir_detail::Fail;
  if (!path.is_absolute()) return Fail(path.string() + " is not an absolute path.");
  StateFd at(::open("/", O_PATH | O_DIRECTORY | O_CLOEXEC));
  if (!at.Valid()) return Fail(std::string("/ cannot be opened (") + std::strerror(errno) + ").");
  std::filesystem::path shown = "/";
  bool made = false;
  for (auto part = std::next(path.begin()); part != path.end(); ++part) {
    const std::string name = part->string();
    if (name.empty()) continue;  // a trailing separator
    if (name == "." || name == "..") return Fail(path.string() + " has a '" + name + "' name.");
    struct stat parent{};
    if (::fstat(at.Get(), &parent) != 0)
      return Fail(shown.string() + " cannot be read (" + std::strerror(errno) + ").");
    const auto verdict = StateRootParentFolder(shown, path, parent, rule.owners, rule.gid);
    if (!verdict.Accepted()) return Fail(verdict.message);
    shown /= name;
    made = false;
    constexpr int kFlags = O_PATH | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC;  // needs only the search bit of `at`
    int fd = ::openat(at.Get(), name.c_str(), kFlags);
    if (fd < 0 && errno == ENOENT) {
      if (!make) return Fail(shown.string() + " is not there.", true);
      if (::mkdirat(at.Get(), name.c_str(), 0700) == 0) made = true;
      else if (errno != EEXIST) return Fail("cannot create " + shown.string() + ": " + std::strerror(errno) + ".");
      fd = ::openat(at.Get(), name.c_str(), kFlags);
    }
    if (fd < 0) return private_dir_detail::OpenProblem(at.Get(), name, shown, path, errno);
    at = StateFd(fd);
    // A folder that this call made gets 0700 exactly (mkdirat's mode is cut by the umask). It must be ours: another
    // user may have made the name between the mkdirat and the openat.
    if (made) {
      struct stat info{};
      if (::fstat(at.Get(), &info) != 0 || info.st_uid != rule.uid)
        return Fail(shown.string() + " belongs to another user (it came while the engine made it). Use a folder that "
                    "only you can change.");
      auto readable = private_dir_detail::Readable(at, shown, true);
      if (!readable.Ok()) return readable;
    }
  }
  struct stat info{};
  if (::fstat(at.Get(), &info) != 0) return Fail(shown.string() + " cannot be read (" + std::strerror(errno) + ").");
  if (info.st_uid != rule.uid)
    return Fail(path.string() + " belongs to another user (uid " + std::to_string(info.st_uid) +
                "), so the engine does not change it. Use a folder of your own.");
  return private_dir_detail::Readable(at, path, (info.st_mode & 07777) != 0700);
}

}  // namespace crucible
