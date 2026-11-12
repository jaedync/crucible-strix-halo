// gen CKPT --jobs FILE: several gen captures after one model load and warmup.
// Each line of FILE is one job: {"args": [gen options], "stdout": PATH, "stderr": PATH}.
// "args" takes the same option pairs as one gen command. Weights, plans and warmup stay;
// every piece of per-request state returns to its fresh value before each job, and the
// process stops if a check finds a field still dirty.
#pragma once
#include "../exl3/safetensors.h"
#include "gen_job_state.h"
#include <fcntl.h>
#include <cerrno>
#include <cstring>
#include <unistd.h>
#include <filesystem>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

struct GenJob {
  size_t line = 0;
  std::vector<std::string> args;
  std::string stdout_path, stderr_path;
  GenOptions options;
  std::vector<int> blocks;
};

constexpr size_t kMaxGenJobs = 256;

inline std::vector<std::string> GenJobArgs(st::Scanner& scanner) {
  std::vector<std::string> args;
  scanner.Expect('[');
  while (!scanner.Peek(']')) { args.push_back(scanner.Str()); scanner.Separator(']'); }
  scanner.Expect(']');
  return args;
}

// Parses one job per nonblank line. Throws std::invalid_argument that names the line.
inline std::vector<GenJob> ParseGenJobs(const std::string& text) {
  std::vector<GenJob> jobs;
  std::stringstream lines(text);
  std::string line;
  for (size_t number = 1; std::getline(lines, line); ++number) {
    if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
    const std::string where = "job file line " + std::to_string(number) + ": ";
    if (jobs.size() == kMaxGenJobs) throw std::invalid_argument(where + "more than 256 jobs");
    GenJob job;
    job.line = number;
    bool have_args = false;
    try {
      st::Scanner scanner(line, "job", true);
      std::set<std::string> fields;
      scanner.Expect('{');
      while (!scanner.Peek('}')) {
        const std::string key = scanner.Str();
        if (!fields.insert(key).second) throw std::invalid_argument("duplicate job field: " + key);
        scanner.Expect(':');
        if (key == "args") { job.args = GenJobArgs(scanner); have_args = true; }
        else if (key == "stdout") job.stdout_path = scanner.Str();
        else if (key == "stderr") job.stderr_path = scanner.Str();
        else throw std::invalid_argument("unknown job field: " + key + " (use args, stdout, stderr)");
        scanner.Separator('}');
      }
      scanner.Expect('}');
      scanner.Finish();
    } catch (const std::exception& error) {
      throw std::invalid_argument(where + error.what());
    }
    if (!have_args) throw std::invalid_argument(where + "a job needs \"args\"");
    // The same parser and defaults as one gen command. --jobs and batch options are not gen options.
    std::vector<char*> argv{nullptr, nullptr, nullptr};
    for (std::string& arg : job.args) argv.push_back(arg.data());
    job.options = CliGenOptions();
    if (!ParseGenOptions(int(argv.size()), argv.data(), 3, job.options))
      throw std::invalid_argument(where + "invalid gen options (option and value pairs, as for one gen command)");
    jobs.push_back(std::move(job));
  }
  if (jobs.empty()) throw std::invalid_argument("the job file has no jobs");
  return jobs;
}

inline std::string PrefillText(int rows) { return rows == kPrefillAuto ? "auto" : std::to_string(rows); }

inline std::string NormalPath(const std::string& path) {
  std::filesystem::path normal = std::filesystem::weakly_canonical(std::filesystem::absolute(path));
  std::string text = normal.string();
  while (text.size() > 1 && text.back() == '/') text.pop_back();
  return text;
}

inline std::string DumpDirectoryIssue(const std::string& dir) {
  std::error_code error;
  if (!std::filesystem::is_directory(dir, error)) return "dump directory " + dir + " does not exist";
  if (!std::filesystem::is_empty(dir, error) || error)
    return "dump directory " + dir + " is not empty; dumps append, so each job needs an empty directory";
  return "";
}

// Validates every job before the model loads. Returns an empty string or the reason to refuse.
inline std::string GenJobsIssue(std::vector<GenJob>& jobs) {
  std::map<std::string, size_t> outputs;
  const GenOptions& first = jobs.front().options;
  for (size_t i = 0; i < jobs.size(); ++i) {
    GenJob& job = jobs[i];
    const std::string name = "job " + std::to_string(i) + " (line " + std::to_string(job.line) + ")";
    if (CheckGenOptions(job.options, job.blocks))
      return name + ": invalid options; see the message above";
    const GenOptions& o = job.options;
    // The context and the prefill rows size the buffers and select the plans of the load.
    if (o.ctx != first.ctx || o.prefill_rows != first.prefill_rows)
      return name + " needs --ctx " + std::to_string(o.ctx) + " --prefill " + PrefillText(o.prefill_rows) +
             ", but this process loads --ctx " + std::to_string(first.ctx) + " --prefill " +
             PrefillText(first.prefill_rows) + " (job 0); run it in a separate process";
    for (const std::string* path : std::initializer_list<const std::string*>{
             &o.dump, &o.state_dump, &o.logits_dump, &job.stdout_path, &job.stderr_path}) {
      if (path->empty()) continue;
      const auto [it, added] = outputs.emplace(NormalPath(*path), i);
      if (!added)
        return name + " and job " + std::to_string(it->second) + " both write " + *path +
               "; every job needs its own outputs";
    }
    if (!o.dump.empty())
      if (const std::string issue = DumpDirectoryIssue(o.dump); !issue.empty()) return name + ": " + issue;
  }
  return "";
}

// Sends stdout and stderr of one job to its own files, and restores the process streams.
class GenJobOutput {
 public:
  GenJobOutput(const std::string& out, const std::string& err) {
    Redirect(out, STDOUT_FILENO, saved_out_);
    Redirect(err, STDERR_FILENO, saved_err_);
  }
  ~GenJobOutput() {
    Restore(STDOUT_FILENO, saved_out_);
    Restore(STDERR_FILENO, saved_err_);
  }
  GenJobOutput(const GenJobOutput&) = delete;
  GenJobOutput& operator=(const GenJobOutput&) = delete;

 private:
  static void Redirect(const std::string& path, int fd, int& saved) {
    if (path.empty()) return;
    std::fflush(nullptr);
    const int file = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (file < 0) throw std::runtime_error("cannot open job output " + std::filesystem::path(path).filename().string() + ": " + std::strerror(errno));
    saved = dup(fd);
    if (saved < 0 || dup2(file, fd) < 0) {
      close(file);
      throw std::runtime_error("cannot redirect job output to " + path);
    }
    close(file);
  }
  static void Restore(int fd, int& saved) {
    if (saved < 0) return;
    std::fflush(nullptr);
    dup2(saved, fd);
    close(saved);
    saved = -1;
  }
  int saved_out_ = -1, saved_err_ = -1;
};

inline std::vector<DeviceSpan> GenScratchSpans() {
  std::vector<DeviceSpan> spans;
  for (GenScratch* s : {&GreedyScratch(), &SampledScratch()}) {
    spans.push_back({s->stacks, size_t(kMaxRows + 1) * kWide * sizeof(float)});
    spans.push_back({s->carry, size_t(kWide) * sizeof(float)});
  }
  return spans;
}

inline void ResetGenScratch() {
  for (const DeviceSpan& span : GenScratchSpans()) HIP_OK(hipMemset(const_cast<void*>(span.pointer), 0, span.bytes));
}

// Runs the checked jobs in order on one engine. finish(eng, options, result) prints what one gen command
// prints after RunGen. Throws std::runtime_error when a job cannot start with clean state.
template <class Finish>
void RunGenJobs(Engine& eng, const std::vector<GenJob>& jobs, Finish finish) {
  for (size_t i = 0; i < jobs.size(); ++i) {
    const GenJob& job = jobs[i];
    const std::string name = "job " + std::to_string(i) + " (line " + std::to_string(job.line) + ")";
    if (!job.options.dump.empty())
      if (const std::string issue = DumpDirectoryIssue(job.options.dump); !issue.empty())
        throw std::runtime_error(name + ": " + issue);
    // The engine reset runs last, so a buffer it does not own and leaves dirty still fails the check.
    // The check proves the reset, not completeness: projection and prefill scratch (H, x, logits, ...)
    // keep the previous job's values. The canary job and the proof against separate processes are the
    // completeness checks.
    ResetGenScratch();
    eng.ResetForJob();
    const auto issues = GenJobStateIssues(eng.JobState(GenScratchSpans()));
    if (!issues.empty())
      throw std::runtime_error(name + ": state is not clean after the reset: " + JoinIssues(issues));
    const double start = Now();
    {
      GenJobOutput output(job.stdout_path, job.stderr_path);
#if CRUCIBLE_DEBUG_NONFINITE & 4
      // Inject probe: a job that ends with the nonfinite error is a result, not a stop. The next job's reset
      // and state check run as usual.
      try {
        const GenResult result = RunGen(eng, job.options, job.blocks);
        finish(eng, job.options, result);
      } catch (const std::invalid_argument& error) {
        if (!std::string_view(error.what()).starts_with("nonfinite model logits")) throw;
        std::printf("probe_error %s\n", error.what());
        std::fprintf(stderr, "error: %s\n", error.what());
        eng.SetInject("");
      }
#else
      const GenResult result = RunGen(eng, job.options, job.blocks);
      finish(eng, job.options, result);
#endif
    }
    std::printf("job %zu done %.3f s\n", i, Now() - start);
    std::fflush(stdout);
  }
}
