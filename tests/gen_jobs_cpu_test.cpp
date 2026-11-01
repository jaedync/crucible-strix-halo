// gen --jobs: job parser, shared-load checks and the state reset between jobs.
// The Python runner (test_gen_cli_cpu.py) supplies gen_cli_cpu.h, the real driver without GPU entry points.
#include "gen_cli_cpu_fixture.h"
#include "gen_cli_cpu.h"
#include "cli/gen_jobs.h"
#include <filesystem>
#include <functional>
#include <stdlib.h>

namespace fs = std::filesystem;

template <class F> std::string Thrown(F f) {
  try { f(); } catch (const std::exception& error) { return error.what(); }
  return "";
}
bool Has(const std::string& text, const std::string& part) { return text.find(part) != std::string::npos; }
std::string Read(const fs::path& path) {
  std::ifstream f(path);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

struct Dir {
  fs::path root;
  Dir() {
    std::string pattern = (fs::temp_directory_path() / "gen-jobs-XXXXXX").string();
    root = mkdtemp(pattern.data());
  }
  ~Dir() { fs::remove_all(root); }
  std::string Sub(const std::string& name) const { fs::create_directories(root / name); return (root / name).string(); }
};

std::string Quote(const std::string& text) {
  std::string out = "\"";
  for (char c : text) { if (c == '"' || c == '\\') out += '\\'; out += c; }
  return out + "\"";
}

std::string Line(const std::vector<std::string>& args, const std::string& out = "", const std::string& err = "") {
  std::string line = "{\"args\": [";
  for (size_t i = 0; i < args.size(); ++i) line += (i ? ", " : "") + Quote(args[i]);
  line += "]";
  if (!out.empty()) line += ", \"stdout\": " + Quote(out);
  if (!err.empty()) line += ", \"stderr\": " + Quote(err);
  return line + "}\n";
}

void ParserTests(const Dir& dir) {
  const std::string a = dir.Sub("a"), b = dir.Sub("b");
  const auto jobs = ParseGenJobs(Line({"--ids", "1,2,3", "--n", "4", "--chunk", "1", "--dump", a}, dir.root / "a.out",
                                      dir.root / "a.err") + "\n  \n" +
                                 Line({"--ids", "5", "--spec", "2", "--prefill", "0", "--dump", b}));
  assert(jobs.size() == 2);
  assert(jobs[0].line == 1 && jobs[1].line == 4);
  assert(jobs[0].options.ids == (std::vector<int>{1, 2, 3}) && jobs[0].options.n == 4 && jobs[0].options.chunk == 1);
  assert(jobs[0].options.dump == a && jobs[0].stdout_path == (dir.root / "a.out").string());
  // The same defaults and rules as one gen command: --chunk alone selects the small-row path.
  assert(jobs[0].options.prefill_rows == 0 && jobs[0].options.spec == kDefaultSpecDepth);
  assert(jobs[1].options.spec == 2 && jobs[1].stdout_path.empty() && jobs[1].stderr_path.empty());
  // spec_sweep and gen take --stop-ids (training/mtp_tune/heads_ab.py passes it to every sweep case).
  const auto stops = ParseGenJobs(Line({"--ids", "5", "--stop-ids", "248044,248046"}));
  assert(stops[0].options.stop_ids == (std::vector<int>{248044, 248046}));

  assert(Has(Thrown([] { ParseGenJobs(""); }), "no jobs"));
  assert(Has(Thrown([] { ParseGenJobs("\n{\"args\": [\"--ids\", \"1\"], \"other\": 1}\n"); }), "line 2"));
  assert(Has(Thrown([] { ParseGenJobs("{\"args\": [\"--ids\", \"1\"], \"other\": 1}"); }), "unknown job field"));
  assert(Has(Thrown([] { ParseGenJobs("{\"stdout\": \"x\"}"); }), "args"));
  assert(Has(Thrown([] { ParseGenJobs("{\"args\": [\"--ids\", 1]}"); }), "line 1"));
  assert(Has(Thrown([] { ParseGenJobs("{\"args\": [\"--ids\", \"1\"]} x"); }), "line 1"));
  assert(Has(Thrown([] { ParseGenJobs("{\"args\": [\"--ids\", \"1\"], \"args\": []}"); }), "duplicate"));
  assert(Has(Thrown([] { ParseGenJobs("{\"args\": [\"--ids\"]}"); }), "gen options"));
  assert(Has(Thrown([] { ParseGenJobs("{\"args\": [\"--streams\", \"2\"]}"); }), "gen options"));
  assert(Has(Thrown([] { ParseGenJobs("{\"args\": [\"--jobs\", \"f\"]}"); }), "gen options"));
  assert(Has(Thrown([] { ParseGenJobs("{\"args\": [\"--n\", \"x\"]}"); }), "gen options"));
}

void CheckTests(const Dir& dir) {
  const std::string a = dir.Sub("c1"), b = dir.Sub("c2");
  const auto issue = [](const std::string& text) {
    auto jobs = ParseGenJobs(text);
    return GenJobsIssue(jobs);
  };
  const std::vector<std::string> base{"--ids", "1,2,3", "--n", "2", "--chunk", "8", "--prefill", "0"};
  const auto with = [&](std::vector<std::string> extra) {
    std::vector<std::string> args = base;
    args.insert(args.end(), extra.begin(), extra.end());
    return args;
  };
  assert(issue(Line(with({"--dump", a})) + Line(with({"--dump", b, "--spec", "0"}))).empty());
  // The load fixes the context and the prefill rows; refuse the job and name the option.
  std::string text = issue(Line(with({"--dump", a})) + Line(with({"--dump", b, "--ctx", "40000"})));
  assert(Has(text, "job 1") && Has(text, "--ctx 40000") && Has(text, "--ctx 32768"));
  text = issue(Line(with({"--dump", a})) + Line({"--ids", "1", "--prefill", "4096", "--dump", b}));
  assert(Has(text, "job 1") && Has(text, "--prefill 4096") && Has(text, "--prefill 0"));
  text = issue(Line({"--ids", "1", "--dump", a}) + Line({"--ids", "1", "--prefill", "4096", "--dump", b}));
  assert(Has(text, "--prefill auto"));
  // --chunk without --prefill means --prefill 0, as in one gen command.
  assert(issue(Line({"--ids", "1", "--chunk", "1", "--dump", a}) + Line(with({"--dump", b}))).empty());
  // Dumps append: two jobs cannot share an output, and a dump directory starts empty.
  text = issue(Line(with({"--dump", a})) + Line(with({"--dump", a + "/"})));
  assert(Has(text, "job 1") && Has(text, "job 0") && Has(text, a));
  text = issue(Line(with({"--dump", a}), dir.root / "same.out") + Line(with({"--dump", b}), dir.root / "same.out"));
  assert(Has(text, "same.out"));
  text = issue(Line(with({"--dump", a}), dir.root / "x.log", dir.root / "x.log"));
  assert(Has(text, "x.log"));
  text = issue(Line(with({"--dump", a, "--dump-logits", b + "/l.f32"})) + Line(with({"--dump", b})));
  assert(text.empty());
  std::ofstream(fs::path(b) / "h_00.f32") << "x";
  text = issue(Line(with({"--dump", b})));
  assert(Has(text, "not empty") && Has(text, b));
  text = issue(Line(with({"--dump", (dir.root / "missing").string()})));
  assert(Has(text, "missing") && Has(text, "does not exist"));
  // The options of each job pass the same checks as one gen command.
  text = issue(Line(with({"--dump", a})) + Line({"--ids", "1", "--chunk", "99", "--prefill", "0"}));
  assert(Has(text, "job 1") && Has(text, "invalid"));
}

GenJobState Clean() { return {}; }

void StateTests() {
  assert(GenJobStateIssues(Clean()).empty());
  const std::vector<std::pair<std::function<void(GenJobState&)>, std::string>> dirty{
      {[](GenJobState& s) { s.history = 3; }, "token history"},
      {[](GenJobState& s) { s.retained_clean = false; }, "retained prefix"},
      {[](GenJobState& s) { s.sequences = 2; }, "sequence set"},
      {[](GenJobState& s) { s.active_first = false; }, "sequence set"},
      {[](GenJobState& s) { s.batch_streams = 1; }, "batch streams"},
      {[](GenJobState& s) { s.verification_spans = true; }, "verification spans"},
      {[](GenJobState& s) { s.sequence_zero = false; }, "sequence device state"},
      {[](GenJobState& s) { s.scratch_zero = false; }, "gen stack scratch"},
      {[](GenJobState& s) { s.draft_selection = true; }, "draft head selection"},
      {[](GenJobState& s) { s.draft_last_only = false; }, "draft head selection"},
      {[](GenJobState& s) { s.draft_margin_reset = false; }, "draft margin"},
      {[](GenJobState& s) { s.overlap_head = true; }, "overlap head"},
      {[](GenJobState& s) { s.profile = true; }, "profile counters"},
      {[](GenJobState& s) { s.profile_entries = 1; }, "profile counters"},
      {[](GenJobState& s) { s.profile_marks = 1; }, "profile counters"},
      {[](GenJobState& s) { s.expert_counts_zero = false; }, "expert counters"},
      {[](GenJobState& s) { s.expert_counting = true; }, "expert counters"},
  };
  for (const auto& [make, name] : dirty) {
    GenJobState s = Clean();
    make(s);
    const auto issues = GenJobStateIssues(s);
    assert(issues.size() == 1 && issues[0] == name);
  }
}

void Finish(Engine&, const GenOptions& o, const GenResult& r) { PrintGen(o, r); }

void RunTests(const Dir& dir) {
  const auto job = [&](const std::string& name, std::vector<std::string> extra = {}) {
    std::vector<std::string> args{"--ids", "1,2,3,4,5", "--n", "6", "--spec", "2", "--prefill", "0",
                                  "--draft-head", "full", "--dump", dir.Sub("run-" + name)};
    args.insert(args.end(), extra.begin(), extra.end());
    return Line(args, dir.root / (name + ".out"), dir.root / (name + ".err"));
  };
  auto jobs = ParseGenJobs(job("first") + job("serial", {"--spec", "0"}) + job("last"));
  assert(GenJobsIssue(jobs).empty());
  Engine engine(32768, 0);
  // A stale stack from an earlier request must not reach the next job.
  GreedyScratch().stacks[17] = 3.f;
  GreedyScratch().carry[5] = 2.f;
  int resets = 0;
  engine.on_reset = [&] { ++resets; };
  RunGenJobs(engine, jobs, Finish);
  assert(resets == 3);
  const auto ids = [](const std::string& out) {
    const std::string text = Read(out);
    return text.substr(0, text.find('\n'));
  };
  assert(ids(dir.root / "first.out") == ids(dir.root / "last.out"));
  assert(Has(ids(dir.root / "first.out"), "ids:") && ids(dir.root / "first.out").size() > 4);
  assert(Read(dir.root / "serial.out").find("verify forwards") == std::string::npos);

  // A field that the reset leaves dirty stops the process before the job runs.
  for (const std::string field : {"history", "sequence"}) {
    auto again = ParseGenJobs(job(field + "-1") + job(field + "-2"));
    Engine leaky(32768, 0);
    leaky.leave_dirty = field;
    const std::string error = Thrown([&] { RunGenJobs(leaky, again, Finish); });
    assert(Has(error, "job 1") && Has(error, "not clean"));
    assert(Has(error, field == "history" ? "token history" : "sequence device state"));
    assert(fs::exists(dir.root / (field + "-1.out")) && !fs::exists(dir.root / (field + "-2.out")));
  }
  // The scratch check reads the real stacks: a dirty row that the reset did not clear is caught.
  {
    auto again = ParseGenJobs(job("scratch"));
    Engine leaky(32768, 0);
    leaky.on_reset = [] { GreedyScratch().carry[1] = 1.f; };
    const std::string error = Thrown([&] { RunGenJobs(leaky, again, Finish); });
    assert(Has(error, "job 0") && Has(error, "gen stack scratch"));
  }
}

int main() {
  Dir dir;
  ParserTests(dir);
  CheckTests(dir);
  StateTests();
  RunTests(dir);
  std::puts("PASS: gen --jobs parser, shared-load checks and per-job state reset");
}
