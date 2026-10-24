#pragma once
#include "../runtime/spec_steps.h"
#include "../switches.h"
#include "../runtime/sampling_generation.h"
#include "../runtime/decode_policy.h"
#include <functional>
#include "../runtime/conversation_resume.h"
#include "../runtime/draft_head_policy.h"
#include "../runtime/draft_vocab.h"
#include "../gfx1151/runtime_memory_policy.h"
#include "../runtime/draft_accept.h"
#include "../runtime/gen_options.h"
#include "version.h"

struct GenResult {
  std::vector<int> gen;
  double prefill_s = 0, decode_s = 0;
  long forwards = 0, accepted = 0, proposed = 0, lookup_steps = 0, lookup_accepted = 0;
  SpecSteps steps;
  bool stopped = false;
  DraftHeadPolicy head_policy;
};

// CLI numbers must consume the full field, including unsigned seeds.
template <class T> T GenNumber(const std::string& value) {
  T result{};
  const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
  if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size())
    throw std::invalid_argument("invalid generation number");
  return result;
}

// Parses option pairs argv[first..argc) into o. Returns false on a bad pair.
bool ParseGenOptions(int argc, char** argv, int first, GenOptions& o) {
  if ((argc - first) % 2) return false;
  try {
  for (int i = first; i + 1 < argc; i += 2) {
    const std::string k = argv[i], v = argv[i + 1];
    if (k == "--ids") o.ids = ParseIds(v);
    else if (k == "--ids-file") {
      std::ifstream f(v);
      if (!f) return false;
      std::stringstream ss;
      ss << f.rdbuf();
      o.ids = ParseIds(ss.str());
    } else if (k == "--n") o.n = std::stoi(v);
    else if (k == "--chunk") { o.chunk = std::stoi(v); o.chunk_set = true; }
    else if (k == "--spec") o.spec = std::stoi(v);
    else if (k == "--draft-head") { o.draft_head = v; ++o.selection_options; }
    else if (k == "--draft-blocks") {
      if (v.empty()) return false;
      o.draft_blocks = v; ++o.selection_options;
    } else if (k == "--draft-tokens") {
      if (v.empty()) return false;
      o.draft_blocks = v; o.draft_dense = true; ++o.selection_options;
    }
    else if (k == "--draft-margin") o.draft_margin = std::stof(v);
    else if (k == "--draft-q") o.draft_q = std::stof(v);
    else if (k == "--draft-adapt") { o.adapt = GenNumber<int>(v); o.adapt_set = true; }
    else if (k == "--draft-head-auto") o.head_auto = GenNumber<int>(v);
    else if (k == "--mtp-prefix") o.mtp_prefix = GenNumber<int>(v);
    else if (k == "--draft-last-head") o.last_head = std::stoi(v);
    else if (k == "--lookup") o.lookup = std::stoi(v);
    else if (k == "--lookup-min") o.lookup_min = std::stoi(v);
    else if (k == "--draft-overlap") o.overlap = std::stoi(v);
    else if (k == "--dump") o.dump = v;
    else if (k == "--dump-state") o.state_dump = v;
    else if (k == "--dump-logits") o.logits_dump = v;
#if CRUCIBLE_DEBUG_NONFINITE & 4
    else if (k == "--inject") o.inject = v;
#endif
    else if (k == "--prefill") o.prefill_rows = std::stoi(v);
    else if (k == "--ctx") o.ctx = std::stoi(v);
    else if (k == "--steps") o.steps = std::stoi(v) != 0;
    else if (k == "--temperature") o.sampling.temperature = GenNumber<double>(v);
    else if (k == "--top-k") o.sampling.top_k = GenNumber<int>(v);
    else if (k == "--top-p") o.sampling.top_p = GenNumber<double>(v);
    else if (k == "--min-p") o.sampling.min_p = GenNumber<double>(v);
    else if (k == "--seed") o.sampling.seed = GenNumber<uint64_t>(v);
    else return false;
  }
  } catch (const std::exception&) { return false; }
  if (o.prefill_rows == kPrefillAuto && o.chunk_set) o.prefill_rows = 0;
  return true;
}

// Validates o and loads its draft selection. Returns 0 or a process exit code.
int CheckGenOptions(const GenOptions& o, std::vector<int>& blocks) {
  if (const std::string error = o.sampling.Error(kVocab); !error.empty()) {
    std::fprintf(stderr, "invalid sampling: %s\n", error.c_str());
    return 2;
  }
  if (o.sampling.temperature > 0 && o.overlap) {
    std::fprintf(stderr, "sampling does not support --draft-overlap; use --draft-overlap 0\n");
    return 2;
  }
  if (o.sampling.temperature > 0 && o.head_auto) {
    std::fprintf(stderr, "sampling does not support the online draft head; use --draft-head-auto 0\n");
    return 2;
  }
  if (o.chunk < 1 || o.chunk > kMaxRows || o.spec < 0 || o.spec >= kMaxRows || o.selection_options > 1 ||
      o.lookup < 0 || o.lookup >= kMaxRows || o.lookup_min < 1 || o.lookup_min > 32 ||
      !std::isfinite(o.draft_margin) || o.draft_margin < 0.f ||
      !std::isfinite(o.draft_q) || o.draft_q < 0.f || o.draft_q > 1.f ||
      (o.draft_head != "full" && o.draft_head != "64k") ||
      o.adapt < 0 || o.adapt > 1 ||
      (AdaptiveDraft(o) && (o.draft_dense || (!BuiltinDraft(o) && o.draft_blocks.empty()))) ||
      o.head_auto < 0 || o.head_auto > 1 ||
      (o.head_auto && ((!BuiltinDraft(o) && o.draft_blocks.empty()) || o.spec == 0)) ||
      (o.last_head != 0 && o.last_head != 1) || (o.overlap != 0 && o.overlap != 1) ||
      o.prefill_rows < kPrefillAuto || o.prefill_rows > kMaxPrefillRows ||
      o.mtp_prefix < 0 || o.mtp_prefix > 262144) {
    std::fprintf(stderr, "bad arguments\n");
    return 2;
  }
  if (!ValidTokenIds(o.ids, kVocab)) {
    std::fprintf(stderr, "token IDs must be in [0, %d)\n", kVocab);
    return 2;
  }
  if (!RequestFitsContext(o.ids.size(), o.n, o.ctx)) {
    std::fprintf(stderr, "the prompt and output must fit --ctx; --n and --ctx must be positive\n");
    return 2;
  }
  blocks.clear();
  if (BuiltinDraft(o)) {
    blocks = ResolveDraftBlocks(o, {});
    return 0;
  }
  if (o.draft_blocks.empty()) return 0;
  std::ifstream f(o.draft_blocks);
  if (!f) {
    std::fprintf(stderr, "cannot open draft selection: %s\n", o.draft_blocks.c_str());
    return 2;
  }
  std::stringstream ss;
  ss << f.rdbuf();
  try {
    blocks = ParseDraftSelection(ss.str());
  } catch (const std::invalid_argument& error) {
    std::fprintf(stderr, "invalid draft selection %s: %s\n", o.draft_blocks.c_str(), error.what());
    return 2;
  }
  const int limit = o.draft_dense ? kVocab : kVocab / 128;
  if (!f || !ValidDraftSelection(blocks, limit)) {
    std::fprintf(stderr, "draft selection %s needs sorted unique IDs in [0, %d)\n", o.draft_blocks.c_str(), limit);
    return 2;
  }
  return 0;
}

// Trunk stacks and the carry row of the current request; the first call allocates them as zeros.
// Greedy and sampled requests keep separate buffers. gen --jobs clears both between jobs.
struct GenScratch { float* stacks; float* carry; };
inline GenScratch& GreedyScratch() {
  static GenScratch scratch{Zeros<float>((size_t)(kMaxRows + 1) * kWide), Zeros<float>(kWide)};
  return scratch;
}
inline GenScratch& SampledScratch() {
  static GenScratch scratch{Zeros<float>(size_t(kMaxRows + 1) * kWide), Zeros<float>(kWide)};
  return scratch;
}

inline bool PublishGen(const GenOptions& o, GenResult& result, size_t first) {
  if (!o.emit && o.stop_ids.empty()) return false;
  const size_t end = std::min(result.gen.size(), size_t(o.n));
  for (size_t i = first; i < end; ++i) {
    const int token = result.gen[i];
    if (o.emit) o.emit(token);
    if (std::find(o.stop_ids.begin(), o.stop_ids.end(), token) != o.stop_ids.end()) {
      result.gen.resize(i + 1);
      result.stopped = true;
      return true;
    }
  }
  return false;
}

inline GenResult RunSampledGen(Engine& eng, const GenOptions& o, const std::vector<int>& blocks) {
  eng.DraftHead(blocks, o.last_head != 0 || o.draft_margin > 0 || o.draft_q > 0, o.draft_dense, AdaptiveDraft(o));
  eng.ObserveDraft(o.ids.data(), int(o.ids.size()));
  float* const stacks = SampledScratch().stacks;
  float* const carry = SampledScratch().carry;
  SampledGenerationOptions options;
  options.sampling = o.sampling;
  options.n = o.n; options.chunk = o.chunk; options.spec = o.n == 1 ? 0 : o.spec;
  options.mtp_prefix = o.mtp_prefix;
  options.lookup = o.lookup; options.lookup_min = o.lookup_min;
  options.draft_margin = o.draft_margin; options.draft_q = o.draft_q;
  options.profile = crucible::SwitchText("CRUCIBLE_PROFILE") != nullptr;
  options.score_prefill = o.last_head == 0;
  options.stop_ids = o.stop_ids; options.dump = o.dump;
  options.conversation = o.conversation;
  options.target_trace = o.sampling_trace;
  GenResult result;
  result.steps.Enable(o.steps);
  const auto copy = [](float* dst, const float* src, size_t count) {
    HIP_OK(hipMemcpyAsync(dst, src, count * sizeof(float), hipMemcpyDeviceToDevice, 0));
  };
  const auto emit = [&](int token) { if (o.emit) o.emit(token); };
  const auto sampled = GenerateSampled(eng, o.ids, options, stacks, carry, copy, emit, result.steps);
  result.gen = sampled.ids;
  result.prefill_s = sampled.prefill_seconds; result.decode_s = sampled.decode_seconds;
  result.forwards = sampled.forwards; result.accepted = sampled.accepted; result.proposed = sampled.attempted;
  result.lookup_steps = sampled.lookup_steps; result.lookup_accepted = sampled.lookup_accepted;
  result.stopped = sampled.stopped;
  return result;
}

// RunPrompt picks the bulk prefill path at compile time; an engine without it would fall back to Forward silently.
static_assert(crucible::HasGemmPrefill<Engine>, "RunGen needs the GEMM prefill engine interface");

// Temperature zero keeps the original engine calls and greedy verification.
GenResult RunGen(Engine& eng, const GenOptions& o, const std::vector<int>& blocks) {
#if CRUCIBLE_DEBUG_NONFINITE & 4
  eng.SetInject(o.inject);
#endif
  const std::vector<int> selected = ResolveDraftBlocks(o, blocks);
  if (o.sampling.temperature > 0) return RunSampledGen(eng, o, selected);
  const std::vector<int>& ids = o.ids;
  const int n = o.n, ctx = o.ctx, spec = o.n == 1 ? 0 : o.spec, lookup = o.lookup, chunk = o.chunk;
  const int prefill_rows = eng.PrefillRows();
  const size_t mtp_start = DraftWarmupStart(ids.size(), o.mtp_prefix);
  GenResult res;
  eng.DraftHead(selected, o.last_head != 0 || o.draft_margin > 0.f || o.draft_q > 0.f,
                o.draft_dense, AdaptiveDraft(o));
  eng.ObserveDraft(ids.data(), (int)ids.size());
  const bool prof = crucible::SwitchText("CRUCIBLE_PROFILE") != nullptr;
  float* const stk = GreedyScratch().stacks;
  float* const carry = GreedyScratch().carry;
  double t0 = Now();
  ConversationResume resume;
  if (o.conversation && o.conversation->restore) resume = o.conversation->restore(stk, carry);
  else eng.Reset();
  // A replayed tail inside the warm-up window keeps its gap-free MTP state. A tail without MTP state (a spec 0 turn)
  // restarts the warm-up at the restored length, as batch serve does. Otherwise a fresh window as usual.
  const bool from_tail = o.conversation && o.conversation->continue_tail;
  const bool continue_mtp = from_tail && resume.length > mtp_start && resume.mpos > 0;
  if (spec && o.mtp_prefix && resume.length && !continue_mtp) {
    if (resume.length > mtp_start && !from_tail) throw std::runtime_error("snapshot overlaps the new MTP warmup window");
    eng.ResetMtp();
    resume = ConversationTargetResume(resume);
  }
  int mpos = resume.mpos;
  float* next_logits = o.logits_dump.empty() ? nullptr : Zeros<float>(kVocab);
  if (prof && prefill_rows) eng.EnableProfile();
  const auto copy = [](float* dst, const float* src, size_t count) {
    HIP_OK(hipMemcpyAsync(dst, src, count * sizeof(float), hipMemcpyDeviceToDevice, 0));
  };
  const std::vector<int> last = RunPrompt(eng, std::span<const int>(ids), chunk, spec > 0, o.dump,
      stk, carry, mpos, copy,
      [&](float* s, const int* tokens, int count, int pos) { eng.Mtp(s, tokens, count, pos, o.last_head == 0); },
      [&](int row) { if (next_logits) copy(next_logits, eng.logits() + size_t(row) * kVocab, kVocab); },
      o.mtp_prefix, resume, o.conversation);
  const bool have_carry = spec && (resume.length < ids.size() || resume.have_carry);
  HIP_OK(hipStreamSynchronize(0));
  res.prefill_s = Now() - t0;
  if (prof && prefill_rows) {
    std::fprintf(stderr, "prefill profile:\n");
    eng.Report(1);
    eng.EnableProfile(false);
  }
  eng.DumpState(o.state_dump, mpos, stk, carry);
  if (next_logits && resume.length == ids.size() && resume.logits.size() == kVocab)
    HIP_OK(hipMemcpyAsync(next_logits, eng.logits(), sizeof(float) * kVocab, hipMemcpyDeviceToDevice, 0));
  if (next_logits) QsaWriteCapture(o.logits_dump, next_logits, kVocab);
  if (next_logits) HIP_OK(hipFree(next_logits));
  eng.ClearExpertCounts();
  std::vector<int>& gen = res.gen;
  gen = {last.back()};
  eng.ObserveDraft(gen.data(), 1);
  PublishGen(o, res, 0);
  FlushDeferredPrompt(o.conversation, stk, carry);
  if (prof) eng.EnableProfile();
  if (prof && o.overlap)
    std::fprintf(stderr, "profile excludes the asynchronous verification head; use wall time for overlap\n");
  SpecSteps& st = res.steps;
  st.Enable(o.steps);
  t0 = Now();
  long& forwards = res.forwards;
  long& accepted = res.accepted;
  // Prompt lookup's history (fed only when lookup is on).
  LookupIndex history(o.lookup_min);
  if (lookup) { history.Append(ids); history.Append(std::span<const int>(&gen.back(), 1)); }
  ConversationResume tail = resume;
  tail.length = eng.Length(); tail.mpos = mpos; tail.replay = 0;
  tail.next_id = gen.back(); tail.have_carry = have_carry; tail.pending_tokens.clear();
  tail.prompt_only = false;
  bool save_tail = o.conversation && o.conversation->save && o.conversation->save_generated_tail && !o.overlap;
  if (spec == 0) {
    for (int i = 1; i < n && !res.stopped; ++i) {
      st.Begin();
      gen.push_back(eng.Forward(&gen.back(), 1, 0, 1, o.dump)[0]);
      if (save_tail) { tail.logits = eng.CopyLogits(1); CopyRows(eng, carry, 0, 0, 1); }
      st.End(SpecSteps::kVerify, 1);
      st.StepDone();
      PublishGen(o, res, gen.size() - 1);
    }
    forwards = long(gen.size()) - 1;
  } else {
    // Pending trunk stacks pair with the exact tokens that follow them.
    if (resume.length != ids.size() || resume.pending_tokens.empty())
      HIP_OK(hipMemcpyAsync(stk, carry, sizeof(float) * kWide, hipMemcpyDeviceToDevice, 0));
    std::vector<int> ctoks = resume.length == ids.size() && !resume.pending_tokens.empty()
                            ? resume.pending_tokens : std::vector<int>{gen.back()};
    int replay = resume.length == ids.size() ? resume.replay : 0;
    tail.pending_tokens = ctoks;
    while ((int)gen.size() < n && !res.stopped) {
      const int rc = (int)ctoks.size();
      const size_t base = eng.Length();
      const int limit = BoundedDraftDepth(std::max(spec, lookup), ctx, (int)base, mpos, rc);
      if (limit == 0) {
        // Commit pending DeltaNet rows once before the serial fallback.
        gen.push_back(eng.Forward(&gen.back(), 1, replay, 1)[0]);
        if (save_tail) {
          tail.logits = eng.CopyLogits(1);
          CopyRows(eng, carry, 0, 0, 1);
          eng.Mtp(stk, ctoks.data(), rc, mpos, false);
          mpos += rc;
          HIP_OK(hipMemcpyAsync(stk, carry, sizeof(float) * kWide, hipMemcpyDeviceToDevice, 0));
          ctoks = {gen.back()};
          tail.mpos = mpos; tail.replay = 0; tail.pending_tokens = ctoks;
          tail.have_carry = false;
        }
        if (lookup) history.Append(std::span<const int>(&gen.back(), 1));
        eng.ObserveDraft(&gen.back(), 1);
        replay = 0;
        ++forwards;
        PublishGen(o, res, gen.size() - 1);
        continue;
      }
      const double head_step_start = Now();
      const bool full_head = o.head_auto && res.head_policy.UseFull();
      st.Begin();
      // No draft past the token limit or after a stop ID can commit a token (draft_stop.h).
      const int room = DraftRoom(size_t(n) - gen.size());
      std::vector<int> drafts{eng.Mtp(stk, ctoks.data(), rc, mpos, true, full_head)[rc - 1]};
      st.End(SpecSteps::kMtpFirst, rc);
      st.Margin(0, eng.DraftMargin());
      mpos += rc;
      std::vector<int> copied = history.Draft(std::min({lookup, limit, room}));
      const bool use_lookup = copied.size() > 1 && copied.front() == drafts.front();
      if (use_lookup) {
        st.Margin(0, -1.f);  // lookup drafts carry no MTP margin
        drafts = copied;
        ++res.lookup_steps;
      } else {
        // --draft-q: q is the estimated chance that every draft so far is
        // accepted. A draft row pays only while q stays above the threshold.
        float q = DraftAcceptProb(eng.DraftMargin());
        for (int k = 1; k < std::min({spec, limit, room}) && ChainMayGrow(drafts, o.stop_ids, room); ++k) {
          if (o.draft_q > 0.f ? q * kDraftNextPrior < o.draft_q : eng.DraftMargin() < o.draft_margin) break;
          st.Begin();
          CopyRows(eng, stk, kMaxRows, k == 1 ? rc - 1 : 0, 1);
          const int next = eng.Mtp(stk + (size_t)kMaxRows * kWide, &drafts.back(), 1, mpos + k - 1,
                                   true, full_head)[0];
          st.End(SpecSteps::kMtpChain, 1);
          q *= DraftAcceptProb(eng.DraftMargin());
          // A dropped draft is the same as a rejected one: its MTP cache rows
          // are rewritten by the next catch-up call.
          if (o.draft_q > 0.f && q < o.draft_q) break;
          st.Margin(k, eng.DraftMargin());
          drafts.push_back(next);
        }
      }
      TrimDrafts(drafts, o.stop_ids, room);
      const int depth = (int)drafts.size();
      res.proposed += depth;
      std::vector<int> rows{gen.back()};
      rows.insert(rows.end(), drafts.begin(), drafts.end());
      st.Begin();
      const std::vector<int> ys = eng.Forward(rows.data(), depth + 1, replay, 1, "",
                                               o.overlap ? mpos : -1);
      st.End(SpecSteps::kVerify, depth + 1);
      int a = 0;
      while (a < depth && drafts[a] == ys[a]) ++a;
      eng.Truncate(base + a + 1);
      if (o.overlap) {
        mpos += a;
        CopyRows(eng, stk, 0, a, 1);
        ctoks = {ys[a]};
      } else {
        CopyRows(eng, stk, 0, 0, a + 1);
        ctoks.assign(ys.begin(), ys.begin() + a + 1);
      }
      const size_t first = gen.size();
      gen.insert(gen.end(), ys.begin(), ys.begin() + a + 1);
      if (lookup) history.Append(std::span<const int>(ys.data(), a + 1));
      eng.ObserveDraft(ys.data(), a + 1);
      if (use_lookup) res.lookup_accepted += a;
      replay = a;
      if (save_tail) {
        tail.mpos = mpos; tail.replay = replay; tail.next_id = ys[a];
        tail.pending_tokens = ctoks; tail.logits = eng.CopyLogits(1, a);
        tail.have_carry = false;
      }
      accepted += a;
      ++forwards;
      st.Accepted(depth, a);
      st.StepDone();
      PublishGen(o, res, first);
      if (o.head_auto) res.head_policy.Observe(a + 1, (Now() - head_step_start) * 1e3);
    }
    if (!res.stopped) gen.resize(n);
  }
  res.decode_s = Now() - t0;
  if (save_tail && res.forwards > 0) {
    const size_t logical = ids.size() + gen.size() - 1;
    if (eng.Length() > logical) {
      // Stop and length limits can cut an accepted window. Its first row is committed.
      const int keep = ConversationRetainedRows(eng.Length(), logical, tail.pending_tokens.size());
      if (!keep) save_tail = false;
      else {
        eng.Truncate(logical);
        tail.replay = keep - 1;
        tail.pending_tokens.resize(keep);
        tail.logits = eng.CopyLogits(1, keep - 1);
      }
    }
    if (save_tail) {
      tail.length = eng.Length(); tail.next_id = gen.back();
      o.conversation->save(tail, stk, carry);
    }
  }
  return res;
}

void PrintGen(const GenOptions& o, const GenResult& r) {
  std::printf("ids:");
  for (int g : r.gen) std::printf(" %d", g);
  std::printf("\nprefill %zu tok in %.3f s (%.1f tok/s), decode %d tok in %.3f s (%.2f tok/s)", o.ids.size(),
              r.prefill_s, o.ids.size() / r.prefill_s, o.n - 1, r.decode_s, (o.n - 1) / r.decode_s);
  if (o.spec > 0)
    std::printf(", %ld verify forwards, %.2f drafts accepted per forward (%.1f%%)", r.forwards,
                r.forwards ? (double)r.accepted / r.forwards : 0.,
                r.proposed ? 100.0 * r.accepted / r.proposed : 0.);
  if (o.lookup > 0)
    std::printf(", %ld lookup steps, %ld lookup drafts accepted", r.lookup_steps, r.lookup_accepted);
  std::printf("\n");
  if (o.head_auto)
    std::printf("head policy: restricted %ld steps, full %ld steps, final %s\n",
                r.head_policy.Steps(false), r.head_policy.Steps(true),
                r.head_policy.IncumbentFull() ? "full" : "restricted");
}

#include "../serve/serve.h"
#include "../serve/batched_serve.h"

inline int ServeCommand(int argc, char** argv) {
  ServeSettings settings;
  try {
    std::vector<std::string_view> args;
    for (int index = 3; index < argc; ++index) args.emplace_back(argv[index]);
    settings = ParseServeSettings(args);
  } catch (const std::runtime_error& error) {
    std::fprintf(stderr, "invalid serve options: %s\n", error.what());
    std::fprintf(stderr, "usage: crucible serve CKPT [--ctx N] [--spec D] [--draft-q Q] "
                         "[--draft-head full|64k] [--mtp-prefix N] [--draft-adapt 0|1] "
                         "[--conversation-cache-mib 0..131072] [--prefill 0..4096] "
                         "[--conversation-cache-exact-retry 0|1]\n");
    return 2;
  }
  return Serve(argv[2], settings);
}

#include "batch_cli.h"

// What one gen command prints after RunGen. gen --jobs prints the same into each job's files.
inline void FinishGen(Engine& eng, const GenOptions& o, const GenResult& r) {
  if (AdaptiveDraft(o) && o.spec > 0 && o.n > 1)
    std::fprintf(stderr, "final draft vocabulary %d tokens\n", eng.DraftVocabulary());
  eng.Report((int)r.forwards);
  PrintGen(o, r);
  if (o.steps) r.steps.Print(stderr);
}

#include "gen_jobs.h"

inline int GenJobsCommand(int argc, char** argv) {
  if (argc != 5) {
    std::fprintf(stderr, "usage: crucible gen CKPT --jobs FILE (the jobs carry all other options)\n");
    return 2;
  }
  std::ifstream file(argv[4]);
  if (!file) {
    std::fprintf(stderr, "cannot open job file: %s\n", argv[4]);
    return 2;
  }
  std::stringstream text;
  text << file.rdbuf();
  std::vector<GenJob> jobs;
  try {
    jobs = ParseGenJobs(text.str());
  } catch (const std::invalid_argument& error) {
    std::fprintf(stderr, "invalid job file %s: %s\n", argv[4], error.what());
    return 2;
  }
  if (const std::string issue = GenJobsIssue(jobs); !issue.empty()) {
    std::fprintf(stderr, "job file %s refused: %s\n", argv[4], issue.c_str());
    return 2;
  }
  const GenOptions& load = jobs.front().options;
  std::fprintf(stderr, "gen jobs: %zu jobs after one load (--ctx %d --prefill %s)\n", jobs.size(), load.ctx,
               PrefillText(load.prefill_rows).c_str());
  Engine eng(argv[2], load.ctx, load.prefill_rows);
  try {
    RunGenJobs(eng, jobs, FinishGen);
  } catch (const std::runtime_error& error) {
    std::fprintf(stderr, "gen jobs stopped: %s\n", error.what());
    return 1;
  }
  return 0;
}

// A bench program that includes engine.hip defines ENGINE_LIBRARY, gets this CLI as CrucibleCliMain
// and writes its own main. The served binary leaves ENGINE_LIBRARY unset and keeps main.
#ifdef ENGINE_LIBRARY
#define ENGINE_CLI_ENTRY CrucibleCliMain
#else
#define ENGINE_CLI_ENTRY main
#endif
int ENGINE_CLI_ENTRY(int argc, char** argv) {
  // Before any HIP setup: the version needs no GPU and no checkpoint.
  if (crucible::IsVersionRequest(argc, argv)) {
    std::printf("%s\n", crucible::VersionLine().c_str());
    return 0;
  }
  // Apply before dispatch: serve and pfbench also initialize HIP below.
  if (!SetRuntimeMemoryDefaults()) {
    std::perror("cannot set HIP host allocation policy");
    return 1;
  }
  std::fprintf(stderr, "HSA_USERPTR_FOR_PAGED_MEM=%s HSA_USE_SVM=%s GPU_PINNED_MIN_XFER_SIZE=%s\n",
               crucible::SwitchText("HSA_USERPTR_FOR_PAGED_MEM"), crucible::SwitchText("HSA_USE_SVM"),
               crucible::SwitchText("GPU_PINNED_MIN_XFER_SIZE"));
  // The numerics mode decides which switches apply; a bad value stops here, before any engine is
  // built, so every command (serve, gen, gen --jobs, batch, pfbench) exits 2 and no request meets it.
  try {
    ValidateNumericsSwitches();
    std::fprintf(stderr, "numerics: %s\n", NumericsModeText());
  }
  catch (const std::invalid_argument& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 2;
  }
  if (argc >= 3 && std::string(argv[1]) == "serve") {
    if (const auto batch = TryBatchedServeCommand(argc, argv)) return *batch;
    return ServeCommand(argc, argv);
  }
  if (argc >= 3 && std::string(argv[1]) == "pfbench") return PrefillBench(argc, argv);
  if (argc < 3 || std::string(argv[1]) != "gen") {
    std::fprintf(stderr,
                 "usage: crucible gen CKPT --ids a,b,c [--n N] [--chunk C] [--spec D] [--dump DIR] "
                 "[--ctx N] [--draft-blocks F | --draft-tokens F] [--draft-margin M] [--lookup D] "
                 "[--draft-last-head 0|1] [--draft-overlap 0|1] [--draft-q Q] [--steps 0|1] "
                 "[--temperature T] [--top-k K] [--top-p P] [--min-p P] [--seed S] "
                 "[--draft-head-auto 0|1] [--draft-head full|64k] [--mtp-prefix N]\n"
                 "       (default: --spec 7 --draft-q 0.3 --draft-head 64k --mtp-prefix 256;\n"
                 "       --spec 0 is serial, --draft-head full uses the full head, --mtp-prefix 0 is full warmup)\n"
                 "       [--prefill R|0] (default: automatic GEMM; --chunk C selects small rows)\n"
                 "       [--dump-state DIR] [--dump-logits FILE]\n"
                 "       crucible gen CKPT --jobs FILE   (one load, one job per line: "
                 "{\"args\":[gen options],\"stdout\":F,\"stderr\":F}; every job needs the same --ctx and --prefill)\n"
                 "       crucible serve CKPT [--ctx N] [--spec D] [--draft-q Q] "
                 "[--draft-head full|64k] [--mtp-prefix N] [--draft-adapt 0|1] "
                 "[--conversation-cache-mib 0..131072] [--prefill 0..4096] "
                 "[--conversation-cache-exact-retry 0|1]\n");
    return 2;
  }
  if (argc >= 4 && std::string(argv[3]) == "--jobs") return GenJobsCommand(argc, argv);
  if (HasBatchOptions(argc, argv)) return BatchCommand(argc, argv);
  GenOptions o = CliGenOptions();
  std::vector<int> blocks;
  if (!ParseGenOptions(argc, argv, 3, o)) {
    std::fprintf(stderr, "bad arguments\n");
    return 2;
  }
  if (const int rc = CheckGenOptions(o, blocks)) return rc;
  Engine eng(argv[2], o.ctx, o.prefill_rows);
  const GenResult r = RunGen(eng, o, blocks);
  FinishGen(eng, o, r);
  return 0;
}
