// Restore one prefix before each timing sample. This excludes rollback drift.
#define ENGINE_LIBRARY
#include "../engine/engine.hip"
#include <array>
#include <filesystem>

namespace {
struct SavedBuffer { void* target; uint8_t* copy; size_t bytes; };
struct EngineStateIO {
#ifdef CRUCIBLE_ROW_PORT
  static void Configure(Engine& e, int R, RowPolicy policy) {
    e.PlansFor(R);
    auto& p = *e.plans_.at(R); p.policy = policy;
    auto group = [&](Group& g, bool expert = false) {
      g.fuse = expert ? 0 : policy.fuse; g.cache = policy.cache;
      for (auto& j : g.jobs) j.cache_policy = policy.cache;
      HIP_OK(hipMemcpy(g.d_jobs,g.jobs.data(),g.jobs.size()*sizeof(LinJob),hipMemcpyHostToDevice));
    };
    for (auto& layer : p.L) {
      layer.state_cache = policy.state;
      group(layer.a_in); group(layer.a_out); group(layer.gu,true); group(layer.dn,true);
    }
    group(p.head); group(p.mtp_in);
  }
  static void ConfigureAll(Engine& e, bool tuned) {
    setenv("CRUCIBLE_TUNE_ROWS", tuned ? "1" : "0", 1);
    for (int R=1;R<=8;++R)
      Configure(e,R,ResolveRowPolicy(R,e.m_.head.K,e.m_.L[0].moe.experts[0].K[0]));
  }
#endif
  static size_t LinBytes(const Lin& lin) { return size_t(lin.I)*lin.O*lin.K/8+2*lin.I+2*lin.O; }
  static size_t HcBytes(const Hc& hc) { return size_t(hc.M+kRank)*kWide*2+4*kWide; }
  static std::pair<size_t,size_t> WeightBytes(Engine& e) {
    size_t dense=HcBytes(e.m_.final_hc)+LinBytes(e.m_.head);
    dense+=size_t(12800)*kHidden*2+3*4*kWide+4*4*kWide+16*160*2;
    for (const auto& layer : e.m_.L) {
      dense+=HcBytes(layer.hca)+HcBytes(layer.hcm)+size_t(kExperts+1)*kHidden*2;
      dense+=LinBytes(layer.moe.sg)+LinBytes(layer.moe.su)+LinBytes(layer.moe.sd);
      if (layer.attn) {
        const auto& q=layer.qsa;
        dense+=LinBytes(q.q)+LinBytes(q.k)+LinBytes(q.v)+LinBytes(q.idx)+LinBytes(q.o)+4*(256*2+128*2);
      } else {
        const auto& g=layer.gdn;
        dense+=LinBytes(g.qkv)+LinBytes(g.z)+LinBytes(g.out)+2*96*kHidden+4*(4*10240+48*2+128);
      }
    }
    const auto& expert=e.m_.L[0].moe.experts[0];
    size_t routed=0;
    for (int K : expert.K) routed+=size_t(kHidden)*kMoeInter*K/8+2*(kHidden+kMoeInter);
    return {dense,routed};
  }
  static std::vector<SavedBuffer> Save(Engine& e) {
    std::vector<SavedBuffer> saved;
    auto add = [&](void* p, size_t n) {
      uint8_t* q; HIP_OK(hipMalloc(&q, n));
      HIP_OK(hipMemcpy(q, p, n, hipMemcpyDeviceToDevice));
      saved.push_back({p, q, n});
    };
    SequenceState& state = *e.sequence_;
    for (const auto& [l,p] : state.gstate) add(p, sizeof(float)*48*128*128);
    for (const auto& [l,p] : state.ghist) add(p, sizeof(float)*16*10240);
    add(state.ple_ring, sizeof(float)*16*kWide);
    for (const auto& [l,c] : state.qc) {
      add(c.k, sizeof(__half)*e.ctx_*512); add(c.v, sizeof(__half)*e.ctx_*512);
      add(c.ik_raw, QsaRawSnapshotBytes(c.raw_slots, e.ctx_));
      add(c.ik_pool, sizeof(float)*QsaPoolFloats(e.ctx_));
    }
    return saved;
  }
  static void Restore(Engine& e, const std::vector<SavedBuffer>& saved,
                      const std::vector<int>& prefix) {
    for (const auto& b : saved) HIP_OK(hipMemcpyAsync(b.target,b.copy,b.bytes,hipMemcpyDeviceToDevice,0));
    SequenceState& state = *e.sequence_;
    state.hist = prefix;
    state.retained.Reset();
    if (!state.retained.Forward((int)prefix.size(),0,(int)prefix.size()).empty()) std::exit(1);
    HIP_OK(hipDeviceSynchronize());
  }
  static std::vector<uint8_t> Outputs(Engine& e, int R) {
    const size_t hbytes = size_t(R)*kWide*sizeof(float), lbytes = size_t(R)*kVocab*sizeof(float);
    std::vector<uint8_t> out(hbytes+lbytes);
    HIP_OK(hipMemcpy(out.data(),e.H_,hbytes,hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(out.data()+hbytes,e.logits_,lbytes,hipMemcpyDeviceToHost));
    return out;
  }
};
}

int main(int argc,char** argv) {
  if (argc != 3 && argc != 4) {
    std::fprintf(stderr,"usage: forward_rows CHECKPOINT IDS_FILE [NEW_DUMP_DIR]\n"); return 2;
  }
  if (argc == 4) {
    if (std::filesystem::exists(argv[3])) { std::fprintf(stderr,"use a new dump directory\n"); return 2; }
    std::filesystem::create_directories(std::string(argv[3])+"/chunk1");
    std::filesystem::create_directories(std::string(argv[3])+"/chunk8");
  }
  std::ifstream file(argv[2]); std::stringstream text; text << file.rdbuf();
  const auto prefix = ParseIds(text.str());
  if (!file || prefix.empty() || prefix.size()>504 || !ValidTokenIds(prefix,kVocab)) return 2;
#ifdef CRUCIBLE_ROW_PORT
  setenv("CRUCIBLE_TUNE_ROWS","1",1);
#endif
  Engine engine(argv[1],512); engine.WarmPlans(); engine.CountExperts(true);
#ifdef CRUCIBLE_ROW_PORT
  EngineStateIO::ConfigureAll(engine,false);
#endif
  for (size_t i=0;i<prefix.size();i+=8) {
    const int n=std::min<size_t>(8,prefix.size()-i);
    engine.Forward(prefix.data()+i,n,0,n);
  }
  const auto saved=EngineStateIO::Save(engine);
  // Rows 9..13: CRUCIBLE_SPEC_SPAN (one lone verify span past kMaxRows rows).
  const std::array<int,kSpecSpanMax> tokens{71093,12305,198,727,73111,1393,25,514,279,1879,315,13,576};
  const int span=engine.SpanRows();
  if (prefix.size()+size_t(span)>512) return 2;
  const size_t one_h=kWide*sizeof(float), one_l=kVocab*sizeof(float);
  std::vector<uint8_t> serial_h(size_t(span)*one_h), serial_l(size_t(span)*one_l);
  EngineStateIO::Restore(engine,saved,prefix);
  for (int r=0;r<span;++r) {
    engine.Forward(tokens.data()+r,1,0,1);
    const auto output=EngineStateIO::Outputs(engine,1);
    std::copy_n(output.begin(),one_h,serial_h.begin()+r*one_h);
    std::copy_n(output.begin()+one_h,one_l,serial_l.begin()+r*one_l);
  }
  const auto [dense,routed]=EngineStateIO::WeightBytes(engine);
  std::printf("weights: dense=%zu routed_expert=%zu serial_roof_ms=%.3f\n",dense,routed,(dense+routed*480)/245.5e6);
#ifdef CRUCIBLE_ROW_PORT
  const std::array<std::pair<const char*,RowPolicy>,9> policies{{
    {"base",{}},{"fuse",{3,0,0,0,24}},{"hints",{3,4,4,0,24}},
    {"b2w8",{3,4,4,2,8}},{"b2w16",{3,4,4,2,16}},{"b2w24",{3,4,4,2,24}},
    {"b4w8",{3,4,4,4,8}},{"b4w16",{3,4,4,4,16}},{"b4w24",{3,4,4,4,24}}}};
  std::fprintf(stdout,"R,policy,median_ms,min_ms,experts_per_layer,roof_ms,roof_pct,exact\n");
#else
  const std::array<std::pair<const char*,int>,1> policies{{{"base",0}}};
  std::fprintf(stdout,"R,policy,median_ms,min_ms,experts_per_layer,roof_ms,roof_pct,exact\n");
#endif
  // R 9..span: one lone verify span (ForwardVerifySpans, the serve path of a solo step past kMaxRows rows).
  for (int R=1;R<=span;++R) {
    std::vector<uint8_t> reference(serial_h.begin(),serial_h.begin()+R*one_h);
    reference.insert(reference.end(),serial_l.begin(),serial_l.begin()+R*one_l);
    for (const auto& [name,policy] : policies) {
#ifdef CRUCIBLE_ROW_PORT
      EngineStateIO::Configure(engine,R,policy);
#endif
      std::vector<double> times; engine.ClearExpertCounts();
    for (int rep=0;rep<7;++rep) {
      EngineStateIO::Restore(engine,saved,prefix);
      const double begin=Now();
      if (R<=kMaxRows) engine.Forward(tokens.data(),R,0,1);
      else engine.ForwardVerifySpans({VerifySpan{0,std::vector<int>(tokens.begin(),tokens.begin()+R),0,1}});
      if (rep>=2) times.push_back((Now()-begin)*1000);
    }
    const auto counts=engine.ExpertCounts();
    const double experts=(double)counts[2*R]/counts[2*R+1];
    std::sort(times.begin(),times.end());
    const double roof=(dense+routed*48*experts)/245.5e6;
    const auto output=EngineStateIO::Outputs(engine,R);
    const bool exact=output==reference;
    std::printf("%d,%s,%.6f,%.6f,%.3f,%.3f,%.2f,%s\n",R,R<=kMaxRows ? name : "span",times[2],times[0],experts,roof,
                100*roof/times[2],exact ? "PASS" : "FAIL");
    std::fflush(stdout);
    if (!exact) return 1;
    }
    // FORWARD_ROWS_PROFILE=1: per-stage GPU time of this R (event marks; the marks add their own small cost).
    if (std::getenv("FORWARD_ROWS_PROFILE") && R<=kMaxRows) {
      std::fprintf(stderr, "R=%d ", R);
      engine.EnableProfile(true);
      for (int rep=0;rep<5;++rep) { EngineStateIO::Restore(engine,saved,prefix); engine.Forward(tokens.data(),R,0,1); }
      engine.Report(5);
      engine.EnableProfile(false);
    }
  }
  GenOptions options; options.ids=prefix; options.n=160; options.ctx=512;
#ifdef CRUCIBLE_ROW_PORT
  EngineStateIO::ConfigureAll(engine,false);
#endif
  const auto result=RunGen(engine,options,{});
  PrintGen(options,result);
#ifdef CRUCIBLE_ROW_PORT
  EngineStateIO::ConfigureAll(engine,true);
  const auto tuned=RunGen(engine,options,{});
  PrintGen(options,tuned);
  std::printf("serial160 exact: %s\n",result.gen==tuned.gen ? "PASS" : "FAIL");
  if (result.gen!=tuned.gen) return 1;
  options.spec=2;
  const auto spec=RunGen(engine,options,{});
  PrintGen(options,spec);
  std::printf("gate3 spec2: %s\n",result.gen==spec.gen ? "PASS" : "FAIL");
  if (result.gen!=spec.gen) return 1;
  if (argc == 4) {
    options.n=1; options.spec=0;
    for (int chunk : {1,8}) {
      options.chunk=chunk; options.dump=std::string(argv[3])+"/chunk"+std::to_string(chunk);
      RunGen(engine,options,{});
    }
    std::printf("gate dumps complete: %s\n",argv[3]);
  }
#endif
  for (const auto& b : saved) HIP_OK(hipFree(b.copy));
}
