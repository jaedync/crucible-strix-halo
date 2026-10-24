// Engine members stay separate from decode plan construction.
// The prefill GEMMs decode EXL3 K3..8 (PrefillWidthSupported; K2 is a decode-path width).
// Returns the unsupported tensors, or an empty string. The constructor checks
// right after the weights load, before any row-sized allocation.
std::string Engine::PrefillWidthIssue() const {
  std::string bad;
  auto check=[&](const Lin& w, const std::string& name) {
    if (w.tiles && !PrefillWidthSupported(w.K)) bad+=" "+name+"(K"+std::to_string(w.K)+")";
  };
  for (int l=0; l<kLayers && bad.size()<200; ++l) {
    const auto& L=m_.L[l]; const std::string p="L"+std::to_string(l)+".";
    if (L.attn) {
      check(L.qsa.q,p+"q"); check(L.qsa.k,p+"k"); check(L.qsa.v,p+"v"); check(L.qsa.idx,p+"idx");
      check(L.qsa.o,p+"o");
    } else {
      check(L.gdn.qkv,p+"qkv"); check(L.gdn.z,p+"z"); check(L.gdn.out,p+"out");
    }
    check(L.moe.sg,p+"sg"); check(L.moe.su,p+"su"); check(L.moe.sd,p+"sd");
    // Routed experts may also be K3.5 (code 35), which only the kPrefillMoeK7Half kernels hold, or K2
    // and K2.5 (code 25), which only the kPrefillMoeK6Low kernels hold (PrefillMoeKmax below).
    for (const auto& e : L.moe.experts)
      for (int i=0; i<3; ++i)
        if (!PrefillWidthSupported(e.K[i]) && !HalfDecodeK(e.K[i]) && e.K[i]!=2) {
          bad+=" "+p+"experts(K"+KName(e.K[i])+")"; break;
        }
    if (bad.empty() && !PrefillMoeKmax(MoeLayerWidths(L))) bad+=" "+p+"moe(widths)";
  }
  return bad;
}

// The widths of a layer's MoE projections (shared expert, then every routed expert): PrefillMoeKmax's input.
std::vector<int> Engine::MoeLayerWidths(const Layer& L) {
  std::vector<int> widths{L.moe.sg.K,L.moe.su.K,L.moe.sd.K};
  for (const auto& e : L.moe.experts) widths.insert(widths.end(),{e.K[0],e.K[1],e.K[2]});
  return widths;
}

// kPrefillAuto picks the GEMM path when the weights allow it and falls back
// to the small-row path otherwise; an explicit row count must be supported.
int Engine::ChoosePrefillRows(int requested) {
  if (requested==0) { prefill_path_="small-row path (--prefill 0)"; return 0; }
  // Routed experts have no front split (ExpertPtrs; Model::ExpertW rejects
  // one), so the shared experts are the only MoE weights to check.
  std::vector<std::pair<std::string,int>> moe_front;
  for (int l=0; l<kLayers; ++l) {
    const auto& M=m_.L[l].moe; const std::string p="L"+std::to_string(l)+".";
    moe_front.insert(moe_front.end(),{{p+"sg",M.sg.front_s},{p+"su",M.su.front_s},{p+"sd",M.sd.front_s}});
  }
  if (const std::string front=PrefillMoeFrontIssue(moe_front); !front.empty()) {
    std::fprintf(stderr,"MoE prefill GEMMs read engine-order tiles, but these are in front order:%s; "
                 "run with --prefill 0\n",front.c_str());
    std::exit(2);
  }
  const std::string bad=PrefillWidthIssue();
  if (bad.empty()) {
    const int rows=requested==kPrefillAuto ? kPrefillDefaultRows : requested;
    prefill_path_="GEMM path, "+std::to_string(rows)+"-row chunks";
    return rows;
  }
  const std::string why="prefill GEMMs need EXL3 K3..8 weights; this checkpoint has"+bad;
  if (requested!=kPrefillAuto) {
    std::fprintf(stderr,"%s; run with --prefill 0\n",why.c_str());
    std::exit(2);
  }
  prefill_path_="small-row path ("+why+")";
  return 0;
}

void Engine::InitPrefill() {
  // The persistent scan needs full-row q/k/v/ut/y and g/beta for one core part.
  const size_t core=(size_t)(std::min(prefill_rows_,kPrefillCoreRows)+63)/64*64;
  constexpr size_t vectors=48*64*128, matrices=48*64*64;
  // The vscan path keeps q and k of the whole launch ([part][16][rows][128]): the input GEMM's
  // GdnPrepEpi writes them before the first part. +16 MiB each per 2,048 rows over 6,144.
  static_assert(kPrefillCoreRows==2048,"GdnPrepEpi takes 2,048-row parts");
  const size_t gdn_parts=(size_t)(std::max(pf_room_rows_,1)+kPrefillCoreRows-1)/kPrefillCoreRows;
  const size_t qk=std::max(48*core*128,gdn_parts*prefill::kGdnPrepPartFloats);
  pf_gdn_.q=Scratch<float>(qk); pf_gdn_.k=Scratch<float>(qk);
  pf_gdn_.v=Scratch<float>(48*core*128); pf_gdn_.w=Scratch<float>(vectors);
  pf_gdn_.ut=Scratch<float>(48*core*128); pf_gdn_.u=Scratch<float>(vectors);
  pf_y_=Scratch<float>(48*core*128); pf_gdn_.y=pf_y_; pf_gdn_.wk=Scratch<float>(vectors);
  pf_gdn_.g=Scratch<float>(48*core); pf_gdn_.beta=Scratch<float>(48*core);
  pf_gdn_micro_=NumericsOn("CRUCIBLE_PF_GDN_MICRO");
  pf_gdn_.lower=Scratch<float>(matrices); pf_gdn_.qk=Scratch<float>(matrices);
  pf_ab_=Scratch<float>(pf_room_rows_*96);
  pf_prompt_stacks_=Scratch<float>((size_t)prefill_rows_*kWide);
  if (pf_pad_h_) {
    pf_hstride_=kWide+64;
    pf_H_=Scratch<float>((size_t)pf_room_rows_*pf_hstride_);
  }
  // Fragment-order copies of the trunk HC fn and up (1.29 GB, one buffer)
  // for the bulk dots and mix: 4K -44 ms, same bits.
  // Decode keeps the row-major weights; the MTP layer's HC keeps them too (fnf and upf stay null).
  {
    const size_t fnh=(size_t)(kRank+kStreams+15)/16*16*kWide,uph=(size_t)kWide*kRank;
    __half* p=Zeros<__half>((fnh+uph)*kLayers*2);
    for (Layer& L : m_.L)
      for (Hc* hc : {&L.hca,&L.hcm}) {
        if ((size_t)(hc->M+15)/16*16*kWide>fnh) std::abort();
        hc->fnf=p; hc->upf=p+fnh; p+=fnh+uph;
        HcFragFnKernel<<<(fnh+255)/256,256>>>(hc->fn,hc->M,hc->fnf);
        HcFragUpKernel<<<(uph+255)/256,256>>>(hc->up,hc->upf);
      }
    HIP_OK(hipGetLastError()); HIP_OK(hipDeviceSynchronize());
    std::fprintf(stderr,"prefill HC: fragment-order fn and up copies\n");
  }
  pf_thi_=Scratch<__half>((size_t)pf_room_rows_*kRank);
  pf_tlo_=Scratch<__half>((size_t)pf_room_rows_*kRank);
  pf_context_=Scratch<float>((size_t)prefill_rows_*kQsaOutWidth);
  pf_qhi_=Scratch<__half>((size_t)prefill_rows_*kQsaOutWidth);
  // Two-pass top-k scratch, only for contexts past kQsaCandidates blocks.
  if (ctx_/4>kQsaCandidates)
    pf_qsa_select_=Scratch<float>(QsaSelectScratchFloats(std::min(prefill_rows_,kPrefillCoreRows)));
  // Approved prefill numerics: fp16 high halves for the HC dots and QSA
  // operands and fp16 pd.
  pf_split_=NumericsOn("CRUCIBLE_PF_EXACT_SPLIT");
  pf_qlo_=pf_split_ ? Scratch<__half>((size_t)prefill_rows_*kQsaOutWidth) : nullptr;
  std::fprintf(stderr,"prefill numerics: %s\n",pf_split_ ? "exact split (CRUCIBLE_PF_EXACT_SPLIT)"
               : "fp16 HC dots and QSA operands, fp16 pd");
  auto& a=pf_moe_;
  const size_t routes=(size_t)pf_room_rows_*kTopK, rows=(size_t)pf_room_rows_*kSlots;
  a.dispatch.selected=sel_moe_;
  a.dispatch.tile_counts=Zeros<int>(kExperts*((routes+255)/256));
  a.dispatch.counts=Zeros<int>(kExperts); a.dispatch.offsets=Zeros<int>(kExperts+1);
  a.dispatch.pairs=Zeros<int>(routes); a.dispatch.row_ids=Zeros<int>(routes);
  a.inverse=Zeros<int>(routes); a.item_counts=Zeros<int>(2);
  pf_tiles_=PrefillTilesFromEnv(pf_tiles_);
  pf_x_=Scratch<__half>((size_t)pf_room_rows_*kHidden);
  // Without the union list the flash runs per row: another key order.
  if (!NumericsOn("CRUCIBLE_PF_QSA_ROWS")) {
    const int part=std::min(prefill_rows_,kPrefillCoreRows);
    pf_qsa_union_={Zeros<int>(prefill::QsaUnionListInts(part)),
                   Zeros<int>((part+prefill::kUnionRows-1)/prefill::kUnionRows)};
  }
  pf_plegate_=Scratch<float>((size_t)pf_room_rows_*kStreams);
  a.gu=Zeros<LinJob>((kExperts+1)*2); a.down=Zeros<LinJob>(kExperts+1);
  a.g_gu=Zeros<Exl3GemmJob>((kExperts+1)*2); a.g_down=Zeros<Exl3GemmJob>(kExperts+1);
  const size_t items=((rows+31)/32+kExperts+1)*40;
  a.i_gu=Zeros<Exl3GemmItem>(items); a.i_down=Zeros<Exl3GemmItem>(items);
  // pg, pu, pd, xg, xu and xd live in the stage arena (AllocateStageArena).
  a.x=x_;
  a.fx_items=Zeros<int>(2*((rows+15)/16+kExperts+1));
  a.fx_uoff=Zeros<int>(kExperts+1); a.fx_xoff=Zeros<int>(kExperts+1);
  a.fx_ready=Zeros<int>((rows+15)/16+kExperts+1); a.fx_count=Zeros<int>(3);
  const char* prune=crucible::SwitchText("CRUCIBLE_PF_PRUNE");
  pf_prune_=!(prune && std::string(prune)=="0");
  Build(pf_head_,{Job(m_.head,x_,kHidden,logits_,kVocab,kEpiStore,nullptr,1)},1,false,0,{m_.head.front_s});
}

const Engine::PrefillPlans& Engine::PrefillPlansFor(int R) {
  const auto it=prefill_plans_.find(R);
  if (it!=prefill_plans_.end()) { it->second.first=++plan_tick_; return *it->second.second; }
  // A server sees many prompt lengths: keep the most recently used plan sets.
  if (prefill_plans_.size()>=kPrefillPlanCache) FreePrefillPlans(true);
  const auto t0=std::chrono::steady_clock::now();
  auto p=std::make_unique<PrefillPlans>();
  for (int l=0; l<kLayers; ++l) {
    const auto& w=m_.L[l]; auto& g=p->layers[l];
    std::vector<LinJob> input;
    std::vector<int> front;
    if (IsQsa(l)) {
      input={Job(w.qsa.q,x_,kHidden,q_,kQsaQWidth,kEpiStore,nullptr,R),
        Job(w.qsa.k,x_,kHidden,k_,kQsaKvWidth,kEpiStore,nullptr,R),
        Job(w.qsa.v,x_,kHidden,v_,kQsaKvWidth,kEpiStore,nullptr,R),
        Job(w.qsa.idx,x_,kHidden,idx_,kQsaIdxWidth,kEpiStore,nullptr,R)};
      front={w.qsa.q.front_s,w.qsa.k.front_s,w.qsa.v.front_s,w.qsa.idx.front_s};
    } else {
      input={Job(w.gdn.qkv,x_,kHidden,qkv_,kGdnQkvWidth,kEpiStore,nullptr,R),
        Job(w.gdn.z,x_,kHidden,z_,kGdnValueWidth,kEpiStore,nullptr,R)};
      front={w.gdn.qkv.front_s,w.gdn.z.front_s};
    }
    g.input.Build(input,pf_tiles_.In(R),nullptr,partial_,partial_capacity_,xh_,xh_capacity_,front);
    const Lin& out=w.attn ? w.qsa.o : w.gdn.out;
    // An ablated output projection stores y; AblateWrite does the write-back.
    const LinJob oj=abl_attn_[l] ? Job(out,nullptr,0,abl_y_,kHidden,kEpiStore,nullptr,R)
                                 : Job(out,nullptr,0,PrefillH(R),PrefillHStride(R),kEpiResidual,post_,R);
    g.output.Build({oj},pf_tiles_.Out(R),pf_xpack_,partial_,partial_capacity_,xh_,xh_capacity_,
                   {out.front_s});
  }
  const size_t bytes=p->DeviceBytes();
  if (bytes>kPrefillPlanBytes)
    std::fprintf(stderr,"prefill plan exceeds 1 GiB; retain only this custom plan\n");
  // One new plan can temporarily exceed the retained budget during its build.
  while (PrefillPlanBudgetExceeded(prefill_plans_.size(),prefill_plan_bytes_,bytes))
    FreePrefillPlans(true);
  prefill_plan_bytes_+=bytes;
  auto& slot=prefill_plans_[R];
  slot={++plan_tick_,std::move(p)};
  std::fprintf(stderr,"prefill plan for %d rows built in %.1f ms (%zu cached, %.1f MiB)\n",R,
               std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count(),
               prefill_plans_.size(),prefill_plan_bytes_/1048576.0);
  return *slot.second;
}

// Frees the least recently used plan set, or all of them. Plans own device
// arrays that queued kernels may still read, so the device drains first.
void Engine::FreePrefillPlans(bool oldest_only) {
  if (prefill_plans_.empty()) return;
  HIP_OK(hipDeviceSynchronize());
  auto old=prefill_plans_.begin();
  for (auto i=prefill_plans_.begin(); i!=prefill_plans_.end(); ++i)
    if (i->second.first<old->second.first) old=i;
  for (auto i=prefill_plans_.begin(); i!=prefill_plans_.end();) {
    if (oldest_only && i!=old) { ++i; continue; }
    const size_t bytes=i->second.second->DeviceBytes();
    prefill_plan_bytes_-=bytes;
    if (oldest_only)
      std::fprintf(stderr,"prefill plan for %d rows evicted (%.1f MiB)\n",i->first,bytes/1048576.0);
    for (auto& layer : i->second.second->layers) { layer.input.Free(); layer.output.Free(); }
    i=prefill_plans_.erase(i);
    if (oldest_only) break;  // The erased LRU iterator is no longer comparable.
  }
}

// Last-layer row pruning (idea from Gufo, gufo-org/gufo, MIT): after the last layer writes its K/V and indexer
// rows, only the final trunk stacks remain, and a chunk whose caller reads none of them (no head, no MTP
// warm-up row) needs none. Dumps, the interleave check and the QSA captures read every row, so
// they keep the full layer.
bool Engine::PrefillPrunes(int R, bool predict, int keep, const std::string& dump) const {
  return pf_prune_ && !predict && keep>=R && dump.empty() && !interleave_check_ && m_.L[kLayers-1].attn &&
         !crucible::SwitchText("CRUCIBLE_QSA_CAPTURE_DIR");
}

// Partial pruning of a chunk with read rows: the last layer's HC, router, MoE and combine run only on rows from
// the returned row (a multiple of kPrefillPruneAlign; 0 runs every row). The head reads R-1 and the MTP warm-up
// reads rows from keep. Every skipped kernel computes each row on its own, with the shapes of the whole chunk.
int Engine::PrefillFirstRow(int R, bool predict, int keep, const std::string& dump) const {
  if (!PrefillPrunes(R,false,R,dump)) return 0;
  return PruneFirstRow(R,predict,keep,kPrefillPruneAlign);
}

void Engine::PrefillBlock(int li, const Layer& L, const PrefillLayer& P, int R, int pos0, bool cache_only,
                          int first) {
  // Above 8 rows the bulk inputs are fp16 (x_ is fp16-exact); the fp32 opt-in was deleted in train 18.
  const bool half_x=R>8;
  HcRead(L.hca,R,true,half_x ? pf_x_ : nullptr,half_x,PrefillH(R),PrefillHStride(R)); Mark("hc_mix");
  const auto lin=[&](const char* n) { Mark(n); };
  // GDN q/k prep in the input GEMM epilogue, on the vscan path (not micro, not eager).
  const bool gdn_epi=!L.attn && !cache_only && !pf_gdn_micro_ && !prefill::GdnEager() &&
                     P.input.fused;
  if (gdn_epi)
    P.input.Run(true,true,lin,false,half_x ? pf_x_ : nullptr,
                prefill::GdnPrepEpi{qkv_,pf_gdn_.q,pf_gdn_.k,L.gdn.conv,R});
  else P.input.Run(true,true,lin,false,half_x ? pf_x_ : nullptr);
  Mark(L.attn ? "qsa_proj" : "gdn_proj");
  if (cache_only) {
    // The same cache kernels as LaunchQsaPrefill on the same parts; the layer's output rows are dead.
    for (int o=0; o<R; o+=kPrefillCoreRows) {
      auto a=QArgs(L.qsa,sequence_->qc.at(li),nullptr,std::min(kPrefillCoreRows,R-o),pos0+o,sequence_->Mrope());
      a.q+=(size_t)o*kQsaQWidth; a.k+=(size_t)o*kQsaKvWidth; a.v+=(size_t)o*kQsaKvWidth; a.idx+=(size_t)o*kQsaIdxWidth;
      prefill::LaunchQsaPrefillCache(a);
      PrefillCutAfter(o+std::min(kPrefillCoreRows,R-o),{sequence_->qc.at(li).ik_raw});
    }
    Mark("qsa_prep");
    return;
  }
  // The lanes GDN and QSA gate kernels write the packed out-projection input
  // directly, so the out-projection needs no repack.
  const bool direct=P.output.repack && !pf_gdn_micro_;
  __half* out=direct ? P.output.repack : P.output.jobs[0].xh;
  if (L.attn) {
    // The QSA and GDN prefill kernels take at most kPrefillCoreRows rows; later
    // parts see earlier parts through the cache or state, as chunks do.
    for (int o=0; o<R; o+=kPrefillCoreRows) {
      const int n=std::min(kPrefillCoreRows,R-o), skip=std::clamp(first-o,0,n);
      if (skip==n) {
        // No output row of this part is read: cache rows only.
        auto c=QArgs(L.qsa,sequence_->qc.at(li),nullptr,n,pos0+o,sequence_->Mrope());
        c.q+=(size_t)o*kQsaQWidth; c.k+=(size_t)o*kQsaKvWidth; c.v+=(size_t)o*kQsaKvWidth; c.idx+=(size_t)o*kQsaIdxWidth;
        prefill::LaunchQsaPrefillCache(c);
        PrefillCutAfter(o+n,{sequence_->qc.at(li).ik_raw});
        continue;
      }
      auto a=QArgs(L.qsa,sequence_->qc.at(li),direct ? out : out+(size_t)o*kQsaOutWidth,std::min(kPrefillCoreRows,R-o),pos0+o,
                   sequence_->Mrope());
      if (direct) { a.pack_rows=R; a.pack_off=o; }
      a.q+=(size_t)o*kQsaQWidth; a.k+=(size_t)o*kQsaKvWidth; a.v+=(size_t)o*kQsaKvWidth; a.idx+=(size_t)o*kQsaIdxWidth;
      const auto mark=[&](const char* n) { Mark(n); };
      const prefill::QsaPrefill qp{a,pf_context_,pf_qhi_,pf_qlo_,pf_qsa_select_};
      if (pf_qsa_union_.list) prefill::LaunchQsaPrefillUnion(qp,pf_qsa_union_,mark,nullptr,skip);
      else prefill::LaunchQsaPrefill(qp,mark,nullptr,skip);
      if (!warming_) CaptureQsa(a,li,false);
      PrefillCutAfter(o+n,{sequence_->qc.at(li).ik_raw});
    }
    Mark("qsa_gate");
  } else {
    // x_ is fp16-exact, so one WMMA pass gives exact products for a/b.
    if (half_x) LaunchDenseHalf(L.gdn.ab,96,pf_x_,pf_ab_,96,R);
    else LaunchDense(L.gdn.ab,96,kHidden,x_,kHidden,pf_ab_,96,R);
    Mark("gdn_ab");
    for (int o=0; o<R; o+=kPrefillCoreRows) {
      const int n=std::min(kPrefillCoreRows,R-o);
      auto a=pf_gdn_;
      a.input={qkv_+(size_t)o*kGdnQkvWidth,z_+(size_t)o*kGdnValueWidth,half_x ? nullptr : x_+(size_t)o*kHidden,L.gdn.ab,L.gdn.conv,
               L.gdn.A_log,L.gdn.dt_bias,L.gdn.gamma,sequence_->ghist.at(li),sequence_->gstate.at(li),L.gdn.out.suh,
               direct ? out : out+(size_t)o*kGdnValueWidth,sequence_->gpend.at(li),pos0+o,n,0,n};
      if (direct) { a.input.pack_rows=R; a.input.pack_off=o; }
      a.ab_logits=pf_ab_+(size_t)o*96; a.chunks=(n+63)/64;
      if (gdn_epi) {
        a.q=pf_gdn_.q+(size_t)(o/kPrefillCoreRows)*prefill::kGdnPrepPartFloats;
        a.k=pf_gdn_.k+(size_t)(o/kPrefillCoreRows)*prefill::kGdnPrepPartFloats;
      }
      if (pf_gdn_micro_) prefill::LaunchGdnMicro(a,pf_y_);
      else prefill::LaunchGdnLanes(a,nullptr,true,gdn_epi ? P.input.shape.rm*16 : 0);
      PrefillCutAfter(o+n,{sequence_->gstate.at(li),sequence_->ghist.at(li),sequence_->gpend.at(li)});
    }
    Mark("gdn_core");
  }
  P.output.Run(false,true,lin,direct);
  if (abl_attn_[li]) AblateWrite(abl_attn_[li],PrefillH(R),PrefillHStride(R),R);
  Mark(L.attn ? "qsa_oproj" : "gdn_oproj");
  // The MoE half keeps the shapes and numerics of all R rows; it runs on rows [first, R) only.
  const PrefillGemmShape moe=pf_tiles_.Moe(R);
  const bool half_pd=!pf_split_ && PrefillMoeHalfOut(moe);
  const int shape_rows=R;
  float* const H=PrefillH(shape_rows)+(size_t)first*PrefillHStride(shape_rows);
  const int ldh=PrefillHStride(shape_rows);
  R-=first;
  HcRead(L.hcm,R,true,xh_,half_x && moe.packed,H,ldh,shape_rows); Mark("hc_mix");
  LaunchDense(L.moe.router,kExperts+1,kHidden,x_,kHidden,rlog_,kExperts+1,R,true);
  PrefillMoeRouteWaveKernel<<<(R+7)/8,256>>>(rlog_,sel_moe_,wts_,R); Mark("moe_router");
  // K6 projections (4.05 shared expert) select the K6 kernel instantiations, K7 and K8 (h6_ng8
  // shared experts) the wide ones, K3.5 experts with K7 the kPrefillMoeK7Half ones, K2 and K2.5 experts
  // with K6 the kPrefillMoeK6Low ones (PrefillWidthIssue rejects any other set before the GEMM path runs).
  const int kmax=PrefillMoeKmax(MoeLayerWidths(L));
  const bool k6=kmax==6;
  // Above 8 rows HcRead(hcm) leaves the fp16 copy of x_ in xh_ for the router
  // and the packed input transform.
  auto a=pf_moe_; a.dispatch.rows=R; a.tile_rows=16*moe.rm; a.packed=moe.packed; a.xh=xh_;
  if (R<=8) PrefillPackKernel<<<(R*kHidden+255)/256,256>>>(x_,xh_,nullptr,R*kHidden);
  prefill::LaunchMoeDispatch(a.dispatch);
  HIP_OK(hipMemsetAsync(a.item_counts,0,2*sizeof(int),0));
  // Packed inputs: one queue kernel runs the input transform and gate/up.
  const bool fused=a.packed;
  if (fused) PrefillMoeFxPlanKernel<<<1,512>>>(a,L.moe.d_experts,L.moe.sg,L.moe.su,L.moe.sd);
  else PrefillMoePlanKernel<<<1,512>>>(a,L.moe.d_experts,L.moe.sg,L.moe.su,L.moe.sd);
  Mark("moe_route");
  // M1: packed RM 2, 4 and 6 run gate/up and the mid step in one
  // kernel (bitwise equal to FX + Mid); other shapes keep FX + Mid. 4.05
  // (k6) is back on the pair path with the RM6 window body (findings
  // section 11); merge only after its night pair passes the kill line.
  // Only the byte-tested K sets take the pair path (prefill_moe_kset.h).
  const bool tested=PrefillMoePairKSet(k6,L.moe.sg.K,L.moe.su.K,L.moe.sd.K,L.moe.experts);
  const bool pair=fused && tested && PrefillMoePairShape(moe);
  if (pair) {
    LaunchPrefillMoePair(moe.rm,a,k6); Mark("moe_gateup");
  } else if (fused) {
    LaunchPrefillMoeFxK(moe.rm,a,kmax); Mark("moe_gateup");
  } else {
    if (a.packed) PrefillMoeInputPackedKernel<<<dim3(40,(R*kSlots+15)/16),256>>>(a);
    else PrefillMoeInputKernel<<<(R*kSlots*40+7)/8,256>>>(a);
    Mark("moe_xform");
    LaunchPrefillMoeGemmK(moe,a.g_gu,a.i_gu,a.item_counts,kmax); Mark("moe_gateup");
  }
  if (!pair) { PrefillMoeMidKernel<<<(R*kSlots*5+7)/8,256>>>(a); Mark("moe_mid"); }
  LaunchPrefillMoeGemmK(moe,a.g_down,a.i_down,a.item_counts+1,kmax,half_pd); Mark("moe_down");
  if (const float* ab=AblMlp(li)) {
    if (half_pd) PrefillMoeCombineKernel<true,true><<<(R*20+7)/8,256>>>(a,wts_,nullptr,abl_y_,kHidden);
    else PrefillMoeCombineKernel<false,true><<<(R*20+7)/8,256>>>(a,wts_,nullptr,abl_y_,kHidden);
    AblateWrite(ab,H,ldh,R);
  } else if (half_pd) PrefillMoeCombineKernel<true><<<(R*20+7)/8,256>>>(a,wts_,post_,H,ldh);
  else PrefillMoeCombineKernel<false><<<(R*20+7)/8,256>>>(a,wts_,post_,H,ldh);
  Mark("moe_combine");
}

// x4-tail: the spec of SetPrefillJoin for this call. The retained ledger gets the
// window start of the plan without joins (the snapshot holds it), and the cut copies arm for PrefillBlock. Runs
// after retained.Forward, before any kernel. Host code only.
void Engine::BeginPrefillJoin(int R, int pos0) {
  pf_cut_={}; pf_cut_done_.clear();
  if (!std::exchange(pf_join_set_,false)) return;
  crucible::PrefillJoinSpec join=std::exchange(pf_join_,{});
  std::vector<std::pair<const void*,size_t>> spans;
  for (const auto& span : sequence_->FixedSpans()) spans.emplace_back(span.pointer,span.bytes);
  for (const auto& [layer,cache] : sequence_->qc)
    if (layer!=kLayers && cache.raw_slots)
      spans.emplace_back(cache.ik_raw,size_t(cache.raw_slots)*kQsaRawDim*sizeof(float));
  if (!crucible::PrefillJoinValid(join,R,prefill_rows_,spans)) {
    std::fprintf(stderr,"prefill join plan fault (x4-tail): rows %d, ledger %d, cut %d, %zu copies\n",R,join.ledger,
                 join.cut,join.copies.size());
    std::abort();
  }
  if (join.ledger) {
    const size_t end=size_t(pos0)+size_t(R);
    if (!crucible::RetainedPrefix::FromFields({end,end,size_t(pos0+join.ledger),0},sequence_->retained)) std::abort();
  }
  if (!join.cut) return;
  pf_cut_done_.assign(join.copies.size(),0);
  pf_cut_=std::move(join);
}

// x4-tail: after the core part that ends at row `end` of the launch, the copies of `sources` (when the cut is there).
// The null stream keeps them in order with the kernels: each copy reads its span as the part left it.
void Engine::PrefillCutAfter(int end, std::initializer_list<const void*> sources) {
  if (!pf_cut_.cut || end!=pf_cut_.cut) return;
  for (const void* source : sources)
    for (size_t i=0; i<pf_cut_.copies.size(); ++i) {
      const auto& copy=pf_cut_.copies[i];
      if (copy.source!=source) continue;
      if (pf_cut_done_[i]) std::abort();
      HIP_OK(hipMemcpyAsync(copy.target,copy.source,copy.bytes,hipMemcpyDeviceToDevice,0));
      pf_cut_done_[i]=1;
    }
}

// x4-tail: the PLE ring at the cut holds rows cut-16..cut-1 of this launch (the cut is at least one core part in).
// The ring kernel of LaunchPrefillPle runs again with the cut buffer as its ring and the cut as its rows, while
// pf_plenv_ still holds the launch's fp16 values.
void Engine::PrefillCutPle(int pos0) {
  if (!pf_cut_.cut) return;
  if (!batch_streams_.empty() || VerificationSpansActive()) std::abort();
  for (size_t i=0; i<pf_cut_.copies.size(); ++i) {
    const auto& copy=pf_cut_.copies[i];
    if (copy.source!=sequence_->ple_ring) continue;
    if (pf_cut_done_[i]) std::abort();
    PleArgs a{kv_,m_.ple.nk1,m_.ple.nq1,m_.ple.nc1,m_.ple.conv,static_cast<float*>(copy.target),nullptr,pos0,
              pf_cut_.cut,0};
    PrefillPleRingKernel<<<dim3(kWide/256,16),256>>>(a,pf_plenv_);
    HIP_OK(hipGetLastError());
    pf_cut_done_[i]=1;
  }
}

std::vector<int> Engine::Prefill(const int* tokens, int R, const std::string& dump, bool predict,
                                 const PrefillBetween* between, int keep) {
  if (!ContextRowsFit((int)Length(),R,sequence_->capacity,pf_room_rows_)) std::abort();
  if (!ValidTokenIds(std::span<const int>(tokens,R),kVocab)) std::abort();
  // Prefill commits every row, so the retained ledger advances like Forward(R,0,R).
  const std::string state_error=sequence_->retained.Forward(R,0,R);
  if (!state_error.empty()) {
    std::fprintf(stderr,"prefill state error: %s; use Reset() to start a new sequence\n",state_error.c_str());
    std::exit(1);
  }
  const int pos0=sequence_->hist.size();
  BeginPrefillJoin(R,pos0);
  sequence_->hist.insert(sequence_->hist.end(),tokens,tokens+R);
  const auto& plans=PrefillPlansFor(R);
  if (!pf_host_tokens_) pf_host_tokens_=std::make_unique<PinnedInputBuffer<int,kMaxPrefillLaunchRows>>();
  if (!pf_host_rows_) pf_host_rows_=std::make_unique<PinnedInputBuffer<int64_t,kMaxPrefillLaunchRows*16>>();
  pf_host_tokens_->CopyTo(tok_,tokens,R);
  pf_host_rows_->FillTo(rows_,R*16,[&](int64_t* rows) { PleRows(pos0,R,rows); });
  PleRowsUploaded(pf_host_rows_->data(),R*16);
  float* const bulk_H=PrefillH(R);
  const int ldh=PrefillHStride(R);
  if (between && !InterleaveAllowed(R,kMaxRows,bulk_H!=H_)) between=nullptr;
  Mark("start"); EmbedKernel<<<dim3(kHidden/256,R),256>>>(m_.embed,tok_,bulk_H,ldh);
  OverlayImages(*sequence_,pos0,R,bulk_H,ldh);
  Mark("embed"); Dump(dump,0,R,bulk_H,ldh);
  const bool prune=PrefillPrunes(R,predict,keep,dump);
  const int first=prune ? 0 : PrefillFirstRow(R,predict,keep,dump);
  RunTrunk([&](int) { Ple(R,pos0,bulk_H,ldh); PrefillCutPle(pos0); Mark("ple"); },
           [&](int l) {
             const bool last=l==kLayers-1;
             PrefillBlock(l,m_.L[l],plans.layers[l],R,pos0,prune && last,last ? first : 0);
           },
           [&](int l) {
             Dump(dump,l+1,R,bulk_H,ldh);
             if (between && InterleaveAfterLayer(l,between->every,kLayers)) {
               // The decode pass selects its own streams and profile scope; the chunk continues on this sequence.
               SequenceState* const prefilling=sequence_;
               const std::string scope=profile_scope_;
               between->run();
               sequence_=prefilling; profile_scope_=scope;
             }
           });
  // x4-tail: every cut copy ran once (a source that no part reached is a plan fault).
  for (const char done : pf_cut_done_) if (!done) std::abort();
  pf_cut_={}; pf_cut_done_.clear();
  // The stacks stay in bulk_H: the head reads its row there, and readers copy rows with CopyPrefillStacks (T2).
  last_prefill_rows_=R;
  if (!predict) { Collect(); HIP_OK(hipDeviceSynchronize()); CheckChunk(pos0,R,false); return {}; }
  HcReadR<1>(m_.final_hc,false,R-1,bulk_H+(size_t)(R-1)*ldh);
  if (R>1) HIP_OK(hipMemcpyAsync(x_,x_+(size_t)(R-1)*kHidden,kHidden*sizeof(float),hipMemcpyDeviceToDevice,0));
  Mark("hc_mix"); RunGroup(pf_head_,true,true);
  ArgmaxKernel<<<1,1024>>>(logits_,kVocab,ids_); Mark("head"); Collect();
  CheckChunk(pos0,R,true);
  int result; HIP_OK(hipMemcpy(&result,ids_,sizeof(int),hipMemcpyDeviceToHost));
#if CRUCIBLE_DEBUG_NONFINITE & 1
  RequireNoNonfiniteSites();
#endif
  sampling::RequireFiniteHeadIds(std::span(&result,1));
  return {result};
}

// The plans of the bulk MTP warm-up for R rows, or null when a weight has no prefill GEMM (EXL3 K3..6) or a job does
// not fit the shared workspaces: the 8-row path then runs.
const Engine::MtpPrefillPlan* Engine::MtpPrefillPlanFor(int R) {
  const auto it=mtp_prefill_plans_.find(R);
  if (it!=mtp_prefill_plans_.end()) return it->second.get();
  // Keep the full chunk and one other row count; plans own device arrays that queued kernels may read.
  if (mtp_prefill_plans_.size()>=2) {
    HIP_OK(hipDeviceSynchronize());
    for (auto i=mtp_prefill_plans_.begin(); i!=mtp_prefill_plans_.end();)
      if (i->first!=prefill_rows_) {
        if (i->second) {
          for (auto& g : i->second->fuse) g.Free();
          i->second->layer.input.Free();
        }
        i=mtp_prefill_plans_.erase(i);
      } else ++i;
  }
  auto& slot=mtp_prefill_plans_[R];
  const auto& mtp=m_.mtp;
  const auto& qsa=mtp.L.qsa;
  for (const Lin* lin : {&mtp.fe,&mtp.fh,&qsa.q,&qsa.k,&qsa.v,&qsa.idx})
    if (!PrefillWidthSupported(lin->K)) return nullptr;
  // E (eple_), N (qkv_), fc_embedding (z_) and fc_hidden (kv_): attention scratch that the cache-only MTP block
  // never reads. Each holds R rows of its width (AllocateStageArena).
  std::vector<LinJob> jobs{Job(mtp.fe,eple_,kHidden,z_,kHidden,kEpiStore,nullptr,R)};
  std::vector<int> front{mtp.fe.front_s};
  for (int st=0; st<kStreams; ++st) {
    jobs.push_back(Job(mtp.fh,qkv_+st*kHidden,kWide,kv_+st*kHidden,kWide,kEpiStore,nullptr,R));
    front.push_back(mtp.fh.front_s);
  }
  for (size_t j=0; j<jobs.size(); ++j) jobs[j].S=front[j];  // carried through PackJobs, reset by Build
  const auto groups=crucible::PackJobs(jobs,xh_capacity_,partial_capacity_,
                                       [](const LinJob& a,const LinJob& b) { return (a.S>0)==(b.S>0); });
  if (groups.empty()) return nullptr;
  auto p=std::make_unique<MtpPrefillPlan>();
  for (const auto& group : groups) {
    std::vector<int> fs;
    for (const LinJob& job : group) fs.push_back(job.S);
    std::vector<LinJob> built=group;
    for (LinJob& job : built) job.S=1;
    p->fuse.emplace_back();
    p->fuse.back().Build(built,pf_tiles_.In(R),nullptr,partial_,partial_capacity_,xh_,xh_capacity_,fs);
  }
  p->layer.input.Build({Job(qsa.q,x_,kHidden,q_,kQsaQWidth,kEpiStore,nullptr,R),Job(qsa.k,x_,kHidden,k_,kQsaKvWidth,kEpiStore,nullptr,R),
                        Job(qsa.v,x_,kHidden,v_,kQsaKvWidth,kEpiStore,nullptr,R),Job(qsa.idx,x_,kHidden,idx_,kQsaIdxWidth,kEpiStore,nullptr,R)},
                       pf_tiles_.In(R),nullptr,partial_,partial_capacity_,xh_,xh_capacity_,
                       {qsa.q.front_s,qsa.k.front_s,qsa.v.front_s,qsa.idx.front_s});
  slot=std::move(p);
  return slot.get();
}

bool Engine::MtpWarm(const float* first, const float* rows, const int* tokens, int count, int mpos) {
  if (prefill_rows_<=0 || VerificationSpansActive() || !crucible::MtpBulkFits(count,pf_launch_rows_))
    return false;
  if (!ContextRowsFit(mpos,count,sequence_->mtp_capacity,pf_launch_rows_)) {
    std::fprintf(stderr,"MTP context full or invalid row count\n"); std::exit(1);
  }
  if (!ValidTokenIds(std::span<const int>(tokens,count),kVocab)) {
    std::fprintf(stderr,"token IDs must be in [0, %d)\n",kVocab); std::exit(2);
  }
  EnsureMtpCache(*sequence_,mpos);
  const MtpPrefillPlan* plan=MtpPrefillPlanFor(count);
  if (!plan) return false;
  if (!pf_host_tokens_) pf_host_tokens_=std::make_unique<PinnedInputBuffer<int,kMaxPrefillLaunchRows>>();
  pf_host_tokens_->CopyTo(tok_,tokens,count);
  profile_scope_="mtp_bulk:";
  Mark("start");
  MtpPrepPairsKernel<<<count,1024>>>(m_.embed,tok_,first,rows,m_.mtp.ne1,m_.mtp.nh1,eple_,qkv_);
  const auto lin=[&](const char* n) { Mark(n); };
  for (const auto& group : plan->fuse) group.Run(true,true,lin);
  MtpFuseStrideKernel<<<dim3(kWide/256,count),256>>>(kv_,z_,PrefillH(count),PrefillHStride(count));
  Mark("mtp_in");
  // Only the MTP cache rows matter here: the layer's output rows are dead (cache-only, as a pruned last layer).
  PrefillBlock(kLayers,m_.mtp.L,plan->layer,count,mpos,true,0);
  Collect();
  return true;
}

// Check mode: the chunk's trunk stacks (tight rows) and its final logits row, keyed by the full prefix.
void Engine::CheckChunk(int pos0, int R, bool predict) {
  if (!interleave_check_) return;
  std::vector<float> host((size_t)R*kWide);
  HIP_OK(hipMemcpy(host.data(),TightRows(PrefillH(R),PrefillHStride(R),R),host.size()*sizeof(float),
                   hipMemcpyDeviceToHost));
  uint64_t logits=0;
  if (predict) {
    std::vector<float> row(kVocab);
    HIP_OK(hipMemcpy(row.data(),logits_,row.size()*sizeof(float),hipMemcpyDeviceToHost));
    logits=Fnv1a(row.data(),row.size()*sizeof(float));
  }
  interleave_check_.Chunk(Fnv1a(std::span<const int>(sequence_->hist.data(),sequence_->hist.size())),pos0,R,
                          Fnv1a(host.data(),host.size()*sizeof(float)),logits);
}

void Engine::DumpState(const std::string& dir, int mtp_rows, const float* stk, const float* carry) {
  if (dir.empty()) return;
  for (const auto& [l,state] : sequence_->gstate) {
    const auto base=dir+"/gdn_"+std::to_string(l);
    QsaWriteCapture(base+".state.f32",state,kGdnHeads*kGdnDim*kGdnDim);
    QsaWriteCapture(base+".hist.f32",sequence_->ghist.at(l),16*kGdnQkvWidth);
  }
  QsaWriteCapture(dir+"/ple.ring.f32",sequence_->ple_ring,16*kWide);
  // QSA caches of the committed rows (the MTP layer, kLayers, up to mtp_rows), so a check can compare every cache
  // byte that the last layer and the MTP warm-up wrote.
  for (const auto& [l,c] : sequence_->qc) {
    const size_t n=l==kLayers ? (size_t)std::max(mtp_rows,0) : Length();
    if (!n) continue;
    const auto base=dir+"/qsa_"+std::to_string(l);
    QsaWriteCapture(base+".k.f16",c.k,n*kQsaKvWidth);
    QsaWriteCapture(base+".v.f16",c.v,n*kQsaKvWidth);
    QsaWriteCapture(base+".raw.f32",c.ik_raw,QsaRawSnapshotBytes(c.raw_slots,n)/sizeof(float));
    QsaWriteCapture(base+".pool.f32",c.ik_pool,QsaPoolFloats((int)n));
  }
  if (stk) QsaWriteCapture(dir+"/mtp.stk.f32",stk,(size_t)(kMaxRows+1)*kWide);
  if (carry) QsaWriteCapture(dir+"/mtp.carry.f32",carry,kWide);
}
