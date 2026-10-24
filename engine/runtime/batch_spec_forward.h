// Included inside Engine. Packed verification (G6d): one pass over independent spans.
 private:
  bool VerificationSpansActive() const { return !verify_spans_.empty(); }
  struct ActiveVerifySpan { int stream,row,pos,rows,replay,commit; };
  std::vector<ActiveVerifySpan> verify_spans_;
 public:
  std::vector<int> ForwardVerifySpans(const std::vector<VerifySpan>& spans,const std::string& dump="") {
    if (!batch_streams_.empty() || VerificationSpansActive()) throw std::invalid_argument("nested verification pass");
    std::vector<SpecPrefix> prefixes;
    for (const auto& state:sequences_) prefixes.push_back({state->hist.size(),state->capacity,state->retained});
    // Only a pass above kMaxRows rows asks MaxPassRows (it may build the 16-row plans).
    size_t total=0;
    for (const auto& span:spans) total+=span.tokens.size();
    const int cap=total>size_t(kMaxRows)?MaxPassRows():kMaxRows;
    if (const auto error=SpecSpansError(spans,prefixes,cap,kMaxRows,kVocab); !error.empty())
      throw std::invalid_argument(error);
    std::vector<int> tokens;
    for (const auto& span:spans) tokens.insert(tokens.end(),span.tokens.begin(),span.tokens.end());
    const int rows=tokens.size();
    const Plans& P=PlansFor(rows);
    std::vector<int64_t> ple_rows(size_t(rows)*16);
    verify_spans_.reserve(spans.size());
    for (const auto& span:spans) sequences_[span.stream]->hist.reserve(prefixes[span.stream].length+span.tokens.size());
    struct Clear { std::vector<ActiveVerifySpan>& spans; ~Clear() { spans.clear(); } } clear{verify_spans_};
    PrepareVerifySpans(spans,ple_rows);
    const BatchHostInputs input(tokens,std::move(ple_rows));
    lookahead_.Start(P.policy);
    host_tokens_.CopyTo(tok_,input.tokens().data(),input.tokens().size());
    host_rows_.CopyTo(rows_,input.rows().data(),input.rows().size());
    PleRowsUploaded(host_rows_.data(),(int)input.rows().size());
    profile_scope_="batch_verify:";
    Mark("start");
    EmbedKernel<<<dim3(kHidden/256,rows),256>>>(m_.embed,tok_,H_);
    Mark("embed"); Dump(dump,0,rows);
    RunTrunk([&](int) { Ple(rows,0); Mark("ple"); },
             [&](int layer) { Block(layer,m_.L[layer],P.L[layer],rows,0,0,1); },
             [&](int layer) { Dump(dump,layer+1,rows); });
    return Head(m_.final_hc,P,rows);
  }
  // One MTP pass over several requests' rows (grouped drafting). Each span is Mtp(stacks, tokens, R, mpos, score)
  // on its own stream: the shared projections run over all rows, attention per span on its stream's MTP cache.
  // Each scored span's last row gets its own HC read (as Mtp) and one draft-head pass scores all of them.
  // Leaves span i's rows in H() from row Offset(i). Returns one choice per span ({-1, 0} when not scored).
  struct MtpSpan { int stream; const float* stacks; std::vector<int> tokens; int mpos; bool score; };
  std::vector<DraftChoice> MtpSpans(const std::vector<MtpSpan>& spans) {
    if (!batch_streams_.empty() || VerificationSpansActive()) throw std::invalid_argument("nested MTP pass");
    int rows=0;
    for (const auto& span:spans) {
      if (span.stream<0 || size_t(span.stream)>=sequences_.size() || span.tokens.empty())
        throw std::invalid_argument("invalid MTP span");
      if (!ContextRowsFit(span.mpos,int(span.tokens.size()),sequences_[span.stream]->mtp_capacity,kMaxRows) ||
          !ValidTokenIds(std::span(span.tokens),kVocab)) throw std::invalid_argument("MTP span outside its context");
      rows+=int(span.tokens.size());
    }
    if (rows<1 || rows>kMaxRows) throw std::invalid_argument("an MTP pass holds one to eight rows");
    const bool scoring=std::any_of(spans.begin(),spans.end(),[](const MtpSpan& span) { return span.score; });
    const bool block_head=draft_last_only_ && draft_blocks_ && !draft_dense_;
    // Checked before any device work, so a refused call leaves no pass behind and no lookahead running.
    if (scoring && !block_head) throw std::logic_error("grouped MTP scoring needs the block draft head in last-row mode");
    struct Restore { SequenceState*& current; SequenceState* saved; ~Restore() { current=saved; } } restore{sequence_,sequence_};
    struct Clear { std::vector<ActiveVerifySpan>& spans; ~Clear() { spans.clear(); } } clear{verify_spans_};
    std::vector<int> tokens;
    int offset=0;
    for (const auto& span:spans) {
      EnsureMtpCache(*sequences_[span.stream],span.mpos);
      const int count=int(span.tokens.size());
      HIP_OK(hipMemcpyAsync(mtp_stacks_+size_t(offset)*kWide,span.stacks,sizeof(float)*count*kWide,
                            hipMemcpyDeviceToDevice,0));
      tokens.insert(tokens.end(),span.tokens.begin(),span.tokens.end());
      verify_spans_.push_back({span.stream,offset,span.mpos,count,0,count});
      offset+=count;
    }
    const Plans& P=PlansFor(rows);
    lookahead_.Start(P.policy);
    host_tokens_.CopyTo(tok_,tokens.data(),rows);
    profile_scope_="mtp_spans:";
    Mark("start");
    MtpPrepKernel<<<rows,1024>>>(m_.embed,tok_,mtp_stacks_,m_.mtp.ne1,m_.mtp.nh1,mE_,mN_);
    RunGroup(P.mtp_in,true,true);
    MtpFuseKernel<<<dim3(kWide/256,rows),256>>>(mhp_,mep_,H_);
    Mark("mtp_in");
    Block(kLayers,m_.mtp.L,P.L[kLayers],rows,0,0,rows);
    std::vector<int> scored;
    for (size_t i=0;i<spans.size();++i) if (spans[i].score) scored.push_back(int(i));
    std::vector<DraftChoice> out(spans.size(),DraftChoice{-1,0.f});
    if (scored.empty()) {
      lookahead_.Stop();
      Collect();
      return out;
    }
    const Hc& hc=m_.mtp.mixer;
    for (size_t k=0;k<scored.size();++k) {
      const auto& span=verify_spans_[size_t(scored[k])];
      const float* last=H_+size_t(span.row+span.rows-1)*kWide;
      float* x=x_+k*kHidden;
      if (hc_wmma_) HcWLaunch(hc.fn,hc.M,hc.up,hc.w1,last,1,x,nullptr,hcw_,++hcw_epoch_);
      else {
        HcDotsKernel<1><<<((hc.M+1)*kStreams+7)/8,256>>>(hc.fn,hc.M,last,dots_,ss_);
        HcMixKernel<1><<<kHidden/16,256>>>(hc.up,hc.w1,dots_,hc.M,ss_,last,x,nullptr);
      }
    }
    Mark("hc_mix");
    const int n=int(scored.size());
    RunGroup(DraftGroup(n),true,true);
    for (int k=0;k<n;++k)
      DraftChoiceKernel<<<1,1024>>>(logits_+size_t(k)*draft_head_.O,draft_head_.O,draft_blocks_,choice_+k,false);
    Mark("draft_head");
    lookahead_.Stop();
    Collect();
    std::vector<DraftChoice> choices(n);
    HIP_OK(hipMemcpy(choices.data(),choice_,sizeof(DraftChoice)*n,hipMemcpyDeviceToHost));
    for (int k=0;k<n;++k) out[size_t(scored[k])]=choices[size_t(k)];
    return out;
  }
  // Row offset of MTP span i in H() after MtpSpans (the spans' rows are packed in order).
 private:
  // The block draft head for n rows (n = 1 is draft_group_). Rebuilt with the head (RebuildDraftHead clears them).
  const Group& DraftGroup(int n) {
    if (n==1) return draft_group_;
    auto& group=draft_groups_[size_t(n)];
    if (!group) {
      group=std::make_unique<Group>();
      Build(*group,{Job(draft_head_,x_,kHidden,logits_,draft_head_.O,kEpiStore,nullptr,n)},n,true,0,{draft_head_.front_s});
    }
    return *group;
  }
  std::array<std::unique_ptr<Group>,kMaxRows+1> draft_groups_;
  float* mtp_stacks_=nullptr;
  void PrepareVerifySpans(const std::vector<VerifySpan>& spans,std::vector<int64_t>& rows) {
    struct Restore { SequenceState*& current; SequenceState* saved; ~Restore() { current=saved; } } restore{sequence_,sequence_};
    int offset=0;
    for (const auto& span:spans) {
      sequence_=sequences_[span.stream].get();
      const int pos=sequence_->hist.size(),count=span.tokens.size();
      const auto error=sequence_->retained.Forward(count,span.replay,span.commit);
      if (!error.empty()) throw std::logic_error("verification prefix changed after validation");
      sequence_->hist.insert(sequence_->hist.end(),span.tokens.begin(),span.tokens.end());
      PleRows(pos,count,rows.data()+size_t(offset)*16);
      verify_spans_.push_back({span.stream,offset,pos,count,span.replay,span.commit});
      offset+=count;
    }
  }
  void VerificationAttentionCore(int li,const Layer& L,const LayerGroups& G) {
    // GDN spans share one launch, one grid row per span. Each block reads
    // only its span's arguments, so the arithmetic equals the per-span launches.
    if (!L.attn && verify_spans_.size()>1) {
      PackedStateArgs<GdnArgs,kMaxRows> args{};
      for (size_t i=0;i<verify_spans_.size();++i) {
        const auto& span=verify_spans_[i];
        args.rows[i]=GdnRow(li,L,G,span.rows,span.pos,span.replay,span.commit,*sequences_[span.stream],span.row);
      }
      const dim3 grid(48,unsigned(verify_spans_.size()));
      if (G.state_cache==6) GdnKernel<4,true><<<grid,256>>>(args);
      else if (G.state_cache==4) GdnKernel<4><<<grid,256>>>(args);
      else GdnKernel<0><<<grid,256>>>(args);
      return;
    }
#ifndef CRUCIBLE_QSA_TRACE
    if (L.attn && verify_spans_.size()>1) {
      if (check_packed_qsa_) return CheckPackedQsa(li,L,G);
      return PackedQsaCore(li,L,G);
    }
#endif
    for (const auto& span:verify_spans_)
      AttentionCore(li,L,G,span.rows,span.pos,span.replay,span.commit,*sequences_[span.stream],span.row);
  }
  // QSA spans share each launch of AttentionCore's chain (prep, pool, score, top, split, merge); the span is the
  // grid's z index. Spans keep their own selection path (LaunchQsaTop); the scratch passes stay one per span.
  // Kill test (CRUCIBLE_VERIFY_QSA_CHECK=1; it synchronizes): the packed chain, then the per-span chains from the
  // same inputs, must give the same output rows and selections bit for bit. Prep normalizes q and the indexer
  // query in place, so both are restored; its cache writes depend only on restored or constant inputs.
  // The lookahead kernel runs on its own stream until the pass ends, so the check waits on stream 0 only:
  // a device-wide wait would deadlock.
  const bool check_packed_qsa_=crucible::SwitchText("CRUCIBLE_VERIFY_QSA_CHECK")!=nullptr;
  long qsa_checks_=0,qsa_mismatches_=0;
  void CheckPackedQsa(int li,const Layer& L,const LayerGroups& G) {
    const auto& last=verify_spans_.back();
    const size_t rows=last.row+last.rows;
    __half* xh=G.a_out.jobs[0].xh;
    auto get=[](const auto* src,size_t n) {
      std::vector<std::remove_cv_t<std::remove_pointer_t<decltype(src)>>> v(n);
      HIP_OK(hipMemcpyAsync(v.data(),src,n*sizeof(v[0]),hipMemcpyDeviceToHost,0));
      HIP_OK(hipStreamSynchronize(0));
      return v;
    };
    const auto q=get(q_,rows*kQsaQWidth),idx=get(idx_,rows*kQsaIdxWidth);
    PackedQsaCore(li,L,G);
    const auto out=get(xh,rows*kQsaOutWidth);
    const auto nsel=get(nsel_,rows);
    const auto sel=get(sel_,rows*kSelBlocks);
    HIP_OK(hipMemcpyAsync(q_,q.data(),q.size()*sizeof(float),hipMemcpyHostToDevice,0));
    HIP_OK(hipMemcpyAsync(idx_,idx.data(),idx.size()*sizeof(float),hipMemcpyHostToDevice,0));
    for (const auto& span:verify_spans_)
      AttentionCore(li,L,G,span.rows,span.pos,span.replay,span.commit,*sequences_[span.stream],span.row);
    const auto out1=get(xh,rows*kQsaOutWidth);
    const auto nsel1=get(nsel_,rows);
    const auto sel1=get(sel_,rows*kSelBlocks);
    bool same=std::memcmp(out.data(),out1.data(),out.size()*sizeof(out[0]))==0 && nsel==nsel1;
    for (size_t r=0;same && r<rows;++r)
      if (nsel[r]>0) same=std::equal(sel.begin()+r*kSelBlocks,sel.begin()+r*kSelBlocks+nsel[r],sel1.begin()+r*kSelBlocks);
    ++qsa_checks_;
    if (!same) ++qsa_mismatches_;
    if (!same || qsa_checks_%100==0)
      std::fprintf(stderr,"verify_qsa_check layer=%d spans=%zu rows=%zu checks=%ld mismatches=%ld\n",li,
                   verify_spans_.size(),rows,qsa_checks_,qsa_mismatches_);
  }
  void PackedQsaCore(int li,const Layer& L,const LayerGroups& G) {
    PackedQsaArgs p{},top{},pass{};
    std::vector<QsaArgs> scratch_top;
    const unsigned spans=verify_spans_.size();
    int rows=0,blocks=0,ntop=0,npass=0;
    for (unsigned i=0;i<spans;++i) {
      const auto& span=verify_spans_[i];
      if (span.row+span.rows>kPackedRows) throw std::logic_error("packed QSA scratch holds kPackedRows rows");
      const QsaArgs a=QsaRow(li,L,G,span.rows,span.pos,*sequences_[span.stream],span.row);
      if (QsaRawUsesChunk(a.c.raw_slots,a.R)) throw std::logic_error("packed QSA expects decode raw rows");
      p.rows[i]=a; p.base[i]=span.row;
      rows=std::max(rows,a.R);
      const int nb=(a.pos0+a.R)/4;
      if (nb>kSelBlocks) blocks=std::max(blocks,nb);
      // The same path as LaunchQsaTop for this span alone.
      if (nb>kSelBlocks && nb<=kQsaCandidates) pass.rows[npass++]=a;
      else if (nb>kQsaCandidates && nb<=kQsaTopMaxBlocks) scratch_top.push_back(a);
      else top.rows[ntop++]=a;
    }
    QsaPrepKernel<<<dim3(33,rows,spans),256>>>(p);
    QsaPoolKernel<<<dim3(1,1,spans),256>>>(p);
    if (blocks) QsaScoreKernel<<<dim3((blocks+255)/256,rows,spans),256>>>(p);
    if (ntop) QsaTopKernel<<<dim3(rows,1,ntop),1024>>>(top);
    if (npass) QsaTopPassKernel<0><<<dim3(rows,1,npass),1024>>>(pass,nullptr,nullptr);
    for (const auto& a:scratch_top) LaunchQsaTop(a,qsa_scratch_);
    QsaGqaSplitKernel<<<dim3(2,rows,kQsaSplits*spans),384>>>(p,qsa_scratch_);
    QsaMergeKernel<<<dim3(24,rows,spans),256>>>(p,qsa_scratch_);
    if (!warming_) for (unsigned i=0;i<spans;++i) CaptureQsa(p.rows[i],li);
  }
  void VerificationPle(float* H,int ldh) {
    if (H!=H_ || ldh!=kWide) throw std::logic_error("verification requires native stack layout");
    for (const auto& span:verify_spans_) {
      PleArgs args{kv_+size_t(span.row)*12800,m_.ple.nk1,m_.ple.nq1,m_.ple.nc1,m_.ple.conv,
        sequences_[span.stream]->ple_ring,H_+size_t(span.row)*kWide,span.pos,span.rows};
      PleMixKernel<<<kStreams,512>>>(args);
    }
  }
