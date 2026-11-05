// Included inside Engine. Independent rows share projections, never temporal state.
 public:
  void ConfigureServeStreams(int count, uint64_t unused_cache_budget = 0, int short_count = 0, int short_ctx = 0) {
    if (count < 1 || count > kMaxRows) throw std::invalid_argument("batch streams must be in [1,8]");
    if (short_count < 0 || short_count >= count || (short_count && (short_ctx < 1 || short_ctx >= ctx_)))
      throw std::invalid_argument("short streams need 1..streams-1 slots and a short context below --ctx");
    const auto host_available = AvailableHostMemory();
    if (!host_available) throw std::runtime_error("cannot read available host memory; batch admission is disabled");
    size_t device_free, device_total;
    HIP_OK(hipMemGetInfo(&device_free, &device_total));
    uint64_t added = 0, slots_state = 0, allocated = 0, first_slot = 0;
    std::vector<int> contexts;
    for (int slot = 0; slot < count; ++slot) contexts.push_back(SlotContext(slot, count, short_count, short_ctx));
    for (int slot = 0; slot < count; ++slot) {
      const uint64_t bytes = SequenceState::ServeBytes(contexts[size_t(slot)], compact_state_, mtp_rows_);
      slots_state += bytes;
      if (slot < int(sequences_.size())) {
        // An existing compact slot may still lack its MTP cache, so it holds less than its whole state.
        const auto& state = *sequences_[size_t(slot)];
        const uint64_t missing =
            MissingMtpCacheBytes(state.mtp_capacity, state.compact_state, state.qc.contains(kLayers));
        allocated += bytes - missing;
        first_slot = bytes - missing;  // every existing slot is slot 0 here: ConfigureServeStreams runs once
        added += missing;
      } else {
        added += bytes;
      }
    }
    added += VisionReserveBytes(contexts);  // zero without --vision
    AdmissionMemory memory;
    memory.host_free = *host_available;
    memory.device_free = device_free;
    memory.states_bytes = slots_state;
    memory.allocated_bytes = allocated;
    memory.first_slot_bytes = first_slot;
    memory.new_states_bytes = added;
    memory.cache_budget = unused_cache_budget;
    memory.streams = count;
    memory.context = ctx_;
    // A fresh start at a suggested setting, charged by the same code: slot 0 before the check, the other slots, the
    // vision reserve and slot 0's missing MTP cache after it. The user keeps every other option, so a candidate must
    // satisfy the server's rule for short streams (0 < --short-ctx < --ctx and --short-streams below --streams); a
    // short slot keeps kMaxRows rows above --short-ctx, which can be more than the candidate's full context.
    const auto try_setting = [this, short_count, short_ctx](int streams, int context) -> AdmissionCharge {
      AdmissionCharge charge;
      if (streams < 1 || (short_count > 0 && (short_count >= streams || short_ctx < 1 || short_ctx >= context))) {
        charge.allowed = false;
        return charge;
      }
      std::vector<int> slot_contexts;
      for (int slot = 0; slot < streams; ++slot)
        slot_contexts.push_back(slot < streams - short_count ? context : short_ctx + kMaxRows);
      for (const int slot_context : slot_contexts)
        charge.states_bytes += SequenceState::ServeBytes(slot_context, compact_state_, mtp_rows_);
      charge.first_slot_bytes = SequenceState::ServeBytes(slot_contexts[0], compact_state_, mtp_rows_) -
          MissingMtpCacheBytes(SequenceState::MtpCapacity(slot_contexts[0], mtp_rows_), compact_state_, false);
      for (size_t slot = 1; slot < slot_contexts.size(); ++slot)
        charge.new_states_bytes += SequenceState::ServeBytes(slot_contexts[slot], compact_state_, mtp_rows_);
      charge.new_states_bytes += MissingMtpCacheBytes(
          SequenceState::MtpCapacity(slot_contexts[0], mtp_rows_), compact_state_, false);
      charge.new_states_bytes += VisionReserveBytes(slot_contexts);
      return charge;
    };
    const auto error = BatchAdmissionError(memory, try_setting);
    if (!error.empty()) throw std::runtime_error(error);
    // Existing slots keep the full context; short slots are always new, so they come last.
    if (short_count && int(sequences_.size()) > count - short_count)
      throw std::logic_error("short slots must be allocated after the full-context slots");
    ConfigureStreams(count - short_count);
    std::vector<DeviceSpan> spans;
    while (int(sequences_.size()) < count) {
      AddSequence(SlotContext(int(sequences_.size()), count, short_count, short_ctx));
      const auto added_spans = sequences_.back()->Spans();
      spans.insert(spans.end(), added_spans.begin(), added_spans.end());
    }
    const char* warmup = crucible::SwitchText("CRUCIBLE_WARMUP");
    if (!spans.empty() && (!warmup || std::string(warmup) != "0")) WarmDeviceBuffers(spans);
  }

  // Batch serve allocates every compact MTP cache at startup, inside the admitted budget.
  // A lazy promotion at the first speculative request could fail on memory, so that
  // request's result would depend on load. Probes that test lazy promotion skip this call.
  void EnsureServeMtpCaches() {
    for (auto& state : sequences_) EnsureMtpCache(*state, 0);
  }

  int StreamCapacity(int stream) const { return sequences_.at(size_t(stream))->capacity; }
  // Positions of a slot's MTP cache: the bound of every MTP position that a draft step writes.
  int StreamMtpRows(int stream) const { return sequences_.at(size_t(stream))->mtp_capacity; }

  int SlotContext(int slot, int count, int short_count, int short_ctx) const {
    // Short slots keep kMaxRows of headroom: a speculative window may verify rows past the admitted context,
    // and serial generation bounds those rows only by the full context.
    return slot < count - short_count ? ctx_ : short_ctx + kMaxRows;
  }

  void ConfigureStreams(int count) {
    if (count < 1 || count > kMaxRows) {
      std::fprintf(stderr, "stream count must be in [1, %d]\n", kMaxRows);
      std::exit(2);
    }
    const size_t extra = count > (int)sequences_.size() ? count - sequences_.size() : 0;
    size_t free, total;
    HIP_OK(hipMemGetInfo(&free, &total));
    // The measured R=1..8 plan sweep adds about 437 MB after engine construction.
    constexpr size_t kPlanReserveBytes = 512ull << 20;
    if (extra && (free <= kPlanReserveBytes || extra > (free - kPlanReserveBytes) / SequenceState::ServeBytes(ctx_, compact_state_, mtp_rows_))) {
      std::fprintf(stderr, "not enough device memory for stream state; reduce --streams or --ctx\n");
      std::exit(2);
    }
    std::vector<DeviceSpan> added;
    while ((int)sequences_.size() < count) {
      AddSequence();
      const auto spans = sequences_.back()->Spans();
      added.insert(added.end(), spans.begin(), spans.end());
    }
    const char* warmup = crucible::SwitchText("CRUCIBLE_WARMUP");
    if (!added.empty() && (!warmup || std::string(warmup) != "0")) WarmDeviceBuffers(added);
  }

  // The stream of the last SelectStream (batched serve keeps the MTP pair buffers for each stream).
  int ActiveStream() const {
    for (size_t stream = 0; stream < sequences_.size(); ++stream)
      if (sequences_[stream].get() == sequence_) return int(stream);
    return -1;
  }

  void SelectStream(int stream) {
    if (stream < 0 || size_t(stream) >= sequences_.size()) {
      std::fprintf(stderr, "stream ID is out of range\n");
      std::exit(2);
    }
    sequence_ = sequences_[stream].get();
  }

  std::vector<int> ForwardBatch(const std::vector<int>& tokens, const std::vector<int>& streams,
                                const std::string& dump = "") {
    CheckBatch(tokens, streams);
    for (const int stream : streams) MaterializeGdn(*sequences_[stream]);
    const int R = (int)tokens.size();
    const Plans& P = PlansFor(R);
    // Host inputs must live through Head's blocking copy, including a future pinned upload path.
    const BatchHostInputs inputs(tokens, PrepareBatch(tokens, streams));
    // Serial and batched decode run here. Head stops the lookahead; it was never started on this path, so R = 1 HC
    // reads were cold. The prefetch only reads: tokens do not change.
    lookahead_.Start(P.policy);
    host_tokens_.CopyTo(tok_, inputs.tokens().data(), inputs.tokens().size());
    host_rows_.CopyTo(rows_, inputs.rows().data(), inputs.rows().size());
    PleRowsUploaded(host_rows_.data(), (int)inputs.rows().size());
    batch_streams_ = streams;
    profile_scope_ = "batch:";
    Mark("start");
    EmbedKernel<<<dim3(kHidden / 256, R), 256>>>(m_.embed, tok_, H_);
    Mark("embed");
    Dump(dump, 0, R);
    RunTrunk([&](int) { Ple(R, 0); Mark("ple"); },
             [&](int l) { Block(l, m_.L[l], P.L[l], R, 0, 0, R); },
             [&](int l) { Dump(dump, l + 1, R); });
    auto out = Head(m_.final_hc, P, R);
    batch_streams_.clear();
    return out;
  }

 private:
  void CheckBatch(const std::vector<int>& tokens, const std::vector<int>& streams) const {
    std::vector<size_t> lengths;
    std::vector<int> capacities;
    for (const auto& s : sequences_) {
      lengths.push_back(s->hist.size());
      capacities.push_back(s->capacity);
    }
    const std::string error = BatchError(streams, tokens, lengths, capacities, kMaxRows, kVocab);
    if (!error.empty()) {
      std::fprintf(stderr, "%s\n", error.c_str());
      std::exit(2);
    }
    // Validate every prefix on a copy first. A bad stream cannot partially advance its peers.
    for (int s : streams) {
      auto prefix = sequences_[s]->retained;
      const std::string state_error = prefix.Forward(1, 0, 1);
      if (!state_error.empty()) {
        std::fprintf(stderr, "batch requires committed stream state: %s\n", state_error.c_str());
        std::exit(2);
      }
    }
  }

  std::vector<int64_t> PrepareBatch(const std::vector<int>& tokens, const std::vector<int>& streams) {
    SequenceState* saved = sequence_;
    std::vector<int64_t> rows;
    for (size_t r = 0; r < tokens.size(); ++r) {
      sequence_ = sequences_[streams[r]].get();
      const int pos = (int)sequence_->hist.size();
      sequence_->retained.Forward(1, 0, 1);
      sequence_->hist.push_back(tokens[r]);
      const auto one = PleRows(pos, 1);
      rows.insert(rows.end(), one.begin(), one.end());
    }
    sequence_ = saved;
    return rows;
  }

  GdnArgs GdnRow(int li, const Layer& L, const LayerGroups& G, int R, int pos0,
                 int replay, int commit, SequenceState& state, int row) {
    return {qkv_ + (size_t)row * kGdnQkvWidth, z_ + (size_t)row * kGdnValueWidth, x_ + (size_t)row * kHidden,
            L.gdn.ab, L.gdn.conv, L.gdn.A_log, L.gdn.dt_bias, L.gdn.gamma,
            state.ghist.at(li), state.gstate.at(li), L.gdn.out.suh,
            G.a_out.jobs[0].xh + (size_t)row * kGdnValueWidth, state.gpend.at(li), pos0, R, replay, commit};
  }

  void BatchAttentionCore(int li, const Layer& L, const LayerGroups& G, int R) {
    // Independent states use temporal R=1, one launch per request.
    for (int row = 0; row < R; ++row) {
      SequenceState& state = *sequences_[batch_streams_[row]];
      AttentionCore(li, L, G, 1, (int)state.hist.size() - 1, 0, 1, state, row);
    }
  }

  QsaArgs QsaRow(int li, const Layer& L, const LayerGroups& G, int R, int pos0, SequenceState& state, int row) {
    QsaArgs a = QArgs(L.qsa, state.qc.at(li), G.a_out.jobs[0].xh + (size_t)row * kQsaOutWidth, R, pos0, state.Mrope());
    a.q += (size_t)row * kQsaQWidth;
    a.k += (size_t)row * kQsaKvWidth;
    a.v += (size_t)row * kQsaKvWidth;
    a.idx += (size_t)row * kQsaIdxWidth;
    a.scores += (size_t)row * a.sstride;
    a.sel += row * kSelBlocks;
    a.nsel += row;
    return a;
  }

  void AttentionCore(int li, const Layer& L, const LayerGroups& G, int R, int pos0,
                     int replay, int commit, SequenceState& state, int row) {
    if (L.attn) {
      const QsaArgs a = QsaRow(li, L, G, R, pos0, state, row);
#ifdef CRUCIBLE_QSA_TRACE
      QsaTrace(li, 0);
#endif
      // At R = 1 Prep runs the pool and, below 2048 positions, the selection's result; the split attention
      // merges in its own launch (QsaAttnTailKernel).
      const bool tail = R == 1;
      if (tail) {
        QsaPrepKernel<QsaArgs, 0, true><<<dim3(33, 1), 256>>>(a);
      } else {
        QsaPrepKernel<<<dim3(33, R), 256>>>(a);
        QsaPoolKernel<<<1, 256>>>(a);
      }
      LaunchQsaRetainRaw(a);
#ifdef CRUCIBLE_QSA_TRACE
      QsaTrace(li, 1);
#endif
      if (!tail || (pos0 + 1) / 4 > kSelBlocks) LaunchQsaSelect(a, qsa_scratch_);
#ifdef CRUCIBLE_QSA_TRACE
      QsaTrace(li, 2);
#endif
      if (tail) QsaAttnTailKernel<<<dim3(2, 1, kQsaSplits), 384>>>(a, qsa_scratch_, qsa_sync_);
      else LaunchQsaAttn(a, qsa_scratch_);
#ifdef CRUCIBLE_QSA_TRACE
      QsaTrace(li, 3);
#endif
      if (!warming_) CaptureQsa(a, li);
    } else {
      LaunchGdnCore(GdnRow(li, L, G, R, pos0, replay, commit, state, row), G.state_cache);
    }
  }
