// The file formats of the NVMe tier (state_disk.h): the checksum, the byte streams, the stored policy and resume fields, the state file header (version 2:
// an index section and a resume section) and the block file header. Pure functions on host bytes; CPU only.
#pragma once
#include "conversation_resume.h"
#include "sha256.h"
#include "snapshot_bytes.h"
#include "state_blocks.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace crucible {
inline constexpr size_t kStateAlign = 4096;  // O_DIRECT offsets, lengths and buffers
// Version 2: a state file holds the bytes that its block files do not hold.
inline constexpr uint64_t kStateMagic = 0x3254415453435243ull;  // "CRCSTAT2"
inline constexpr uint32_t kStateVersion = 2;
inline constexpr size_t kStateFixedHeader = 56;
inline constexpr uint64_t kStateMaxHeader = uint64_t(256) << 20;

inline std::string StateSha256(const uint8_t* p, size_t n) {
  Sha256 hash;
  hash.Update(p, n);
  return hash.HexDigest();
}
inline uint64_t StateRoundUp(uint64_t n) { return (n + kStateAlign - 1) / kStateAlign * kStateAlign; }

// A fast check against torn or damaged files, not a cryptographic hash: four independent multiply-xor lanes.
inline uint64_t StateChecksum(const uint8_t* p, size_t n) {
  constexpr uint64_t kMul = 0x9E3779B97F4A7C15ull;
  uint64_t lane[4] = {0x243F6A8885A308D3ull, 0x13198A2E03707344ull, 0xA4093822299F31D0ull, 0x082EFA98EC4E6C89ull};
  size_t i = 0;
  for (; i + 32 <= n; i += 32)
    for (int k = 0; k < 4; ++k) {
      uint64_t word;
      std::memcpy(&word, p + i + 8 * k, 8);
      lane[k] = (lane[k] ^ word) * kMul;
      lane[k] ^= lane[k] >> 29;
    }
  uint64_t h = uint64_t(n) * kMul;
  for (const uint64_t v : lane) h = (h ^ v) * kMul, h ^= h >> 31;
  for (; i < n; ++i) h = (h ^ p[i]) * kMul, h ^= h >> 31;
  return h;
}

class StateBytesOut {
 public:
  void U8(uint8_t v) { out_.push_back(v); }
  void U32(uint32_t v) { Raw(&v, 4); }
  void U64(uint64_t v) { Raw(&v, 8); }
  void I32(int32_t v) { Raw(&v, 4); }
  void Str(const std::string& s) { U64(s.size()); Raw(s.data(), s.size()); }
  template<class T> void Vec(const std::vector<T>& v) { U64(v.size()); Raw(v.data(), v.size() * sizeof(T)); }
  void Ints(std::span<const int> v) { U64(v.size()); Raw(v.data(), v.size() * sizeof(int)); }
  void Raw(const void* p, size_t n) {
    if (!n) return;
    const auto* bytes = static_cast<const uint8_t*>(p);
    out_.insert(out_.end(), bytes, bytes + n);
  }
  std::vector<uint8_t>& Bytes() { return out_; }
 private:
  std::vector<uint8_t> out_;
};

class StateBytesIn {
 public:
  StateBytesIn(const uint8_t* p, size_t n) : p_(p), n_(n) {}
  uint8_t U8() { uint8_t v; Raw(&v, 1); return v; }
  uint32_t U32() { uint32_t v; Raw(&v, 4); return v; }
  uint64_t U64() { uint64_t v; Raw(&v, 8); return v; }
  int32_t I32() { int32_t v; Raw(&v, 4); return v; }
  std::string Str() { std::string s(Count(1), '\0'); Raw(s.data(), s.size()); return s; }
  template<class T> std::vector<T> Vec() { std::vector<T> v(Count(sizeof(T))); Raw(v.data(), v.size() * sizeof(T)); return v; }
  void Raw(void* out, size_t n) {
    if (!n) return;
    if (n > n_ - at_) throw std::runtime_error("state header overrun");
    std::memcpy(out, p_ + at_, n);
    at_ += n;
  }
  size_t At() const { return at_; }
 private:
  size_t Count(size_t each) {
    const uint64_t count = U64();
    if (count > (n_ - at_) / each) throw std::runtime_error("state header overrun");
    return size_t(count);
  }
  const uint8_t* p_;
  size_t n_, at_ = 0;
};

// The number of members of an aggregate. A new member of ConversationPolicy or ConversationResume fails the
// static_asserts below until WritePolicy and WriteResume store it: a field left out would let two states that
// differ in it share one file name.
struct StateAnyField { template<class T> operator T() const; };
template<class T, class... A>
constexpr size_t StateFieldCount() {
  if constexpr (requires { T{A{}..., StateAnyField{}}; }) return StateFieldCount<T, A..., StateAnyField>();
  else return sizeof...(A);
}
// Optional members of a later train. Stored when present.
template<class T> constexpr size_t StateKvFields() {
  return size_t(requires(T t) { t.kv_quant; }) + size_t(requires(T t) { t.kv_layout; });
}
template<class T> void WriteKvFields(StateBytesOut& out, const T& t) {
  if constexpr (requires { t.kv_quant; }) out.U64(uint64_t(t.kv_quant));
  if constexpr (requires { t.kv_layout; }) out.U64(uint64_t(t.kv_layout));
}
// The snapshot's own KV format fields, in order (a fake-quant state never restores into real storage).
template<class T> std::vector<uint64_t> SnapshotKvFields(const T& t) {
  std::vector<uint64_t> fields;
  if constexpr (requires { t.kv_quant; }) fields.push_back(uint64_t(t.kv_quant));
  if constexpr (requires { t.kv_layout; }) fields.push_back(uint64_t(t.kv_layout));
  return fields;
}
template<class T> bool SetSnapshotKvFields(const std::vector<uint64_t>& fields, T& t) {
  if (fields.size() != StateKvFields<T>()) return false;
  size_t at = 0;
  if constexpr (requires { t.kv_quant; }) t.kv_quant = decltype(t.kv_quant)(fields[at++]);
  if constexpr (requires { t.kv_layout; }) t.kv_layout = decltype(t.kv_layout)(fields[at++]);
  return true;
}
template<class T> void ReadKvFields(StateBytesIn& in, T& t) {
  if constexpr (requires { t.kv_quant; }) t.kv_quant = decltype(t.kv_quant)(in.U64());
  if constexpr (requires { t.kv_layout; }) t.kv_layout = decltype(t.kv_layout)(in.U64());
}
static_assert(StateFieldCount<ConversationPolicy>() == 16 + StateKvFields<ConversationPolicy>(),
              "a ConversationPolicy member changed: store it in WritePolicy and ReadPolicy (state_disk.h)");
static_assert(StateFieldCount<ConversationResume>() == 9,
              "a ConversationResume member changed: store it in WriteResume and ReadResume (state_disk.h)");

// The tenant first: train 23 wrote the raw tenant just before the policy in every state name and header, so the
// bytes of a state file and its name stay the same (W1).
inline void WritePolicy(StateBytesOut& out, const ConversationPolicy& p) {
  out.Raw(p.tenant.data(), p.tenant.size());
  for (const int v : {p.spec, p.chunk, p.prefill_rows, p.lookup, p.lookup_min, p.mtp_prefix}) out.I32(v);
  for (const bool v : {p.sampled, p.dense, p.adaptive, p.last_head, p.overlap, p.head_auto}) out.U8(v);
  out.U32(p.margin_bits);
  out.U32(p.confidence_bits);
  out.Vec(p.draft_selection);
  WriteKvFields(out, p);
}
inline ConversationPolicy ReadPolicy(StateBytesIn& in) {
  ConversationPolicy p;
  in.Raw(p.tenant.data(), p.tenant.size());
  for (int* v : {&p.spec, &p.chunk, &p.prefill_rows, &p.lookup, &p.lookup_min, &p.mtp_prefix}) *v = in.I32();
  for (bool* v : {&p.sampled, &p.dense, &p.adaptive, &p.last_head, &p.overlap, &p.head_auto}) *v = in.U8() != 0;
  p.margin_bits = in.U32();
  p.confidence_bits = in.U32();
  p.draft_selection = in.Vec<int>();
  ReadKvFields(in, p);
  return p;
}
inline void WriteResume(StateBytesOut& out, const ConversationResume& r) {
  out.U64(r.length);
  for (const int v : {r.mpos, r.replay, r.next_id}) out.I32(v);
  out.U8(r.have_carry);
  out.Vec(r.pending_tokens);
  out.Vec(r.logits);
  out.U8(r.prompt_only);
  out.U64(r.prompt_boundary);
}
inline ConversationResume ReadResume(StateBytesIn& in) {
  ConversationResume r;
  r.length = size_t(in.U64());
  for (int* v : {&r.mpos, &r.replay, &r.next_id}) *v = in.I32();
  r.have_carry = in.U8() != 0;
  r.pending_tokens = in.Vec<int>();
  r.logits = in.Vec<float>();
  r.prompt_only = in.U8() != 0;
  r.prompt_boundary = size_t(in.U64());
  return r;
}

// What the index keeps of one state file: the full key (compared, never only hashed), the resume fields that decide
// whether a request can use it, and the names of its block files. `resume` has no logits.
struct StateMeta {
  std::vector<int> key;
  ConversationPolicy policy;  // with the tenant: one source for the state's name, its blocks and its matches
  bool compact = false, shared = false, guarded = false;
  ConversationResume resume;
  uint64_t payload = 0, file_bytes = 0, used = 0;  // file_bytes: the state file only (each block counts once)
  std::string name;
  std::vector<std::string> blocks;  // block file names, in the state file's order
  bool superseded = false;  // index only: a longer state on disk extends this one (its own branch moved on)
};
// One block of a state file's list: its digest, its size and the snapshot ranges that it fills.
struct StateBlockEntry {
  StateDigest digest{};
  uint64_t bytes = 0;
  std::vector<StreamRange> ranges;
};
// One state file's header, parsed. The index section (Rebuild reads only it) comes first, then the resume section.
struct StateHeader {
  StateMeta meta;
  std::string identity;
  std::vector<std::pair<int, int>> layout;
  std::vector<uint64_t> kv;  // SnapshotKvFields
  uint64_t block_rows = 0;
  std::vector<StateBlockEntry> blocks;
  std::vector<StreamRange> rest;
  // The resume section (only with ParseStateHeader(..., true)): hist, retained fields, logits, rest chunk sums.
  std::vector<int> hist;
  std::array<size_t, 4> retained{};
  ConversationResume resume;  // with logits
  std::vector<uint64_t> sums;  // the rest stream's chunks
  uint64_t header_bytes = 0, index_used = 0;
};
inline void WriteRanges(StateBytesOut& out, std::span<const StreamRange> ranges) {
  out.U64(ranges.size());
  for (const auto& range : ranges) out.U64(range.offset), out.U64(range.bytes);
}
inline std::vector<StreamRange> ReadRanges(StateBytesIn& in) {
  const uint64_t count = in.U64();
  if (count > (uint64_t(1) << 20)) throw std::runtime_error("state header ranges");
  std::vector<StreamRange> ranges(count);
  for (auto& range : ranges) range.offset = in.U64(), range.bytes = in.U64();
  return ranges;
}
inline uint64_t StateChunks(uint64_t bytes) {
  return (bytes + SnapshotChunkPool::kChunkBytes - 1) / SnapshotChunkPool::kChunkBytes;
}
template<class Snapshot>
std::vector<uint8_t> BuildStateHeader(const std::string& identity, const StateMeta& meta, const Snapshot& snapshot,
                                      uint64_t block_rows, const std::vector<StateBlockEntry>& blocks,
                                      std::span<const StreamRange> rest, const std::vector<uint64_t>& sums) {
  StateBytesOut index;
  index.Str(identity);
  WritePolicy(index, meta.policy);
  index.U8(meta.compact);
  index.U8(meta.shared);
  index.U8(meta.guarded);
  index.Ints(meta.key);
  index.U64(snapshot.qsa_layout.size());
  for (const auto& [layer, slots] : snapshot.qsa_layout) index.I32(layer), index.I32(slots);
  index.Vec(SnapshotKvFields(snapshot));
  index.U64(block_rows);
  index.U64(blocks.size());
  for (const auto& block : blocks) {
    index.Raw(block.digest.data(), block.digest.size());
    index.U64(block.bytes);
    WriteRanges(index, block.ranges);
  }
  WriteRanges(index, rest);
  // The resume fields decide whether a request can use the state (Best), so the index holds them, without logits.
  ConversationResume fields = snapshot.resume;
  fields.logits.clear();
  WriteResume(index, fields);
  StateBytesOut resume;
  resume.Ints(snapshot.hist);
  for (const size_t v : snapshot.retained.Fields()) resume.U64(v);
  resume.Vec(snapshot.resume.logits);
  resume.Vec(sums);
  const uint64_t index_used = kStateFixedHeader + index.Bytes().size() + 8;
  const uint64_t used = index_used + resume.Bytes().size() + 8, total = StateRoundUp(used);
  StateBytesOut out;
  out.U64(kStateMagic);
  out.U32(kStateVersion);
  out.U32(0);
  out.U64(total);
  out.U64(used);
  out.U64(snapshot.bytes.size());
  out.U64(SnapshotChunkPool::kChunkBytes);
  out.U64(index_used);
  out.Raw(index.Bytes().data(), index.Bytes().size());
  out.U64(StateChecksum(out.Bytes().data(), out.Bytes().size()));
  out.Raw(resume.Bytes().data(), resume.Bytes().size());
  out.U64(StateChecksum(resume.Bytes().data(), resume.Bytes().size()));
  auto& bytes = out.Bytes();
  bytes.resize(total, 0);
  return std::move(bytes);
}
// The fixed part: header bytes, used bytes, payload, index end. Throws on bounds that no writer gives.
struct StateFixedPart { uint64_t header_bytes, used, payload, index_used; };
inline StateFixedPart ParseStateFixed(const uint8_t* bytes, size_t size, uint64_t magic, uint32_t version) {
  StateBytesIn fixed(bytes, std::min(size, kStateFixedHeader));
  if (fixed.U64() != magic || fixed.U32() != version) throw std::runtime_error("not a state file");
  fixed.U32();
  StateFixedPart part{};
  part.header_bytes = fixed.U64();
  part.used = fixed.U64();
  part.payload = fixed.U64();
  if (fixed.U64() != SnapshotChunkPool::kChunkBytes) throw std::runtime_error("state header chunk size");
  part.index_used = fixed.U64();
  if (part.used < kStateFixedHeader + 8 || part.used > part.header_bytes || part.header_bytes % kStateAlign ||
      part.header_bytes > kStateMaxHeader || part.index_used < kStateFixedHeader + 8 || part.index_used > part.used)
    throw std::runtime_error("state header bounds");
  return part;
}
// Throws on any damage. `bytes` holds at least the index section; with `full`, the whole header.
inline StateHeader ParseStateHeader(const uint8_t* bytes, size_t size, bool full = true) {
  const auto part = ParseStateFixed(bytes, size, kStateMagic, kStateVersion);
  if ((full ? part.header_bytes : part.index_used) > size) throw std::runtime_error("state header bounds");
  StateHeader header;
  header.header_bytes = part.header_bytes;
  header.index_used = part.index_used;
  header.meta.payload = part.payload;
  uint64_t sum;
  std::memcpy(&sum, bytes + part.index_used - 8, 8);
  if (sum != StateChecksum(bytes, part.index_used - 8)) throw std::runtime_error("state header checksum");
  StateBytesIn in(bytes + kStateFixedHeader, part.index_used - 8 - kStateFixedHeader);
  header.identity = in.Str();
  header.meta.policy = ReadPolicy(in);
  header.meta.compact = in.U8() != 0;
  header.meta.shared = in.U8() != 0;
  header.meta.guarded = in.U8() != 0;
  header.meta.key = in.Vec<int>();
  const uint64_t layers = in.U64();
  if (layers > 4096) throw std::runtime_error("state header layout");
  for (uint64_t i = 0; i < layers; ++i) {
    const int layer = in.I32();
    header.layout.emplace_back(layer, in.I32());
  }
  header.kv = in.Vec<uint64_t>();
  header.block_rows = in.U64();
  const uint64_t blocks = in.U64();
  if (blocks > 65536) throw std::runtime_error("state header blocks");
  header.blocks.resize(blocks);
  for (auto& block : header.blocks) {
    in.Raw(block.digest.data(), block.digest.size());
    block.bytes = in.U64();
    block.ranges = ReadRanges(in);
    if (RangeBytes(block.ranges) != block.bytes || !block.bytes) throw std::runtime_error("state header block");
  }
  header.rest = ReadRanges(in);
  header.meta.resume = ReadResume(in);
  if (!header.meta.resume.logits.empty()) throw std::runtime_error("state header fields");
  // Every byte of the snapshot comes from exactly one block range or rest range.
  std::vector<std::span<const StreamRange>> lists{header.rest};
  for (const auto& block : header.blocks) lists.emplace_back(block.ranges);
  if (!RangesCover(lists, header.meta.payload)) throw std::runtime_error("state header cover");
  header.meta.file_bytes = header.header_bytes + StateRoundUp(RangeBytes(header.rest));
  if (!full) return header;
  std::memcpy(&sum, bytes + part.used - 8, 8);
  if (sum != StateChecksum(bytes + part.index_used, part.used - 8 - part.index_used))
    throw std::runtime_error("state header checksum");
  StateBytesIn resume(bytes + part.index_used, part.used - 8 - part.index_used);
  header.hist = resume.Vec<int>();
  for (size_t& v : header.retained) v = size_t(resume.U64());
  header.resume = header.meta.resume;
  header.resume.logits = resume.Vec<float>();
  header.sums = resume.Vec<uint64_t>();
  if (header.sums.size() != StateChunks(RangeBytes(header.rest)) || header.hist.size() != header.meta.key.size())
    throw std::runtime_error("state header fields");
  return header;
}
// A block file: one header (identity, tenant, digest, chunk checksums), then the block stream.
inline constexpr uint64_t kBlockMagic = 0x314B4F4C42435243ull;  // "CRCBLOK1"
inline constexpr uint32_t kBlockVersion = 1;
struct BlockHeader {
  std::string identity;
  StateTenant tenant{};
  StateDigest digest{};
  uint64_t payload = 0, header_bytes = 0;
  std::vector<uint64_t> sums;
  uint64_t FileBytes() const { return header_bytes + StateRoundUp(payload); }
};
inline std::vector<uint8_t> BuildBlockHeader(const BlockHeader& header) {
  StateBytesOut body;
  body.Str(header.identity);
  body.Raw(header.tenant.data(), header.tenant.size());
  body.Raw(header.digest.data(), header.digest.size());
  body.Vec(header.sums);
  const uint64_t used = kStateFixedHeader + body.Bytes().size() + 8, total = StateRoundUp(used);
  StateBytesOut out;
  out.U64(kBlockMagic);
  out.U32(kBlockVersion);
  out.U32(0);
  out.U64(total);
  out.U64(used);
  out.U64(header.payload);
  out.U64(SnapshotChunkPool::kChunkBytes);
  out.U64(used);  // no second section
  out.Raw(body.Bytes().data(), body.Bytes().size());
  out.U64(StateChecksum(out.Bytes().data(), out.Bytes().size()));
  auto& bytes = out.Bytes();
  bytes.resize(total, 0);
  return std::move(bytes);
}
inline BlockHeader ParseBlockHeader(const uint8_t* bytes, size_t size) {
  const auto part = ParseStateFixed(bytes, size, kBlockMagic, kBlockVersion);
  if (part.header_bytes > size || part.index_used != part.used) throw std::runtime_error("block header bounds");
  uint64_t sum;
  std::memcpy(&sum, bytes + part.used - 8, 8);
  if (sum != StateChecksum(bytes, part.used - 8)) throw std::runtime_error("block header checksum");
  BlockHeader header;
  header.header_bytes = part.header_bytes;
  header.payload = part.payload;
  StateBytesIn in(bytes + kStateFixedHeader, part.used - 8 - kStateFixedHeader);
  header.identity = in.Str();
  in.Raw(header.tenant.data(), header.tenant.size());
  in.Raw(header.digest.data(), header.digest.size());
  header.sums = in.Vec<uint64_t>();
  if (header.sums.size() != StateChunks(header.payload) || !header.payload) throw std::runtime_error("block header fields");
  return header;
}
}  // namespace crucible
