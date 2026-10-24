// Qwen3.8-Flash-Next model configuration: the layer schedule and the named geometry
// (docs/FLASHNEXT_FORWARD.md sections 1 and 2), shared by the loader and device code.
// Compile-time values only: a sibling model changes this file and rebuilds. There is no runtime
// descriptor and no virtual dispatch, so each name costs nothing over the literal that it
// replaces. tests/test_model_geometry.py keeps new literals out of other files.
// This header is a leaf: it includes nothing, so kernels may include it.
#pragma once
namespace crucible {

// ---- Trunk -------------------------------------------------------------------------------
constexpr char kModelArch[] = "qwen4_exp";  // the HF architecture name
constexpr int kHidden = 2560;
constexpr int kStreams = 4;                // hyper-connection streams
constexpr int kWide = kStreams * kHidden;  // 10240
constexpr int kRank = 320;                 // hyper-connection mixer rank
constexpr int kLayers = 48;
constexpr int kVocab = 248320;
constexpr int kEos = 248044;
constexpr float kEps = 1e-6f;              // every norm epsilon

// ---- Layer schedule ----------------------------------------------------------------------
// Zero-based trunk layers 3, 7, ..., 47 use QSA attention; the others use Gated DeltaNet.
// Layer index kMtpLayer (= kLayers) is the MTP block, a QSA block; the engine stores it at L[kLayers].
enum class LayerKind { kGdn, kQsa };
constexpr int kQsaPeriod = 4;
constexpr int kMtpLayer = kLayers;
constexpr LayerKind Kind(int layer) {
  return layer == kMtpLayer || layer % kQsaPeriod == kQsaPeriod - 1 ? LayerKind::kQsa : LayerKind::kGdn;
}
constexpr bool IsQsa(int layer) { return Kind(layer) == LayerKind::kQsa; }
// PLE runs before zero-based layer 1 (the checkpoint's one-based ple_layer_ids=[2]).
constexpr int kPleLayer = 1;
constexpr bool PleBefore(int layer) { return layer == kPleLayer; }
constexpr char kPleTensor[] = "layers.1.ple.";  // the PLE weights, under the PLE layer
static_assert(kPleTensor[7] == '0' + kPleLayer && kPleTensor[8] == '.');
// Each position gathers one n-gram table row per head; the rows fill the hidden width.
constexpr int kPleHeads = 16;
constexpr int kPleRowDim = kHidden / kPleHeads;  // 160
static_assert(kPleHeads * kPleRowDim == kHidden);

// ---- MoE (every layer) -------------------------------------------------------------------
constexpr int kExperts = 512;
constexpr int kTopK = 10;
constexpr int kSlots = kTopK + 1;  // routed slots + shared expert
constexpr int kMoeInter = 640;

// ---- Gated DeltaNet ----------------------------------------------------------------------
constexpr int kGdnHeads = 48;  // value heads
constexpr int kGdnKeyHeads = 16;
constexpr int kGdnDim = 128;
constexpr int kGdnKeyWidth = kGdnKeyHeads * kGdnDim;           // 2048: q, and k
constexpr int kGdnValueWidth = kGdnHeads * kGdnDim;            // 6144: v, z and the out_proj input
constexpr int kGdnQkvWidth = 2 * kGdnKeyWidth + kGdnValueWidth;  // 10240

// ---- QSA sparse attention ----------------------------------------------------------------
constexpr int kQHeads = 24, kKvHeads = 2, kHeadDim = 256, kIdxHeads = 4, kIdxDim = 128;
constexpr int kRotDim = 64;
constexpr float kRopeTheta = 10000000.f;
constexpr int kSelBlocks = 512;  // selected index blocks
constexpr int kQsaQWidth = kQHeads * 2 * kHeadDim;          // 12288: per head [q | gate]
constexpr int kQsaKvWidth = kKvHeads * kHeadDim;            // 512: k, and v
constexpr int kQsaIdxWidth = (kIdxHeads + 1) * kIdxDim;     // 640: index queries + raw key
constexpr int kQsaOutWidth = kQHeads * kHeadDim;            // 6144: q without gate, context, o_proj input
// Both mixers feed their output projection the same width, so they share its input buffers.
constexpr int kMixerOutWidth = kGdnValueWidth;
static_assert(kQsaOutWidth == kMixerOutWidth);

// ---- Engine limits (not model geometry) ----------------------------------------------------
constexpr int kMaxRows = 8;     // rows of one forward, one verify span and one expert job
constexpr int kPackedRows = 16;  // rows of one packed multi-session verify pass
constexpr float kHadNorm = 0.088388347648f;  // 1/sqrt(128), ExLlamaV3's constant

static_assert(kWide == 10240 && kGdnQkvWidth == 10240 && kGdnValueWidth == 6144);
static_assert(kQsaQWidth == 12288 && kQsaKvWidth == 512 && kQsaIdxWidth == 640 && kQsaOutWidth == 6144);
static_assert(kLayers % kQsaPeriod == 0 && IsQsa(kLayers - 1) && IsQsa(kMtpLayer) && !IsQsa(0) &&
              !PleBefore(0) && PleBefore(1));

}  // namespace crucible
