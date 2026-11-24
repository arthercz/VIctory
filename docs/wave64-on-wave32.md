# Wave64 console programs on 32-lane host waves

Status: milestones 1-4 implemented (mode selection, entry remap, ballot
exchange, lane reads), validated end-to-end on a 32-lane host (RDNA2) by
`emit::valu_tests::wave64_split_*`. Milestones 5-7 (coherent scalar loads,
multi-wave groups, VS/PS upper lanes) and `ds_bpermute` (unsupported
everywhere today) remain. Tracks the NVIDIA/Intel class of hosts
(`D3D12_FEATURE_DATA_D3D12_OPTIONS1.WaveLaneCountMax < 64`), where a
`[WaveSize(64)]` compute shader is refused at PSO creation and every
wave64 program of the title is dropped today. The reference for the
mechanics is XCL's `docs/wave64-on-wave32.md` and its emitter
(`PRELUDE_X64`, `x64_masks.rs`, `upper_lanes.rs`); this note is the
adaptation to VIctory's IR, where EXEC is already gone.

## Why VIctory's IR is a good fit

`victory_shader::structure` lowers EXEC to per-lane booleans before the
emitter runs, and the emitter keeps scalar control flow scalar: a branch
is one DXIL branch, taken uniformly as long as every scalar value agrees
across the console wave. Scalar values come from scalar code, scalar
memory, and the points where a lane mask becomes data — `mask_dword`'s
`waveActiveBallot`. That is exactly the set of operations the two halves
of a split wave must exchange, and (per XCL's analysis) nothing else can
diverge them: DPP, `ds_swizzle`, `v_permlane(x)16` stay inside a 32-lane
half on the host too, and masks read lane by lane (`v_cndmask` selectors,
carry chains) never become whole-wave values.

## The form

One console wave64 = two hardware wave32s of the same thread group. The
console group keeps its `numthreads` (≤ 64 for the barrier variant, see
below). At entry each hardware wave takes the next 32 console thread
indices from a groupshared counter (`atomicrmw add` on TGSM, XCL's
`xcl_x_thread`), because the host GPU does not promise consecutive
`SV_GroupThreadID`s per wave. Console lane = console thread % 64, half =
bit 5. `thread_id` derives the console thread's coordinates from that
index instead of `flattenedThreadIdInGroup`.

The two halves meet — one DXIL `barrier` — exactly where a value must be
whole-wave:

| operation | exchange |
|---|---|
| `mask_dword` ballot (every mask read as data) | one word per half: each stores its 32-bit ballot in TGSM, barrier, both read both and combine to the 64-bit mask |
| `v_readlane` / `v_readfirstlane` | one word: each half gives lane `l & 31`'s value; the owner half of `l` wins |
| `ds_bpermute` | 32 words per half |
| coherent scalar loads (GLC/DLC `s_load*`) | half 0's values, as XCL does |

Both halves then hold the full 64-bit masks and identical SGPRs; every
branch is taken the same way and the existing scalar lowering
(`s_bcnt1_i32_b64`, `s_ff1`, `s_cbranch_execz`, `v_mbcnt_hi`, …) needs no
change. SGPR pairs of a wave64 program stay 64-bit values built from the
two exchanged ballot words.

## Synchronisation: the one-console-wave-per-group barrier variant first

`GroupMemoryBarrierWithGroupSync` is legal only when the group is exactly
one console wave (all threads of the group are the two halves of the same
console wave, so every barrier is reached by both). Most console compute
is ≤ 64 threads per group, which this variant covers at 1.3–2.2× the
native cost on XCL's measurements — the multi-wave rendezvous with
arrival counters (4.6–9.7×, forward-progress assumptions on WARP) is a
later milestone.

A group whose size is not a multiple of 32 has a half that does not
exist: the missing lanes count as inactive (ballot bits 0), exactly as on
the console.

## TGSM budget

Per group: one 32-bit counter plus a per-lane bank for each exchange
region (a word per lane for ballots and lane reads, two for first-lane
pairs). Every exchanged value is uniform across a hardware wave, so each
lane stores its own slot unguarded — predicated-off lanes store too, no
slot goes stale across barriers or loop iterations, and no zeroing or
double-buffering is needed. The counter keeps the elected-lane claim
from the prelude. Program LDS accesses are offset below the reserved
tail; a program using the whole 32 KB LDS loses its top dwords, as in
XCL.

## Emitter changes, by milestone

1. **Mode selection.** `Parts` gains the host's wave capability (or a
   `split_wave64` flag); `pipeline::host` passes it from
   `D3D12_FEATURE_DATA_D3D12_OPTIONS1` at pipeline creation. The shader
   cache keys on it, so a machine with a 64-lane GPU keeps today's DXIL
   byte-for-byte. `lower`'s `wave = if native.wave64 { 64 }` becomes
   `Some(32)` with the split flag, and `numthreads` is unchanged.
2. **Entry remap.** `shader.rs` reserves the TGSM tail; `thread_id` and
   the compute ids come from the remapped console thread index; the
   groupshared counter is zeroed by one lane before the remap (a second
   barrier, outside all loops).
3. **Ballot exchange.** `mask_dword` emits the store/barrier/combine
   sequence instead of `waveActiveBallot`; the `ballots` cache becomes
   per-site (an exchange is not idempotent across barriers). `MaskData`
   classification (console lanes vs host lanes) is unchanged — a split
   program still has console-lane masks.
4. **Lane reads.** `v_readlane` / `v_readfirstlane` / `ds_bpermute` take
   the exchange form; constant lanes use the lane's own half's base
   (`k | (lane & 32)` in hardware terms is folded into the remap).
5. **Coherent scalar loads.** GLC/DLC `s_load*` exchange through the
   16-word slot.
6. **Multi-wave groups.** The arrival-counter rendezvous for groups of
   more than one console wave (mesh amplification, big compute), plus the
   lane-local mask liveness pass that keeps `v_cmp` + `v_cndmask`
   selectors from being exchanged at all (XCL measured it removing most
   of the remaining overhead).
   Status: the rendezvous is implemented (`split_share_rt`): per console
   wave a publication and an arrival counter per half plus double-buffered
   ballot slots; the parity comes from the publication number (the
   pre-increment counter value), the spin polls atomically (a plain load
   can be hoisted out of the loop by the compiler) and is bounded, with a
   half-local fallback past the bound. A console wave whose upper half the
   group size does not launch skips the rendezvous and reads zeros.
   Verified on an RTX 5060 Ti with a 96-thread group (one full console
   wave exchanging, one 32-thread wave without an upper half): the full
   wave counts its own 48 of 64 lanes, the partial wave its 32 of 32.
   Known issue (precise signature, RTX 5060 Ti): the LAST console wave
   of a group, when it has a full upper half, desyncs deterministically �
   every non-last full wave exchanges correctly, and a partial last wave
   (no upper half) skips correctly. The failing wave's lower half reads
   the partner's slot as 0 while its upper half's spin times out; the
   lower half's own arrival counter is never observed by the partner.
   Ruled out by experiment: elected-lane guards (misfires), lane-count
   arithmetic (XCL's WaveActiveCountBits form), spin bound size, spin
   polling shape (single-lane spin with verdict broadcast, XCL's exact
   form), all-lane store/bump (current form), fence modes (8 and 9), and
   full slot zeroing. Every counter, slot index and fence placement
   checks out statically in the module dump; the asymmetry (one
   direction of the arrival handshake is not observed) violates the
   seq_cst ordering the code establishes. Needs GPU-side tracing
   (PIX/NSight) of the two hardware waves' memory operations.
   The test pins the known values so a fix surfaces here.
   Lane reads (`v_readlane`, `v_readfirstlane`) in multi-wave groups are
   refused for now.
7. **VS/PS.** Graphics stages get no `WaveSize`; on a 32-lane host a
   wave64 VS/PS runs as a partial wave whose lanes 32..63 are missing
   (XCL's `upper_lanes` data-flow: constants and spill slots for those
   lanes, identity values for whole-wave reductions).

## Hazards (from XCL, checked against our IR)

1. **Divergent halves** — anything that lets the two halves compute
   different scalar values desynchronises the barriers. Covered: ballots
   and lane reads exchanged, `s_memtime` already lowered to a constant
   (issue #604's family), non-GLC scalar loads of mutating memory are
   incoherent on the console as well.
2. **Barriers in divergent control flow** — the barrier variant requires
   one console wave per group, whose control flow is identical in both
   halves by hazard 1. Console `s_barrier` stays a group barrier; the
   mesh `no_barrier` analysis is unaffected.
3. **Lane numbering** — handled by the entry remap; never trust
   `flattenedThreadIdInGroup` grouping on the host.
4. **Partial waves** — a group of 32 or 96 threads has a missing half;
   ballot combination ORs in 0 for it.

## Verification plan

Synthetic GFX10 programs, per milestone, in the style of
`emit::lower::tests` and `emit::valu_tests`: a wave64 compute program
whose group is one console wave, asserting the emitted op list (ballot
sites become store/barrier/combine) and, with `emit::host`, the pipeline
creates on a 32-lane adapter and a dispatch agrees with a wave32
reference where the program permits one. GPU comparison tests land with
the existing `emit::host` read-back machinery.
