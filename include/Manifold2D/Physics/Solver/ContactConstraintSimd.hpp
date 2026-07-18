#pragma once

// ContactConstraintSimd.hpp -- the SoA contact-constraint BATCH + its packer +
// the lane-wide TGS-Soft solve passes.
//
// DATA + PACKER + LANE-WIDE SOLVE. This header owns three things:
//   1. ContactConstraintSimd -- the Structure-of-Arrays staging buffer (one
//      contact per SIMD lane, up to 2 points), the transposed layout the solve
//      gathers/scatters through.
//   2. Build -- the deterministic packer that fills a batch list from a color's
//      emitted `ContactConstraint`s (a pure COPY; it computes no masses/coeffs).
//   3. namespace SimdSolve -- the lane-wide `f32w` load / gather / select / solve
//      / scatter passes (WarmStart / SolveNormalAndFriction / ApplyRestitution +
//      StoreImpulses). SoftStep::Solve drives these per color per substep; the
//      scalar twin for the un-colorable overflow bucket lives in SoftStep.cpp
//      (Overflow{WarmStart/Solve/Restitution}), and the two MUST stay in lockstep.
//
// ----------------------------------------------------------------------------
// ONE CONTACT PER LANE (the Box2D-v3 layout)
// ----------------------------------------------------------------------------
// A batch packs `width` independent contacts -- one per SIMD lane -- where
// `width` is Mosaic::Simd::f32w::width (8 on AVX2, 4 on NEON, 1 forced-scalar).
// A contact carries up to 2 manifold points, so the per-POINT fields are
// arrays-of-2-of-lane-arrays (points[0], points[1]); the per-CONTACT fields
// (normal, friction, masses, ...) are single lane arrays. We pack per-CONTACT
// (not per-point) so a contact's two points solve together in one lane with the
// same body pair -- the point-2 lane simply masks off (pointValid=0) when a
// contact has a single point. Graph coloring (Task 2) guarantees no two contacts
// in a batch share a dynamic body, so all `width` lanes are independent and the
// whole batch solves with no cross-lane read-modify-write hazard.
//
// ----------------------------------------------------------------------------
// PLAIN SCALAR LANE ARRAYS, alignas(32)
// ----------------------------------------------------------------------------
// The fields below are plain `alignas(32) float[width]` (and int[width] for the
// body indices), NOT `f32w`. We build the batch with ordinary scalar stores;
// the solve `load`s each array into an `f32w` once per sub-step. alignas(32)
// makes those loads/stores naturally aligned (a 256-bit AVX2 vector) so the
// solve can use aligned moves. Storing `f32w` directly here would force the SoA
// to carry the active backend's vector type into Core's data layout for no gain.
//
// ----------------------------------------------------------------------------
// FLOAT MASKS (dynB, pointValid), not bool/int
// ----------------------------------------------------------------------------
// `dynB` and `pointValid` are 1.0f / 0.0f FLOAT lane arrays, not a bool/int mask
// type. The solve turns them into a `b32w` predicate via
// `cmp_gt(load(mask), setzero())` and feeds that to `select(...)` -- the Simd
// wrapper's select path consumes a b32w derived from a float compare, so a float
// mask is exactly the shape it wants (and it `load`s with the same aligned move
// as every other lane array; no separate int mask plumbing). dynB gates whether
// B's velocity is mutated (B dynamic) or treated read-only (static/kinematic/
// span); pointValid gates whether a point contributes (a 1-point contact's
// second lane is a masked no-op).
//
// ----------------------------------------------------------------------------
// PADDING IS A LANE-WIDE NO-OP
// ----------------------------------------------------------------------------
// A color's contact count is rarely a multiple of `width`, so the final batch
// has padding lanes. Build masks them to a pure no-op: invMass*/invInertia* = 0
// (immovable -> zero effective impulse), both points invalid + their impulses 0,
// and bodyIndexA/bodyIndexB = kNullBodyIndex (-1): the gather injects the shared
// zero identity row for a -1 lane (no real-body memory touch) and the scatter
// skips it, so a padding lane never reads or writes a real body. The solve
// therefore runs uniformly across all `width` lanes without a per-lane branch,
// and padding contributes nothing to any body.
//
// SCOPE: the struct + Build (a pure packer) + the lane-wide SimdSolve passes
// (further down this file) that consume a batch list. SoftStep::Solve drives the
// SimdSolve passes per color per substep. Build COPIES field values out of the
// source ContactConstraint/...Point (it does NOT compute effective masses or soft
// coefficients -- those are either already filled on the source cc or computed
// lane-wide at solve time).
//
// PRESENTATION-FREE + C++20-clean: std + the Simd width constant + Solver.hpp
// (for ContactConstraint). No SDL3/NVRHI/Batcher2D. namespace Manifold2D::Physics.

#include <Mosaic/Assert.hpp>
#include <cstdint>
#include <vector>

#include <Mosaic/Simd/Wide.hpp>                  // Mosaic::Simd::f32w::width + ops
#include <Manifold2D/Physics/Solver/BodyState.hpp>   // gather/scatter target (T5 solve -> AoS)
#include <Manifold2D/Physics/Solver/Solver.hpp>       // ContactConstraint / ...Point

// Backend intrinsics for the AoS->SoA transpose gather (Task 3, Gap 2.2). Simd.hpp
// already pulls these in for the active backend; re-including under the SAME ladder
// arm keeps the transpose helpers self-documenting. The transpose width is lane-
// count-specific, so it lives here (a physics-SIMD helper) -- NOT as a portable
// Mosaic::Simd primitive.
#if defined(MOSAIC_HAS_AVX2) && !defined(MOSAIC_SIMD_SCALAR)
    #include <immintrin.h>
#elif defined(MOSAIC_HAS_NEON) && !defined(MOSAIC_SIMD_SCALAR)
    #include <arm_neon.h>
#endif

namespace Manifold2D
{
    namespace Physics
    {
        // Null body-index sentinel (Task 3, Gap 2.2). A read-only B (static body
        // or tile span) and every padding lane pack bodyIndexB/bodyIndexA == -1.
        // The gather injects a shared zero identity row for a -1 lane (no real-body
        // memory touch -- Box2D's B2_NULL_INDEX path), and the scatter skips it.
        // This REPLACES the old scatter-safe dummy slot (a one-hot zero tail every
        // read-only-B/padding lane wrote to) -- removing both that slot and the
        // multithreaded write-contention it would create under the persistent
        // solver region (Gap 1).
        inline constexpr std::int32_t kNullBodyIndex = -1;

        // SoA batch of `kWidth` independent contacts -- one contact per lane,
        // up to 2 points per contact. Built scalar by Build(); loaded into f32w
        // at solve time (Task 5). All float lane arrays are alignas(32) for
        // aligned 256-bit loads; body-index arrays are std::int32_t for the i32w
        // gather of body velocities.
        struct ContactConstraintSimd
        {
            // Lane width == the active SIMD float width (8 AVX2 / 4 NEON / 1
            // scalar). A batch holds exactly this many contacts (the tail batch
            // pads the unused lanes -- see `count` + Build's padding path).
            static constexpr int kWidth = static_cast<int>(Mosaic::Simd::f32w::width);

            // Per-POINT lane arrays (a contact has up to 2 of these). Mirrors
            // ContactConstraintPoint, transposed to SoA. pointValid is the float
            // mask (1.0f if this point is live on this lane, else 0.0f).
            struct Point
            {
                alignas(32) float anchorAx[kWidth];        // anchorA.x
                alignas(32) float anchorAy[kWidth];        // anchorA.y
                alignas(32) float anchorBx[kWidth];        // anchorB.x
                alignas(32) float anchorBy[kWidth];        // anchorB.y
                alignas(32) float baseSep[kWidth];         // baseSeparation
                alignas(32) float normalMass[kWidth];      // effective normal mass
                alignas(32) float tangentMass[kWidth];     // effective tangent mass
                alignas(32) float normalImpulse[kWidth];   // accumulated normal impulse
                alignas(32) float tangentImpulse[kWidth];  // accumulated tangent impulse
                alignas(32) float relVel[kWidth];          // relativeVelocity (restitution)
                alignas(32) float pointValid[kWidth];      // float mask: live point?
            };

            // Per-CONTACT lane arrays (transposed ContactConstraint).
            alignas(32) float normalX[kWidth];      // normal.x (B -> A)
            alignas(32) float normalY[kWidth];      // normal.y
            alignas(32) float friction[kWidth];     // combined friction
            alignas(32) float restitution[kWidth];  // combined restitution
            alignas(32) float biasRate[kWidth];     // soft bias rate
            alignas(32) float massScale[kWidth];    // soft mass scale
            alignas(32) float impulseScale[kWidth]; // soft impulse scale
            alignas(32) float invMassA[kWidth];     // body A inverse mass
            alignas(32) float invInertiaA[kWidth];  // body A inverse inertia
            alignas(32) float invMassB[kWidth];     // body B inverse mass
            alignas(32) float invInertiaB[kWidth];  // body B inverse inertia

            // Body index lanes -> i32w gather/scatter of BodyState rows (AoS).
            // A dense solver row (>= 0), or kNullBodyIndex (-1) for a read-only B
            // (static/span) / a padding lane: the gather injects a shared zero
            // identity row for a -1 lane (no real-body memory touch) and the
            // scatter skips it. A is ALWAYS an awake dynamic, so bodyIndexA is
            // never -1 on a LIVE lane (only padding lanes set it to -1).
            alignas(32) std::int32_t bodyIndexA[kWidth];   // A dense row (live), -1 (pad)
            // B dense row, OR -1 (kNullBodyIndex) for a read-only B that has no
            // dense row (static body or tile span) or a padding lane. A kinematic B
            // keeps its REAL dense row (its authored velocity feeds the push term),
            // so it is NOT -1 -- only static/span/padding are.
            alignas(32) std::int32_t bodyIndexB[kWidth];

            // Float mask: 1.0f iff B is a DYNAMIC body whose velocity the solve
            // mutates (bodyBIsBody && invMassB > 0); 0.0f for a static/kinematic
            // body or a tile-span virtual fixture (B is read-only -> no scatter).
            alignas(32) float dynB[kWidth];

            // Two manifold points; a 1-point contact leaves points[1] masked off.
            Point points[2]{};

            // Live lanes in this batch (<= kWidth). The lanes [count, kWidth)
            // are padding (masked no-ops). Lets the consumer know the live prefix
            // without re-deriving it (the solve still runs all kWidth lanes; this
            // is for write-back / inspection).
            int count = 0;

            // ----------------------------------------------------------------
            // Build -- pack `count` contacts (the color's `refs`, indexing into
            // `ccs`) into ceil(count / kWidth) SoA batches.
            // ----------------------------------------------------------------
            //
            // Pure deterministic packer: for lane L of batch B, if the global
            // index g = B*kWidth + L is < count, copy ccs[refs[g]]'s fields into
            // lane L (computing dynB + per-point pointValid, and zeroing a point
            // whose index >= pointCount); otherwise lane L is PADDING -> a
            // lane-wide no-op (zero inv-mass/inertia on both bodies, both points
            // invalid + impulses 0). It COPIES values only -- it does not compute
            // effective masses or soft coefficients.
            //
            // NULL-INDEX BRANCH (Task 3, Gap 2.2 -- replaces the scatter-safe dummy
            // slot). A padding lane, or a read-only-B lane that has no dense row
            // (static body or tile span), packs bodyIndexB == kNullBodyIndex (-1).
            // The lane-wide gather injects a shared zero identity row for a -1 lane
            // (no real-body memory touch) and the scatter skips it, so such a lane
            // never reads or writes a real body's row. This removes the old
            // dummy-tail slot (and the one-hot write contention every read-only-B
            // lane created on it) -- the caller now sizes BodyStateStore to
            // solverCount (NO +1). bodyIndexA is never -1 on a LIVE lane (A is
            // always an awake dynamic in the solver feed); only padding sets it -1.
            //
            // DENSE solverIndex MAPPING (Phase C, Task 2). Build re-homes the packed
            // body indices from WORLD SLOTS onto the solver's DENSE index space:
            //   * A (always an awake dynamic) -> awakeIndex[cc.bodyA] in [0,awakeCnt)
            //   * B dynamic                   -> awakeIndex[cc.bodyB] in [0,awakeCnt)
            //   * B kinematic (real body, invMassB==0, a live kinematic) ->
            //       awakeCount + kinematicIndex[cc.bodyB] in [awakeCnt, solverCnt)
            //   * B static (real body, invMassB==0, NOT kinematic) -> kNullBodyIndex
            //   * B span (bodyBIsBody==false)                       -> kNullBodyIndex
            // The maps are passed as raw per-world-slot pointers (awakeIndex /
            // kinematicIndex, each sized world.Count()) so this header stays free of
            // a PhysicsWorld.hpp dependency. The kinematic-vs-static gate keys off
            // kinematicIndex[bodyB] != kNotKinematic: a kinematic B carries a REAL
            // dense row (its authored velocity drives the relative-velocity push
            // term), a static B reads the zero identity via the null index.
            //
            // BACKWARD-COMPAT IDENTITY MODE. When awakeIndex == nullptr the packer
            // falls back to the pre-Phase-C contract: bodyIndexA = cc.bodyA verbatim,
            // bodyIndexB = cc.bodyB (any real body) or kNullBodyIndex (span). This
            // keeps the packer-contract tests (which pack already-dense small indices
            // and assert bodyIndex == bodyA/bodyB) compiling + passing unchanged.
            //
            // refs may be null iff count == 0.
            static std::vector<ContactConstraintSimd>
            Build(const ContactConstraint* ccs, const std::uint32_t* refs, int count,
                  const std::uint32_t* awakeIndex = nullptr,
                  const std::uint32_t* kinematicIndex = nullptr,
                  std::uint32_t awakeCount = 0u,
                  std::uint32_t kNotKinematic = 0xFFFFFFFFu)
            {
                std::vector<ContactConstraintSimd> batches;
                if (count <= 0)
                {
                    return batches;
                }

                const int batchCount = (count + kWidth - 1) / kWidth;
                batches.resize(static_cast<std::size_t>(batchCount));

                for (int b = 0; b < batchCount; ++b)
                {
                    ContactConstraintSimd& dst = batches[static_cast<std::size_t>(b)];
                    const int base = b * kWidth;
                    const int live = (count - base) < kWidth ? (count - base) : kWidth;
                    dst.count = live;

                    for (int L = 0; L < kWidth; ++L)
                    {
                        const int g = base + L;
                        if (g < count)
                        {
                            PackLane(dst, L, ccs[refs[g]],
                                     awakeIndex, kinematicIndex, awakeCount, kNotKinematic);
                        }
                        else
                        {
                            PadLane(dst, L);
                        }
                    }
                }

                return batches;
            }

        private:
            // Copy one source ContactConstraint into lane L (the live path).
            // A read-only B with no dense row (static/span) packs kNullBodyIndex (the
            // gather injects the zero identity; the scatter skips it).
            // awakeIndex / kinematicIndex (per-world-slot maps, or nullptr for the
            // identity/back-compat mode) + awakeCount + kNotKinematic re-home the
            // packed body indices onto the dense solverIndex space (Phase C, Task 2).
            static void PackLane(ContactConstraintSimd& dst, int L,
                                 const ContactConstraint& cc,
                                 const std::uint32_t* awakeIndex,
                                 const std::uint32_t* kinematicIndex,
                                 std::uint32_t awakeCount,
                                 std::uint32_t kNotKinematic)
            {
                dst.normalX[L]      = static_cast<float>(cc.normal.x);
                dst.normalY[L]      = static_cast<float>(cc.normal.y);
                dst.friction[L]     = static_cast<float>(cc.friction);
                dst.restitution[L]  = static_cast<float>(cc.restitution);
                dst.biasRate[L]     = static_cast<float>(cc.biasRate);
                dst.massScale[L]    = static_cast<float>(cc.massScale);
                dst.impulseScale[L] = static_cast<float>(cc.impulseScale);
                dst.invMassA[L]     = static_cast<float>(cc.invMassA);
                dst.invInertiaA[L]  = static_cast<float>(cc.invInertiaA);
                dst.invMassB[L]     = static_cast<float>(cc.invMassB);
                dst.invInertiaB[L]  = static_cast<float>(cc.invInertiaB);

                // A is ALWAYS an awake dynamic (orientation rule + the awake-A emit
                // gate), so its dense solverIndex is awakeIndex[cc.bodyA]. In the
                // identity/back-compat mode (awakeIndex == nullptr) we pack cc.bodyA
                // verbatim (the pre-Task-2 contract the packer tests pin).
                dst.bodyIndexA[L]   = (awakeIndex != nullptr)
                                          ? static_cast<std::int32_t>(awakeIndex[cc.bodyA])
                                          : static_cast<std::int32_t>(cc.bodyA);

                // dynB: B's velocity is MUTATED (scattered back) only if B is a real
                // dynamic body. A static/kinematic body (invMassB == 0) or a tile
                // span (bodyBIsBody == false) is read-only -> mask off (no scatter).
                const bool dyn = cc.bodyBIsBody && cc.invMassB > Real(0);
                dst.dynB[L]         = dyn ? 1.0f : 0.0f;

                // bodyIndexB is either a real dense row (gather AND/OR scatter a real
                // body) or kNullBodyIndex (-1: gather the zero identity, never
                // scatter). The dense mapping (Phase C, Task 2) splits the read-only-B
                // case in two on the kinematic-vs-static gate -- THE SUBTLEST POINT
                // OF THE TASK:
                //   * B DYNAMIC: dense awake row awakeIndex[cc.bodyB] -- its velocity
                //     is gathered AND scattered (island unity => an awake A's touching
                //     dynamic B is also awake, so it has a real awake row).
                //   * B KINEMATIC (real body, invMassB==0, kinematicIndex != kNot):
                //     its REAL dense kinematic row awakeCount + kinematicIndex[bodyB].
                //     The solver MUST gather its velocity -- a kinematic plate's
                //     authored +X feeds the relative-velocity term that drives the
                //     push (the solve reads vB for any bodyBIsBody endpoint). The
                //     scatter writes back the UNCHANGED gathered value
                //     (select(dynB=0, new, gathered)), so the kinematic row is
                //     preserved -- idempotent even if several lanes share it, and a
                //     kinematic body is never a dynamic A, so it never collides with
                //     an updated A slot. Packing -1 here would LOSE the push (zero vB).
                //   * B STATIC (real body, invMassB==0, kinematicIndex == kNot) OR a
                //     tile span (bodyBIsBody == false): kNullBodyIndex. The gather
                //     injects the zero identity (a static contributes a zero relative
                //     velocity) with NO real-body memory touch, and the scatter skips
                //     the lane. Using a stale kinematicIndex for a static B would
                //     gather garbage -- the kNotKinematic sentinel guards it.
                // In the identity/back-compat mode (awakeIndex == nullptr) we keep the
                // pre-Phase-C contract: cc.bodyB verbatim for any real body,
                // kNullBodyIndex for a span. A is never -1 on a live lane.
                if (awakeIndex == nullptr)
                {
                    dst.bodyIndexB[L] = cc.bodyBIsBody
                                            ? static_cast<std::int32_t>(cc.bodyB)
                                            : kNullBodyIndex;
                }
                else if (dyn)
                {
                    dst.bodyIndexB[L] = static_cast<std::int32_t>(awakeIndex[cc.bodyB]);
                }
                else if (cc.bodyBIsBody && kinematicIndex[cc.bodyB] != kNotKinematic)
                {
                    dst.bodyIndexB[L] = static_cast<std::int32_t>(awakeCount + kinematicIndex[cc.bodyB]);
                }
                else
                {
                    // static or span -> null index (zero identity gather, no scatter).
                    // NOTE: a non-idiomatic moving ZERO-invMass *dynamic* B
                    // (BodyType::Dynamic with invMass==0) is not dynamic here
                    // (dyn==false) and is not in the kinematic set, so it lands HERE
                    // -> the zero identity. Its velocity is NOT gathered and its push
                    // is dropped BY DESIGN; use a Kinematic body for an infinite-mass
                    // mover.
                    dst.bodyIndexB[L] = kNullBodyIndex;
                }

                for (int p = 0; p < 2; ++p)
                {
                    Point& pt = dst.points[p];
                    if (p < cc.pointCount)
                    {
                        const ContactConstraintPoint& cp = cc.points[p];
                        pt.anchorAx[L]       = static_cast<float>(cp.anchorA.x);
                        pt.anchorAy[L]       = static_cast<float>(cp.anchorA.y);
                        pt.anchorBx[L]       = static_cast<float>(cp.anchorB.x);
                        pt.anchorBy[L]       = static_cast<float>(cp.anchorB.y);
                        pt.baseSep[L]        = static_cast<float>(cp.baseSeparation);
                        pt.normalMass[L]     = static_cast<float>(cp.normalMass);
                        pt.tangentMass[L]    = static_cast<float>(cp.tangentMass);
                        pt.normalImpulse[L]  = static_cast<float>(cp.normalImpulse);
                        pt.tangentImpulse[L] = static_cast<float>(cp.tangentImpulse);
                        pt.relVel[L]         = static_cast<float>(cp.relativeVelocity);
                        pt.pointValid[L]     = 1.0f;
                    }
                    else
                    {
                        // Point absent on this contact (pointCount < 2): a masked
                        // no-op lane. Zero its masses + impulses so even if the
                        // mask were ignored the contribution is identically zero.
                        ZeroPointLane(pt, L);
                    }
                }
            }

            // Mask a whole lane to a no-op (a padding lane in the tail batch).
            // Zero inv-mass/inertia (immovable -> zero impulse), both points
            // invalid, BOTH body indices kNullBodyIndex (-1): the gather injects the
            // zero identity (no real-body memory touch) and the scatter skips the
            // lane, so a padding lane never reads or writes a real body.
            static void PadLane(ContactConstraintSimd& dst, int L)
            {
                dst.normalX[L]      = 0.0f;
                dst.normalY[L]      = 0.0f;
                dst.friction[L]     = 0.0f;
                dst.restitution[L]  = 0.0f;
                dst.biasRate[L]     = 0.0f;
                dst.massScale[L]    = 0.0f;
                dst.impulseScale[L] = 0.0f;
                dst.invMassA[L]     = 0.0f;
                dst.invInertiaA[L]  = 0.0f;
                dst.invMassB[L]     = 0.0f;
                dst.invInertiaB[L]  = 0.0f;
                dst.bodyIndexA[L]   = kNullBodyIndex;
                dst.bodyIndexB[L]   = kNullBodyIndex;
                dst.dynB[L]         = 0.0f;
                for (int p = 0; p < 2; ++p)
                {
                    ZeroPointLane(dst.points[p], L);
                }
            }

            // Zero one point's lane: invalidate it + null its masses/impulses so
            // it cannot contribute (the shared "absent / padding point" reset).
            static void ZeroPointLane(Point& pt, int L)
            {
                pt.anchorAx[L]       = 0.0f;
                pt.anchorAy[L]       = 0.0f;
                pt.anchorBx[L]       = 0.0f;
                pt.anchorBy[L]       = 0.0f;
                pt.baseSep[L]        = 0.0f;
                pt.normalMass[L]     = 0.0f;
                pt.tangentMass[L]    = 0.0f;
                pt.normalImpulse[L]  = 0.0f;
                pt.tangentImpulse[L] = 0.0f;
                pt.relVel[L]         = 0.0f;
                pt.pointValid[L]     = 0.0f;
            }
        };

        // ====================================================================
        // Lane-wide contact solve passes (Task 5) -- the TGS-Soft math ported
        // from SoftStep.cpp, per lane, over Mosaic::Simd.
        // ====================================================================
        //
        // Each pass takes the per-color batch vector + the solver's BodyStateStore
        // (AoS rows, accessed via .data()) and processes ALL kWidth lanes of every
        // batch (padding/read-only-B lanes are masked no-ops by construction -- zero
        // inv-mass and the kNullBodyIndex null-index branch). Graph coloring
        // guarantees no two LIVE lanes in a batch share a dynamic body, so the
        // gather -> math -> scatter is hazard-free; a -1 (null) lane gathers the
        // shared zero identity row and is never scattered (see Build's NULL-INDEX
        // BRANCH note).
        //
        // LOCKSTEP: this SimdSolve trio (WarmStart / SolveNormalAndFriction /
        // ApplyRestitution) must match the scalar overflow trio Overflow{WarmStart/
        // Solve/Restitution} in SoftStep.cpp lane-for-scalar. The colored batches and
        // the overflow constraints share ONE BodyStateStore and compose into ONE
        // velocity store within a single substep, so the lane-wide math here and the
        // width-1 sequential overflow math MUST stay numerically identical (they are
        // the same TGS normal+friction+restitution step at two widths). The plain-
        // float ScalarRef oracle in PhysicsSimdSolverTest.cpp is the third copy and
        // the bit-match gate -- change one, change all three.
        //
        // The math mirrors the scalar SoftStep TGS step EXACTLY (per lane):
        //   WarmStart        <-> SoftStep.cpp Overflow{WarmStart} normal+tangent apply
        //   SolveNormalAndFriction(useBias) <-> Overflow{Solve} (normal solve w/ the
        //                      s>0 / penetration / relax bias branch + the friction
        //                      Coulomb-cone clamp + the per-substep separation re-eval)
        //   ApplyRestitution <-> Overflow{Restitution} (rebound for points that
        //                      approached fast enough AND took a normal impulse)
        // 2D cross helpers inline: CrossWR(w,r) = (-w*r.y, w*r.x);
        // CrossRP(r,p) = r.x*p.y - r.y*p.x; Dot(a,b) = a.x*b.x + a.y*b.y.
        //
        // FP NOTE: the wrapper's plain `* + -` operators round per-op (only
        // mul_add/mul_sub fuse under /fp:strict). The scalar SoftStep.cpp uses
        // plain `*`/`+` everywhere (no std::fma), so to BIT-MATCH the scalar oracle
        // these passes ALSO use plain operators -- NOT mul_add -- for the impulse
        // accumulation. (mul_add would fuse and diverge from the scalar reference,
        // breaking lane-width invariance against the forced-scalar oracle.) The
        // scalar backend's plain operators are exactly C++ float arithmetic, so the
        // 1-wide path reproduces SoftStep.cpp's scalar result bit-for-bit, and the
        // 8-wide path differs only by the same per-lane float rounding the scalar
        // path has -> lane-width invariance holds to <1e-5.

        namespace SimdSolve
        {
            using namespace Mosaic::Simd;

            // Turn a 1.0f/0.0f float mask lane array into a b32w predicate.
            MOSAIC_FORCEINLINE b32w MaskOf(const float* maskArr) noexcept
            {
                return cmp_gt(load(maskArr), setzero());
            }

            // Gathered body-state from the AoS store: one f32w per field.
            struct BodyStateW { f32w vx, vy, w, dpx, dpy, dq; };
            // Velocity-only gather (WarmStart + ApplyRestitution read no dp/dq).
            struct BodyVelW { f32w vx, vy, w; };

            // Shared zero IDENTITY row a null (-1) lane gathers, instead of touching
            // a real body's memory (Box2D's B2_NULL_INDEX path). 32-byte aligned by
            // BodyState's alignas(32), so the AVX2 aligned row load over it is valid.
            // `inline constexpr` -> one object across TUs; address taken at runtime.
            inline constexpr BodyState kIdentityRow{};

            // -----------------------------------------------------------------
            // AoS->SoA TRANSPOSE GATHER (Task 3, Gap 2.2 -- the single-thread win)
            // -----------------------------------------------------------------
            // Box2D's b2GatherBodies model: load each lane's WHOLE 32-byte body row
            // with an aligned vector load (prefetchable, one cache line per 2 bodies),
            // then transpose AoS rows -> SoA field vectors. This replaces the prior
            // naive per-lane scalar field copy (six dependent scalar reads per body)
            // and the old hardware-gather plan. A -1 (null) lane selects the shared
            // zero identity row BEFORE the load, so it never reads states[-1].
            //
            // The transpose only REORDERS bits (it computes nothing), so the gathered
            // lane values are EXACTLY the stored floats -> byte-identical to the naive
            // gather and lane-width invariance holds. The transpose width is lane-
            // count-specific, so it is a backend #if here (NOT a portable Simd op).
            // The scalar (oracle) + NEON + fallback paths use a plain per-lane read.

            // Per-lane row pointers with the null-index identity select. Fills p[W]
            // (idx >= 0 -> &states[idx]; -1 -> &kIdentityRow).
            MOSAIC_FORCEINLINE void GatherRowPtrs(const BodyState* states, i32w idx,
                                                  const BodyState** p) noexcept
            {
                constexpr int W = ContactConstraintSimd::kWidth;
                alignas(32) std::int32_t ix[W];
                istore(ix, idx);
                for (int L = 0; L < W; ++L)
                {
                    p[L] = (ix[L] >= 0) ? (states + ix[L]) : &kIdentityRow;
                }
            }

            // Full gather (vx,vy,w,dpx,dpy,dq) -- feeds SolveNormalAndFriction.
            MOSAIC_FORCEINLINE BodyStateW GatherBodies(const BodyState* states, i32w idx) noexcept
            {
                constexpr int W = ContactConstraintSimd::kWidth;
                const BodyState* p[W];
                GatherRowPtrs(states, idx, p);

#if defined(MOSAIC_HAS_AVX2) && !defined(MOSAIC_SIMD_SCALAR)
                // Eight aligned 256-bit row loads, then an 8x8 AoS->SoA transpose
                // (_mm256_unpacklo/hi_ps + _mm256_shuffle_ps + _mm256_permute2f128_ps).
                // Lane 0 is representative: other lanes are 32-aligned by construction
                // (states[] is alignas(32) stride-32; kIdentityRow is alignas(32)).
                MOSAIC_ASSERT((reinterpret_cast<std::uintptr_t>(p[0]) & 31u) == 0u,
                              "AVX2 gather: row pointer must be 32-byte aligned");
                const __m256 r0 = _mm256_load_ps(reinterpret_cast<const float*>(p[0]));
                const __m256 r1 = _mm256_load_ps(reinterpret_cast<const float*>(p[1]));
                const __m256 r2 = _mm256_load_ps(reinterpret_cast<const float*>(p[2]));
                const __m256 r3 = _mm256_load_ps(reinterpret_cast<const float*>(p[3]));
                const __m256 r4 = _mm256_load_ps(reinterpret_cast<const float*>(p[4]));
                const __m256 r5 = _mm256_load_ps(reinterpret_cast<const float*>(p[5]));
                const __m256 r6 = _mm256_load_ps(reinterpret_cast<const float*>(p[6]));
                const __m256 r7 = _mm256_load_ps(reinterpret_cast<const float*>(p[7]));

                const __m256 t0 = _mm256_unpacklo_ps(r0, r1);
                const __m256 t1 = _mm256_unpackhi_ps(r0, r1);
                const __m256 t2 = _mm256_unpacklo_ps(r2, r3);
                const __m256 t3 = _mm256_unpackhi_ps(r2, r3);
                const __m256 t4 = _mm256_unpacklo_ps(r4, r5);
                const __m256 t5 = _mm256_unpackhi_ps(r4, r5);
                const __m256 t6 = _mm256_unpacklo_ps(r6, r7);
                const __m256 t7 = _mm256_unpackhi_ps(r6, r7);

                const __m256 s0 = _mm256_shuffle_ps(t0, t2, 0x44);
                const __m256 s1 = _mm256_shuffle_ps(t0, t2, 0xEE);
                const __m256 s2 = _mm256_shuffle_ps(t1, t3, 0x44);
                const __m256 s3 = _mm256_shuffle_ps(t1, t3, 0xEE);
                const __m256 s4 = _mm256_shuffle_ps(t4, t6, 0x44);
                const __m256 s5 = _mm256_shuffle_ps(t4, t6, 0xEE);
                const __m256 s6 = _mm256_shuffle_ps(t5, t7, 0x44);
                const __m256 s7 = _mm256_shuffle_ps(t5, t7, 0xEE);

                BodyStateW r;
                r.vx  = f32w{ _mm256_permute2f128_ps(s0, s4, 0x20) }; // col0 = vx
                r.vy  = f32w{ _mm256_permute2f128_ps(s1, s5, 0x20) }; // col1 = vy
                r.w   = f32w{ _mm256_permute2f128_ps(s2, s6, 0x20) }; // col2 = w
                r.dpx = f32w{ _mm256_permute2f128_ps(s3, s7, 0x20) }; // col3 = dpx
                r.dpy = f32w{ _mm256_permute2f128_ps(s0, s4, 0x31) }; // col4 = dpy
                r.dq  = f32w{ _mm256_permute2f128_ps(s1, s5, 0x31) }; // col5 = dq
                return r;
#else
                // Scalar oracle (W==1) + NEON + fallback: plain per-lane read. A pure
                // reorder -> bit-identical to the AVX2 transpose and to the stored
                // floats (the scalar path IS the lane-width-invariance oracle).
                alignas(32) float tvx[W], tvy[W], tw[W], tdpx[W], tdpy[W], tdq[W];
                for (int L = 0; L < W; ++L)
                {
                    const BodyState& s = *p[L];
                    tvx[L]  = s.vx;  tvy[L]  = s.vy;  tw[L]  = s.w;
                    tdpx[L] = s.dpx; tdpy[L] = s.dpy; tdq[L] = s.dq;
                }
                BodyStateW r;
                r.vx  = load(tvx);  r.vy  = load(tvy);  r.w  = load(tw);
                r.dpx = load(tdpx); r.dpy = load(tdpy); r.dq = load(tdq);
                return r;
#endif
            }

            // Velocity-only gather (vx,vy,w) -- the WarmStart/ApplyRestitution passes
            // read no dp/dq, so this loads only each row's low 128 bits and uses a
            // reduced transpose (fewer shuffles, half the row bytes). Same null-index
            // identity select; same byte-identical pure-reorder guarantee.
            MOSAIC_FORCEINLINE BodyVelW GatherVel(const BodyState* states, i32w idx) noexcept
            {
                constexpr int W = ContactConstraintSimd::kWidth;
                const BodyState* p[W];
                GatherRowPtrs(states, idx, p);

#if defined(MOSAIC_HAS_AVX2) && !defined(MOSAIC_SIMD_SCALAR)
                // Load the low __m128 (vx,vy,w,dpx) of each row; build 256-bit lane
                // pairs (lanes 0-3 | 4-7), then unpack + shuffle to vx/vy/w. No
                // permute2f128 needed (insertf128 already placed the high lanes).
                // Lane 0 is representative: other lanes are 32-aligned by construction
                // (states[] is alignas(32) stride-32; kIdentityRow is alignas(32)),
                // so all lanes satisfy the 16-byte requirement checked here.
                MOSAIC_ASSERT((reinterpret_cast<std::uintptr_t>(p[0]) & 15u) == 0u,
                              "SSE gather: row pointer must be 16-byte aligned");
                const __m128 m0 = _mm_load_ps(reinterpret_cast<const float*>(p[0]));
                const __m128 m1 = _mm_load_ps(reinterpret_cast<const float*>(p[1]));
                const __m128 m2 = _mm_load_ps(reinterpret_cast<const float*>(p[2]));
                const __m128 m3 = _mm_load_ps(reinterpret_cast<const float*>(p[3]));
                const __m128 m4 = _mm_load_ps(reinterpret_cast<const float*>(p[4]));
                const __m128 m5 = _mm_load_ps(reinterpret_cast<const float*>(p[5]));
                const __m128 m6 = _mm_load_ps(reinterpret_cast<const float*>(p[6]));
                const __m128 m7 = _mm_load_ps(reinterpret_cast<const float*>(p[7]));

                const __m256 a04 = _mm256_insertf128_ps(_mm256_castps128_ps256(m0), m4, 1);
                const __m256 a15 = _mm256_insertf128_ps(_mm256_castps128_ps256(m1), m5, 1);
                const __m256 a26 = _mm256_insertf128_ps(_mm256_castps128_ps256(m2), m6, 1);
                const __m256 a37 = _mm256_insertf128_ps(_mm256_castps128_ps256(m3), m7, 1);

                const __m256 u0 = _mm256_unpacklo_ps(a04, a15); // [vx,vx,vy,vy | ...]
                const __m256 u1 = _mm256_unpackhi_ps(a04, a15); // [w,w,dpx,dpx | ...]
                const __m256 u2 = _mm256_unpacklo_ps(a26, a37);
                const __m256 u3 = _mm256_unpackhi_ps(a26, a37);

                BodyVelW r;
                r.vx = f32w{ _mm256_shuffle_ps(u0, u2, 0x44) }; // vx[0..7]
                r.vy = f32w{ _mm256_shuffle_ps(u0, u2, 0xEE) }; // vy[0..7]
                r.w  = f32w{ _mm256_shuffle_ps(u1, u3, 0x44) }; // w[0..7]
                return r;
#else
                alignas(32) float tvx[W], tvy[W], tw[W];
                for (int L = 0; L < W; ++L)
                {
                    const BodyState& s = *p[L];
                    tvx[L] = s.vx; tvy[L] = s.vy; tw[L] = s.w;
                }
                return BodyVelW{ load(tvx), load(tvy), load(tw) };
#endif
            }

            // Scatter velocity (vx, vy, w) back to the AoS rows. Only writes a lane
            // where the write mask is true AND the index is non-null (skip-on-false:
            // a non-dynamic B, or a -1 null lane, never updates a real row). We write
            // only the three velocity fields per row (NOT a whole-row store) so the
            // body's dp/dq -- integrated separately -- are left untouched. Uses a
            // store-to-temp / scalar write-back (AVX2 has no scatter instruction; and
            // a per-field masked write is the natural form for partial-row writes).
            MOSAIC_FORCEINLINE void ScatterVel(BodyState* states, i32w idx,
                                               f32w vx, f32w vy, f32w w, b32w write) noexcept
            {
                constexpr int W = ContactConstraintSimd::kWidth;
                alignas(32) std::int32_t idxBuf[W];
                istore(idxBuf, idx);
                alignas(32) float tvx[W], tvy[W], tw_[W], wm[W];
                store(tvx, vx); store(tvy, vy); store(tw_, w);
                store(wm, select(write, splat(1.0f), setzero()));
                for (int L = 0; L < W; ++L)
                {
                    if (wm[L] != 0.0f && idxBuf[L] >= 0)
                    {
                        states[idxBuf[L]].vx = tvx[L];
                        states[idxBuf[L]].vy = tvy[L];
                        states[idxBuf[L]].w  = tw_[L];
                    }
                }
            }

            // ---- WarmStart: apply accumulated n*nImp + t*tImp to body vels -----
            // Lane-wide TGS warm-start; the scalar twin is SoftStep::OverflowWarmStart
            // (SoftStep.cpp). LOCKSTEP -- keep identical lane-for-scalar.

            // Range overload: process batches [begin, end). Loop body unchanged.
            inline void WarmStart(std::vector<ContactConstraintSimd>& batches,
                                  BodyState* states,
                                  std::size_t begin, std::size_t end)
            {
                const b32w allTrue = cmp_gt(splat(1.0f), setzero());
                for (std::size_t i = begin; i < end; ++i)
                {
                    ContactConstraintSimd& cc = batches[i];
                    const i32w ia = iload(cc.bodyIndexA);
                    const i32w ib = iload(cc.bodyIndexB);
                    const b32w dyn = MaskOf(cc.dynB);

                    const f32w nx = load(cc.normalX);
                    const f32w ny = load(cc.normalY);
                    const f32w tx = -ny;            // tangent = (-n.y, n.x)
                    const f32w ty = nx;

                    const f32w iMa = load(cc.invMassA);
                    const f32w iIa = load(cc.invInertiaA);
                    const f32w iMb = load(cc.invMassB);
                    const f32w iIb = load(cc.invInertiaB);

                    // WarmStart consumes velocity only (no dp/dq) -> velocity gather.
                    const BodyVelW sA = GatherVel(states, ia);
                    const BodyVelW sB = GatherVel(states, ib);
                    f32w vAx = sA.vx, vAy = sA.vy, wA = sA.w;
                    f32w vBx = sB.vx, vBy = sB.vy, wB = sB.w;

                    for (int p = 0; p < 2; ++p)
                    {
                        const ContactConstraintSimd::Point& pt = cc.points[p];
                        const f32w nImp = load(pt.normalImpulse);
                        const f32w tImp = load(pt.tangentImpulse);
                        const f32w rAx = load(pt.anchorAx), rAy = load(pt.anchorAy);
                        const f32w rBx = load(pt.anchorBx), rBy = load(pt.anchorBy);

                        // P = n*nImp + tangent*tImp (an inactive point has both
                        // impulses 0 -> contributes nothing, no mask needed).
                        const f32w Px = nx * nImp + tx * tImp;
                        const f32w Py = ny * nImp + ty * tImp;

                        // A += P*iMa ; wA += iIa*(rA x P)
                        vAx = vAx + Px * iMa;
                        vAy = vAy + Py * iMa;
                        wA  = wA  + iIa * (rAx * Py - rAy * Px);
                        // B -= P*iMb ; wB -= iIb*(rB x P)  (gated by dynB)
                        vBx = vBx - Px * iMb;
                        vBy = vBy - Py * iMb;
                        wB  = wB  - iIb * (rBx * Py - rBy * Px);
                    }

                    // A is always dynamic (unconditional scatter); B only where dynB.
                    ScatterVel(states, ia, vAx, vAy, wA, allTrue);
                    ScatterVel(states, ib, vBx, vBy, wB, dyn);
                }
            }
            inline void WarmStart(std::vector<ContactConstraintSimd>& batches,
                                  BodyState* states)
            {
                WarmStart(batches, states, 0, batches.size());
            }

            // ---- Normal + friction solve --------------------------------------
            // Lane-wide TGS normal+friction; the scalar twin is SoftStep::OverflowSolve
            // (SoftStep.cpp). LOCKSTEP -- keep identical lane-for-scalar.
            // invH = 1/h, maxBiasVel = world.ContactPushMaxVelocity().

            // Range overload: process batches [begin, end). Loop body unchanged.
            inline void SolveNormalAndFriction(std::vector<ContactConstraintSimd>& batches,
                                               BodyState* states, float h, bool useBias,
                                               float maxBiasVel,
                                               std::size_t begin, std::size_t end)
            {
                const f32w invH = (h > 0.0f) ? splat(1.0f / h) : setzero();
                const f32w vMaxNeg = splat(-maxBiasVel);
                const f32w one = splat(1.0f);
                const f32w zero = setzero();
                const b32w allTrue = cmp_gt(splat(1.0f), setzero());

                for (std::size_t i = begin; i < end; ++i)
                {
                    ContactConstraintSimd& cc = batches[i];
                    const i32w ia = iload(cc.bodyIndexA);
                    const i32w ib = iload(cc.bodyIndexB);
                    const b32w dyn = MaskOf(cc.dynB);

                    const f32w nx = load(cc.normalX);
                    const f32w ny = load(cc.normalY);
                    const f32w tx = -ny;
                    const f32w ty = nx;

                    const f32w iMa = load(cc.invMassA);
                    const f32w iIa = load(cc.invInertiaA);
                    const f32w iMb = load(cc.invMassB);
                    const f32w iIb = load(cc.invInertiaB);

                    const f32w ccBias = load(cc.biasRate);
                    const f32w ccMassScale = load(cc.massScale);
                    const f32w ccImpScale = load(cc.impulseScale);
                    const f32w ccFriction = load(cc.friction);

                    const BodyStateW sA = GatherBodies(states, ia);
                    const BodyStateW sB = GatherBodies(states, ib);
                    f32w vAx = sA.vx, vAy = sA.vy, wA = sA.w;
                    f32w vBx = sB.vx, vBy = sB.vy, wB = sB.w;

                    const f32w dpAx = sA.dpx, dpAy = sA.dpy, drA = sA.dq;
                    const f32w dpBx = sB.dpx, dpBy = sB.dpy, drB = sB.dq;

                    // ---- normal solve (per point) -----------------------------
                    for (int p = 0; p < 2; ++p)
                    {
                        ContactConstraintSimd::Point& pt = cc.points[p];
                        const f32w rAx = load(pt.anchorAx), rAy = load(pt.anchorAy);
                        const f32w rBx = load(pt.anchorBx), rBy = load(pt.anchorBy);
                        const f32w baseSep = load(pt.baseSep);
                        const f32w nMass = load(pt.normalMass);
                        f32w nImp = load(pt.normalImpulse);

                        // s = baseSep + dot((dpA + drA x rA) - (dpB + drB x rB), n).
                        // CrossWR(dr, r) = (-dr*r.y, dr*r.x).
                        const f32w prAx = dpAx + (-drA * rAy);
                        const f32w prAy = dpAy + ( drA * rAx);
                        const f32w prBx = dpBx + (-drB * rBy);
                        const f32w prBy = dpBy + ( drB * rBx);
                        const f32w s = baseSep + ((prAx - prBx) * nx + (prAy - prBy) * ny);

                        // bias / massScale / impulseScale branch (SoftStep:383-400).
                        //   s > 0  : speculative gap -> bias = s*invH, scale 1/0
                        //   s <= 0 & useBias : bias = max(ccBias*s, -maxBiasVel),
                        //                      massScale/impulseScale from contact
                        //   s <= 0 & !useBias: bias 0, scale 1/0 (relax)
                        const b32w gap = cmp_gt(s, zero);
                        const f32w biasGap = s * invH;
                        const f32w biasPen = max(ccBias * s, vMaxNeg);
                        f32w bias, massScale, impScale;
                        if (useBias)
                        {
                            bias      = select(gap, biasGap, biasPen);
                            massScale = select(gap, one, ccMassScale);
                            impScale  = select(gap, zero, ccImpScale);
                        }
                        else
                        {
                            // Relax pass: penetration branch is bias 0, scale 1/0.
                            bias      = select(gap, biasGap, zero);
                            massScale = one;
                            impScale  = zero;
                        }

                        // vn = dot((vA + wA x rA) - (vB + wB x rB), n).
                        const f32w dvx = (vAx + (-wA * rAy)) - (vBx + (-wB * rBy));
                        const f32w dvy = (vAy + ( wA * rAx)) - (vBy + ( wB * rBx));
                        const f32w vn = dvx * nx + dvy * ny;

                        // impulse = -nMass*massScale*(vn+bias) - impScale*nImp.
                        f32w impulse = -nMass * massScale * (vn + bias) - impScale * nImp;
                        // accumulated clamp >= 0.
                        const f32w newI = max(nImp + impulse, zero);
                        impulse = newI - nImp;
                        nImp = newI;
                        store(pt.normalImpulse, nImp);

                        // P = n*impulse ; apply to A (+) and B (-, gated).
                        const f32w Px = nx * impulse, Py = ny * impulse;
                        vAx = vAx + Px * iMa;
                        vAy = vAy + Py * iMa;
                        wA  = wA  + iIa * (rAx * Py - rAy * Px);
                        vBx = vBx - Px * iMb;
                        vBy = vBy - Py * iMb;
                        wB  = wB  - iIb * (rBx * Py - rBy * Px);
                    }

                    // ---- friction solve (per point) ---------------------------
                    for (int p = 0; p < 2; ++p)
                    {
                        ContactConstraintSimd::Point& pt = cc.points[p];
                        const f32w rAx = load(pt.anchorAx), rAy = load(pt.anchorAy);
                        const f32w rBx = load(pt.anchorBx), rBy = load(pt.anchorBy);
                        const f32w tMass = load(pt.tangentMass);
                        const f32w nImp = load(pt.normalImpulse);
                        f32w tImp = load(pt.tangentImpulse);

                        const f32w dvx = (vAx + (-wA * rAy)) - (vBx + (-wB * rBy));
                        const f32w dvy = (vAy + ( wA * rAx)) - (vBy + ( wB * rBx));
                        const f32w vt = dvx * tx + dvy * ty;

                        f32w impulse = -tMass * vt;

                        // clamp to [-friction*nImp, +friction*nImp] (Coulomb cone).
                        const f32w maxFric = ccFriction * nImp;
                        const f32w cand = tImp + impulse;
                        const f32w newI = max(-maxFric, min(cand, maxFric));
                        impulse = newI - tImp;
                        tImp = newI;
                        store(pt.tangentImpulse, tImp);

                        const f32w Px = tx * impulse, Py = ty * impulse;
                        vAx = vAx + Px * iMa;
                        vAy = vAy + Py * iMa;
                        wA  = wA  + iIa * (rAx * Py - rAy * Px);
                        vBx = vBx - Px * iMb;
                        vBy = vBy - Py * iMb;
                        wB  = wB  - iIb * (rBx * Py - rBy * Px);
                    }

                    // A is always dynamic (unconditional scatter); B only where dynB.
                    ScatterVel(states, ia, vAx, vAy, wA, allTrue);
                    ScatterVel(states, ib, vBx, vBy, wB, dyn);
                }
            }
            inline void SolveNormalAndFriction(std::vector<ContactConstraintSimd>& batches,
                                               BodyState* states, float h, bool useBias,
                                               float maxBiasVel)
            {
                SolveNormalAndFriction(batches, states, h, useBias, maxBiasVel, 0, batches.size());
            }

            // ---- Restitution ---------------------------------------------------
            // Lane-wide TGS restitution; the scalar twin is SoftStep::OverflowRestitution
            // (SoftStep.cpp). LOCKSTEP -- keep identical lane-for-scalar.
            // A whole contact with restitution <= 0 is skipped scalar-side; here we
            // mask it lane-wise (restitution<=0 -> no rebound impulse). A point only
            // rebounds when relVel <= -threshold AND nImp > 0.

            // Range overload: process batches [begin, end). Loop body unchanged.
            inline void ApplyRestitution(std::vector<ContactConstraintSimd>& batches,
                                         BodyState* states, float threshold,
                                         std::size_t begin, std::size_t end)
            {
                const f32w negThresh = splat(-threshold);
                const f32w zero = setzero();
                const b32w allTrue = cmp_gt(splat(1.0f), setzero());

                for (std::size_t i = begin; i < end; ++i)
                {
                    ContactConstraintSimd& cc = batches[i];
                    const i32w ia = iload(cc.bodyIndexA);
                    const i32w ib = iload(cc.bodyIndexB);
                    const b32w dyn = MaskOf(cc.dynB);

                    const f32w rest = load(cc.restitution);
                    const b32w ccHasRest = cmp_gt(rest, zero);
                    // SoftStep skips a contact with restitution<=0 entirely; if NO
                    // lane in this batch has restitution, the whole batch is a no-op.
                    if (none(ccHasRest))
                    {
                        continue;
                    }

                    const f32w nx = load(cc.normalX);
                    const f32w ny = load(cc.normalY);
                    const f32w iMa = load(cc.invMassA);
                    const f32w iIa = load(cc.invInertiaA);
                    const f32w iMb = load(cc.invMassB);
                    const f32w iIb = load(cc.invInertiaB);

                    // Restitution consumes velocity only (no dp/dq) -> velocity gather.
                    const BodyVelW sA = GatherVel(states, ia);
                    const BodyVelW sB = GatherVel(states, ib);
                    f32w vAx = sA.vx, vAy = sA.vy, wA = sA.w;
                    f32w vBx = sB.vx, vBy = sB.vy, wB = sB.w;

                    for (int p = 0; p < 2; ++p)
                    {
                        ContactConstraintSimd::Point& pt = cc.points[p];
                        const f32w rAx = load(pt.anchorAx), rAy = load(pt.anchorAy);
                        const f32w rBx = load(pt.anchorBx), rBy = load(pt.anchorBy);
                        const f32w nMass = load(pt.normalMass);
                        const f32w relVel = load(pt.relVel);
                        f32w nImp = load(pt.normalImpulse);

                        // Active lane: ccHasRest AND relVel <= -threshold AND nImp>0.
                        const b32w fast = cmp_le(relVel, negThresh);
                        const b32w took = cmp_gt(nImp, zero);
                        // combine masks: ccHasRest & fast & took. b32w has no & op,
                        // so AND via select (true-lane keeps next mask, else false).
                        const f32w fastF = select(fast, splat(1.0f), zero);
                        const f32w tookF = select(took, splat(1.0f), zero);
                        const f32w restF = select(ccHasRest, splat(1.0f), zero);
                        const b32w active = cmp_gt(restF * fastF * tookF, zero);

                        const f32w dvx = (vAx + (-wA * rAy)) - (vBx + (-wB * rBy));
                        const f32w dvy = (vAy + ( wA * rAx)) - (vBy + ( wB * rBx));
                        const f32w vn = dvx * nx + dvy * ny;

                        // impulse = -nMass*(vn + restitution*relVel); accumulated >=0.
                        f32w impulse = -nMass * (vn + rest * relVel);
                        const f32w newI = max(nImp + impulse, zero);
                        impulse = newI - nImp;
                        // gate the impulse to active lanes only (inactive -> 0).
                        impulse = select(active, impulse, zero);
                        // nImp only updates on active lanes (matches scalar skip).
                        store(pt.normalImpulse, select(active, newI, nImp));

                        const f32w Px = nx * impulse, Py = ny * impulse;
                        vAx = vAx + Px * iMa;
                        vAy = vAy + Py * iMa;
                        wA  = wA  + iIa * (rAx * Py - rAy * Px);
                        vBx = vBx - Px * iMb;
                        vBy = vBy - Py * iMb;
                        wB  = wB  - iIb * (rBx * Py - rBy * Px);
                    }

                    // A is always dynamic (unconditional scatter); B only where dynB.
                    ScatterVel(states, ia, vAx, vAy, wA, allTrue);
                    ScatterVel(states, ib, vBx, vBy, wB, dyn);
                }
            }
            inline void ApplyRestitution(std::vector<ContactConstraintSimd>& batches,
                                         BodyState* states, float threshold)
            {
                ApplyRestitution(batches, states, threshold, 0, batches.size());
            }

            // ---- Store accumulated impulses back onto ctx.contacts (Hazard 3) --
            // After the solve, the converged impulses live in the SoA batches. Copy
            // them back into ctx.contacts[ref].points[p].normalImpulse/tangentImpulse
            // so the world's stage-3b pool write-back persists warm-start. Keyed by
            // the color's refs (same order Build packed them: lane L of batch b is
            // global index b*kWidth+L = refs index, EXCLUDING padding lanes).
            // Range overload: store impulses for batches [begin, end). The ref index
            // is the GLOBAL batch index (b*W + L), so a sub-range still reads the
            // correct refs slot (refs is the whole color's ref array). Loop body
            // unchanged from the full overload below -- this lets the StoreImpulses
            // stage (Gap 1.1) block-partition a color's batch list like the other
            // colored passes.
            inline void StoreImpulses(const std::vector<ContactConstraintSimd>& batches,
                                      ContactConstraint* ccs, const std::uint32_t* refs,
                                      std::size_t begin, std::size_t end)
            {
                constexpr int W = ContactConstraintSimd::kWidth;
                for (std::size_t b = begin; b < end; ++b)
                {
                    const ContactConstraintSimd& batch = batches[b];
                    for (int L = 0; L < batch.count; ++L)   // live lanes only
                    {
                        const std::uint32_t ref = refs[b * W + static_cast<std::size_t>(L)];
                        ContactConstraint& cc = ccs[ref];
                        for (int p = 0; p < cc.pointCount; ++p)
                        {
                            cc.points[p].normalImpulse =
                                static_cast<Real>(batch.points[p].normalImpulse[L]);
                            cc.points[p].tangentImpulse =
                                static_cast<Real>(batch.points[p].tangentImpulse[L]);
                        }
                    }
                }
            }
            inline void StoreImpulses(const std::vector<ContactConstraintSimd>& batches,
                                      ContactConstraint* ccs, const std::uint32_t* refs)
            {
                StoreImpulses(batches, ccs, refs, 0, batches.size());
            }
        } // namespace SimdSolve

    } // namespace Physics
} // namespace Manifold2D
