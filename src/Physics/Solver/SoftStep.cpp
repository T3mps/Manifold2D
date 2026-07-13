// SoftStep.cpp -- Box2D-v3 TGS Soft constraint solver (M6, Task P2.2).
//
// See SoftStep.hpp for the algorithm overview + the PORT-vs-MODERNIZE framing.
// This TU implements the ISolver phases + the whole-Step Solve() driver. The
// soft coefficients follow Box2D v3 b2MakeSoft; the contact solve follows the
// v3 b2SolveContactsTask normal+friction form (separation re-evaluated each
// sub-step from per-body position deltas).
//
// PRESENTATION-FREE + C++20-clean: Geometry::Vec2 + std + sibling Physics headers.

#include <Manifold2D/Physics/Solver/SoftStep.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>

#include <Manifold2D/Physics/PhysicsWorld.hpp>
#include <Manifold2D/Physics/Joints/Joint.hpp>    // Joint (Prepare/SolveVelocity)
#include <Manifold2D/Physics/Solver/SoftCoeffs.hpp>   // shared MakeSoft + SoftCoeffs
#include <Manifold2D/Physics/Solver/BodyState.hpp>    // BodyStateStore sync defs
#include <Manifold2D/Core/WorkScheduler.hpp>            // IWorkScheduler::ParallelFor (Phase D1)

namespace Manifold2D
{
    namespace Physics
    {
        namespace
        {
            // 2D cross products (port of the Lua applyAt / contact math).
            //   scalar(w) x vec(r) -> vec : (-w*r.y, w*r.x)
            //   vec(r)    x vec(p) -> scalar : r.x*p.y - r.y*p.x
            inline Vec2 CrossWR(Real w, const Vec2& r) noexcept
            {
                return Vec2(-w * r.y, w * r.x);
            }
            inline Real CrossRP(const Vec2& r, const Vec2& p) noexcept
            {
                return r.x * p.y - r.y * p.x;
            }
            inline Real Dot(const Vec2& a, const Vec2& b) noexcept
            {
                return a.x * b.x + a.y * b.y;
            }
        } // namespace

        void SoftStep::DropBody(std::uint32_t slot)
        {
            // Warm-start impulses now live on the persistent Contact, which the
            // broadphase destroys when a fixture/body is removed
            // (ConstraintGraph::DestroyContactsForBody/Fixture), so a recycled slot
            // inherits no stale impulse. The per-body sub-step scratch is re-zeroed
            // at the start of each Solve. Nothing slot-specific to do here; the
            // override stays only to honor the ISolver contract.
            (void)slot;
        }

        // ===================================================================
        // BodyStateStore world<->solver sync (SIMD constraint-solver Task 1;
        // AoS re-layout Task 2; dense re-home Phase C, Task 2)
        // ===================================================================
        //
        // TWO INDEX SPACES live here. The SOLVER hot path uses the DENSE pair
        // SyncInCompacted / SyncOutCompacted (Phase C, Task 2): the scratch is sized
        // solverCount (= AwakeCount()+KinematicCount()) and indexed by solverIndex
        // (awake dynamics at AwakeIndexOf(slot) in [0,awakeCount); kinematics at
        // awakeCount+KinematicIndexOf(slot)). Statics/spans/padding have no row -- they
        // pack the null index (-1) and the lane-wide gather injects a zero identity
        // (Task 3; the old "+1" dummy tail is gone). The legacy SyncIn / SyncOut pair
        // (below) is the original sparse world-slot
        // bridge, kept only for the standalone SyncIn/SyncOut round-trip contract
        // test -- it is NO LONGER on the solver path. Both pairs share the same
        // awake-dynamic predicate (Alive && Dynamic && Awake), differing only in the
        // index they write/read. These defs live here (not the header) because they
        // need the PhysicsWorld slot accessors.
        //
        // CONTRACT (legacy SyncIn): the caller Resize()s the SoA to world.Count()
        // (no +1 dummy tail -- the scatter-safe dummy slot was removed; test-only
        // path) before SyncIn. SyncIn then OVERWRITES only the slots that satisfy the
        // awake-dynamic predicate. Non-matching slots (statics, kinematics, asleep/
        // dead dynamics) are LEFT AS-IS (zero after the caller's Resize). For matched
        // slots SyncIn also zeroes the TGS position-delta accumulators (dp/dq).
        // SyncIn additionally copies all Alive non-dynamic (Static/Kinematic) slots
        // so the solver's B-endpoint velocity reads see current values.

        void BodyStateStore::SyncIn(const PhysicsWorld& world)
        {
            // Two-pass design (Phase B, Task 3):
            //
            // (a) Awake dynamics: mirror velocity + reset the TGS position-delta
            //     accumulators. Only awake dynamics are integrated, so only they need
            //     a fresh dp/dq=0 at the start of the sub-step loop.
            world.ForEachAwake([&](std::uint32_t i)
            {
                const Vec2 v = world.VelSlot(i);
                m_states[i].vx  = v.x;
                m_states[i].vy  = v.y;
                m_states[i].w   = world.AngVelSlot(i);
                m_states[i].dpx = 0.f;
                m_states[i].dpy = 0.f;
                m_states[i].dq  = 0.f;
            });

            // (b) Static / kinematic slots are gathered as contact B-endpoints, so
            //     their velocity MUST stay correct in the AoS every step (statics = 0;
            //     kinematics = their set velocity). This also handles the recycled-slot
            //     hole: a slot that was a moving Dynamic, removed, then recycled as a
            //     Static would keep a stale non-zero velocity in the AoS if SyncIn
            //     never copies it. Non-dynamics do not integrate, so dp/dq stay zero
            //     (they arrive zero from the caller's Resize; we never touch them).
            //
            //     Sleeping dynamics are visited by NEITHER pass -- safe because no
            //     emitted constraint references a sleeping dynamic: EmitContactConstraints
            //     gates on awake-A, and a touching dynamic-dynamic pair shares one
            //     island (islands sleep as a unit), so awake-A implies awake-B.
            //     This is asserted in EmitContactConstraints (Debug builds only).
            const std::uint32_t count = world.Count();
            for (std::uint32_t i = 0; i < count; ++i)
            {
                if (!world.Alive(i) || world.TypeSlot(i) == BodyType::Dynamic)
                {
                    continue; // dead or dynamic (handled by ForEachAwake above)
                }
                const Vec2 v = world.VelSlot(i);
                m_states[i].vx = v.x;
                m_states[i].vy = v.y;
                m_states[i].w  = world.AngVelSlot(i);
                // dp/dq intentionally left at zero (non-dynamics never integrate).
            }
        }

        void BodyStateStore::SyncOut(PhysicsWorld& world) const
        {
            // Phase B, Task 3: iterate the awake-set directly -- it IS the gate.
            // Only awake dynamics were SyncIn'd and integrated; only they get written
            // back. Non-dynamics and sleeping dynamics are never disturbed.
            world.ForEachAwake([&](std::uint32_t i)
            {
                world.SetVelSlot(i, Vec2(m_states[i].vx, m_states[i].vy));
                world.SetAngVelSlot(i, m_states[i].w);
            });
        }

        void BodyStateStore::SyncInCompacted(const PhysicsWorld& world)
        {
            // DENSE fill (Phase C, Task 2). The caller Resize()d to solverCount, so
            // every dense row starts at zero (statics/spans pack the null index and
            // gather a zero identity -- no row to fill).
            //
            // (a) Awake dynamics -> dense row AwakeIndexOf(slot) in [0, awakeCount):
            //     mirror velocity + reset the TGS position-delta accumulators (only
            //     awake dynamics integrate, so only they need a fresh dp/dq = 0).
            world.ForEachAwake([&](std::uint32_t s)
            {
                const std::uint32_t i = world.AwakeIndexOf(s);
                const Vec2 v = world.VelSlot(s);
                m_states[i].vx  = v.x;
                m_states[i].vy  = v.y;
                m_states[i].w   = world.AngVelSlot(s);
                m_states[i].dpx = 0.f;
                m_states[i].dpy = 0.f;
                m_states[i].dq  = 0.f;
            });

            // (b) Kinematics -> dense row awakeCount + KinematicIndexOf(slot) in
            //     [awakeCount, solverCount): mirror velocity ONLY. A kinematic B is a
            //     contact endpoint whose authored velocity drives the relative-velocity
            //     push term, so its dense row MUST carry the current velocity. dp/dq
            //     stay at zero (kinematics never integrate -> the Resize zero stands).
            //     Sleeping dynamics get NO row -- safe because no emitted constraint
            //     references one (EmitContactConstraints gates on awake-A, and a
            //     touching dyn-dyn pair shares one island, so awake-A => awake-B).
            //     Statics likewise get no row (they pack the null index -> zero identity).
            const std::uint32_t awakeCount = world.AwakeCount();
            world.ForEachKinematic([&](std::uint32_t s)
            {
                const std::uint32_t i = awakeCount + world.KinematicIndexOf(s);
                const Vec2 v = world.VelSlot(s);
                m_states[i].vx = v.x;
                m_states[i].vy = v.y;
                m_states[i].w  = world.AngVelSlot(s);
                // dp/dq intentionally left at zero (kinematics never integrate).
            });
        }

        void BodyStateStore::SyncOutCompacted(PhysicsWorld& world) const
        {
            // DENSE write-back (Phase C, Task 2). Read each awake dynamic's velocity
            // from its dense row AwakeIndexOf(slot) and push it to the world. Only
            // awake dynamics were integrated; kinematics are read-only (their dense
            // rows are never written back), and statics/sleeping dynamics have no row.
            world.ForEachAwake([&](std::uint32_t s)
            {
                const std::uint32_t i = world.AwakeIndexOf(s);
                world.SetVelSlot(s, Vec2(m_states[i].vx, m_states[i].vy));
                world.SetAngVelSlot(s, m_states[i].w);
            });
        }

        // ===================================================================
        // Prepare
        // ===================================================================

        void SoftStep::PrepareContacts(SolverContext& ctx)
        {
            PhysicsWorld& w = *ctx.world;

            // High contact hertz that, with the small sub-step, appears rigid --
            // but never out-running the sub-step solve rate. Box2D v3
            // (physics_world.c) clamps to 0.125 * inv_h == 0.125 / h. No behavioral
            // change at the default 4 substeps (min(30, 0.125*240) == 30, same as the
            // old 0.25/h cap); correct for any future sub-4 substep count.
            const Real h = ctx.subDt;
            const Real maxHertz = (h > Real(0)) ? (Real(0.125) / h) : w.ContactHertz();
            const Real contactHertz = std::min(w.ContactHertz(), maxHertz);
            const SoftCoeffs contactSoft = MakeSoft(contactHertz, w.ContactDampingRatio(), h);

            for (std::uint32_t c = 0; c < ctx.contactCount; ++c)
            {
                ContactConstraint& cc = ctx.contacts[c];

                cc.biasRate     = contactSoft.biasRate;
                cc.massScale    = contactSoft.massScale;
                cc.impulseScale = contactSoft.impulseScale;

                const Vec2 n = cc.normal;
                const Vec2 tangent(-n.y, n.x);

                const Real iMa = cc.invMassA, iIa = cc.invInertiaA;
                const Real iMb = cc.invMassB, iIb = cc.invInertiaB;

                const Vec2 vA = w.VelSlot(cc.bodyA);
                const Real wA = w.AngVelSlot(cc.bodyA);
                Vec2 vB(Real(0), Real(0));
                Real wB = Real(0);
                if (cc.bodyBIsBody)
                {
                    vB = w.VelSlot(cc.bodyB);
                    wB = w.AngVelSlot(cc.bodyB);
                }

                for (int p = 0; p < cc.pointCount; ++p)
                {
                    ContactConstraintPoint& cp = cc.points[p];

                    const Vec2 rA = cp.anchorA;
                    const Vec2 rB = cp.anchorB;

                    // Effective normal mass: 1/(iMa+iMb + iIa*(rA x n)^2 + iIb*(rB x n)^2).
                    const Real rnA = CrossRP(rA, n);
                    const Real rnB = CrossRP(rB, n);
                    const Real kNormal = iMa + iMb + iIa * rnA * rnA + iIb * rnB * rnB;
                    cp.normalMass = (kNormal > Real(0)) ? (Real(1) / kNormal) : Real(0);

                    const Real rtA = CrossRP(rA, tangent);
                    const Real rtB = CrossRP(rB, tangent);
                    const Real kTangent = iMa + iMb + iIa * rtA * rtA + iIb * rtB * rtB;
                    cp.tangentMass = (kTangent > Real(0)) ? (Real(1) / kTangent) : Real(0);

                    // Relative normal velocity at the contact (for restitution),
                    // measured as A-relative-to-B along the B->A normal. When A
                    // approaches B this is NEGATIVE (the closing speed).
                    const Vec2 dv = (vA + CrossWR(wA, rA)) - (vB + CrossWR(wB, rB));
                    cp.relativeVelocity = Dot(dv, n);

                    // Warm start: the accumulators ARRIVE already seeded on cp
                    // (EmitContactConstraints copied them off the persistent
                    // Contact's manifold point; 0 for a fresh feature or a
                    // transient tile span). Prepare must NOT reset them -- doing so
                    // would discard the carried-forward impulse history.
                }
            }
        }

        void SoftStep::PrepareJoints(SolverContext& ctx)
        {
            // Prepare each joint with the SUB-STEP dt so its Baumgarte bias
            // (BETA/subDt) is per-substep-correct: the bias is applied once per
            // sub-step's velocity solve, summed over substepCount sub-steps. This
            // is the "soft-constraint formulation" the plan means -- sub-stepping
            // softens the Baumgarte joints. The joints read the committed (start-
            // of-step) positions; SoftStep tracks in-flight motion in deltaPos/
            // deltaRot (committed only at FinalizePositions), so a single Prepare
            // with subDt is correct (re-Preparing each sub-step would recompute
            // the same start-relative bias). Index-ordered (determinism).
            PhysicsWorld& w = *ctx.world;
            for (std::uint32_t k = 0; k < ctx.jointCount; ++k)
            {
                if (ctx.joints[k].joint != nullptr)
                {
                    ctx.joints[k].joint->Prepare(w, ctx.subDt);
                }
            }
        }

        void SoftStep::SolveJoints(SolverContext& ctx)
        {
            // One velocity-constraint pass over the joints (per sub-step). Joints
            // solve alongside contacts in the sub-step velocity solve. Index-
            // ordered (determinism).
            PhysicsWorld& w = *ctx.world;
            for (std::uint32_t k = 0; k < ctx.jointCount; ++k)
            {
                if (ctx.joints[k].joint != nullptr)
                {
                    ctx.joints[k].joint->SolveVelocity(w);
                }
            }
        }

        // ===================================================================
        // SIMD lane-wide solve helpers (Part 1) -- integrate / bridge / overflow
        // ===================================================================
        //
        // These operate on m_bodyState (the packed SoA the lane-wide passes
        // gather/scatter through), NOT the world velocity SoA. The world<->SoA
        // sync happens at the Step boundary (SyncIn/SyncOut) + around joint passes
        // (SyncVelTo/FromWorld). The integrate loops stay scalar O(n) (not the hot
        // path); the WIN is the lane-wide colored contact solve.

        void SoftStep::IntegrateVelocitiesRange(SolverContext& ctx, Real h,
                                                std::size_t begin, std::size_t end)
        {
            // Phase B, Task 3: iterate the awake-set directly -- the set guarantees
            // Alive + Dynamic + Awake. Only the InvMassSlot<=0 guard is retained
            // (a degenerate zero-invMass dynamic is valid; it never integrates).
            // Phase C, Task 2: index m_bodyState by the DENSE solverIndex
            // AwakeIndexOf(slot), NOT the world slot.
            // Phase D1, Task 4: iterate over the INDEXABLE AwakeBodies() vector.
            // Gap 1.1: this is the [begin,end) loop body a solver-region body stage
            // block runs. Per-body writes are disjoint (each awake slot is unique),
            // so any partition is scatter-safe and byte-identical at any worker count.
            PhysicsWorld& w = *ctx.world;
            const Vec2 g = ctx.gravity;
            const std::vector<std::uint32_t>& aw = w.AwakeBodies();
            for (std::size_t j = begin; j < end; ++j)
            {
                const std::uint32_t s = aw[j];
                if (w.InvMassSlot(s) <= Real(0))
                {
                    continue; // pinned dynamic -- never integrates
                }
                const std::uint32_t i = w.AwakeIndexOf(s);
                float vx = m_bodyState[i].vx;
                float vy = m_bodyState[i].vy;
                float wv = m_bodyState[i].w;
                // Box2D v3 order (solver.c:102-106): damp the OLD velocity FIRST,
                // then add the UNDAMPED gravity delta -- v = h*g + linearDamping*v_old.
                // (Arcane has no external force/torque accumulators, so the delta is
                // gravity only; angular velocity gets damping only.) This is
                // byte-identical to the old damp-the-gravity-too form whenever d == 0
                // (the branch is skipped); for d > 0 the gravity term is no longer
                // scaled by f, so the discrete terminal velocity is g/d + g*h (the
                // semi-implicit fixed point) instead of the old g/d.
                const Real d = w.LinDampSlot(s);
                if (d > Real(0))
                {
                    const float f = static_cast<float>(Real(1) / (Real(1) + d * h));
                    vx *= f; vy *= f; wv *= f;
                }
                vx += static_cast<float>(g.x * h);
                vy += static_cast<float>(g.y * h);
                // Clamp to max linear speed (Box2D v3 b2IntegrateVelocitiesTask,
                // solver.c:108-114). Manifold2D units == Box2D units.
                const float maxLin   = static_cast<float>(w.MaxLinearVelocity());
                const float maxLinSq = maxLin * maxLin;
                if (vx * vx + vy * vy > maxLinSq)
                {
                    const float ratio = maxLin / std::sqrt(vx * vx + vy * vy);
                    vx *= ratio; vy *= ratio;
                }
                // Clamp to max angular speed = kMaxRotation * invDt (full-step
                // inverse), solver.c:75,116-122. Arcane exposes no allowFastRotation
                // opt-out (all bodies == Box2D default allowFastRotation=false), so
                // the clamp is unconditional.
                const float maxAng   = static_cast<float>(kMaxRotation * ctx.invDt);
                const float maxAngSq = maxAng * maxAng;
                if (wv * wv > maxAngSq)
                {
                    const float ratio = maxAng / std::abs(wv);
                    wv *= ratio;
                }
                m_bodyState[i].vx = vx;
                m_bodyState[i].vy = vy;
                m_bodyState[i].w  = wv;
            }
        }

        void SoftStep::IntegratePositionsRange(SolverContext& ctx, Real h,
                                               std::size_t begin, std::size_t end)
        {
            // Phase B, Task 3: iterate the awake-set directly -- Alive+Dynamic+Awake
            // is guaranteed. dp/dq += v*h for the TGS separation re-evaluation.
            // Phase C, Task 2: index by the DENSE solverIndex AwakeIndexOf(slot).
            // Gap 1.1: the [begin,end) loop body a solver-region body stage block
            // runs. Each body writes its own dense row (AwakeIndexOf is a bijection),
            // so writes are disjoint -> any partition is byte-identical.
            PhysicsWorld& w = *ctx.world;
            const float fh = static_cast<float>(h);
            const std::vector<std::uint32_t>& aw = w.AwakeBodies();
            for (std::size_t j = begin; j < end; ++j)
            {
                const std::uint32_t s = aw[j];
                const std::uint32_t i = w.AwakeIndexOf(s);
                m_bodyState[i].dpx += m_bodyState[i].vx * fh;
                m_bodyState[i].dpy += m_bodyState[i].vy * fh;
                m_bodyState[i].dq  += m_bodyState[i].w  * fh;
            }
        }

        void SoftStep::FinalizePositionsSoA(SolverContext& ctx)
        {
            // Phase B, Task 3: iterate the awake-set directly -- Alive+Dynamic+Awake
            // is guaranteed. Commit the compound-COM position from SoA dp/dq.
            // For localCenter==(0,0) this reduces to p=p0+dp, a=a0+dq.
            // Phase C, Task 2: index dp/dq by the DENSE solverIndex AwakeIndexOf(slot);
            // CommitSlotPosition still takes the WORLD slot `s` (unchanged).
            // Phase D1, Task 4: this loop is intentionally SERIAL -- CommitSlotPosition
            // calls UpdateMoverProxies which modifies the shared DynamicTree; concurrent
            // calls from different workers would race on tree-node updates even for
            // disjoint body slots (tree rotations touch shared parent nodes). It runs
            // OUTSIDE the persistent solver region (post-region, after SyncOutCompacted),
            // called ONCE per step (not in the substep hot loop), so it is not the
            // bottleneck. The IntegrateVelocities/IntegratePositions body stages, by
            // contrast, write disjoint m_bodyState dense rows and ARE run inside the MT
            // region (IntegrateVelocitiesRange / IntegratePositionsRange).
            PhysicsWorld& w = *ctx.world;
            const std::vector<std::uint32_t>& aw = w.AwakeBodies();
            for (std::size_t j = 0; j < aw.size(); ++j)
            {
                const std::uint32_t s = aw[j];
                const std::uint32_t i = w.AwakeIndexOf(s);
                const Vec2 dp(static_cast<Real>(m_bodyState[i].dpx),
                              static_cast<Real>(m_bodyState[i].dpy));
                const Real dr = static_cast<Real>(m_bodyState[i].dq);
                const Vec2 p0 = w.PosSlot(s);
                const Real a0 = w.GetAngle(w.HandleOf(s));
                const Vec2 lc = w.LocalCenterSlot(s);
                const Vec2 c0 = p0 + RotateVec(a0, lc);
                const Vec2 c  = c0 + dp;
                const Real a  = a0 + dr;
                const Vec2 p  = c - RotateVec(a, lc);
                w.CommitSlotPosition(s, p, a);
            }
        }

        void SoftStep::SyncVelToWorld(SolverContext& ctx)
        {
            // Push the SoA velocities back to the world so the scalar joint passes
            // read the in-flight (contact-solved) velocities. Awake dynamics only.
            // Phase C, Task 2: index m_bodyState by the DENSE solverIndex; iterate
            // the awake-set (it IS the Alive+Dynamic+Awake gate).
            PhysicsWorld& w = *ctx.world;
            w.ForEachAwake([&](std::uint32_t s)
            {
                const std::uint32_t i = w.AwakeIndexOf(s);
                w.SetVelSlot(s, Vec2(static_cast<Real>(m_bodyState[i].vx),
                                     static_cast<Real>(m_bodyState[i].vy)));
                w.SetAngVelSlot(s, static_cast<Real>(m_bodyState[i].w));
            });
        }

        void SoftStep::SyncVelFromWorld(SolverContext& ctx)
        {
            // Pull the world velocities back into the SoA after a joint pass mutated
            // them. dp/dq untouched (joints do not read them). Awake dynamics only.
            // Phase C, Task 2: index m_bodyState by the DENSE solverIndex.
            PhysicsWorld& w = *ctx.world;
            w.ForEachAwake([&](std::uint32_t s)
            {
                const std::uint32_t i = w.AwakeIndexOf(s);
                const Vec2 v = w.VelSlot(s);
                m_bodyState[i].vx = static_cast<float>(v.x);
                m_bodyState[i].vy = static_cast<float>(v.y);
                m_bodyState[i].w  = static_cast<float>(w.AngVelSlot(s));
            });
        }

        // ----- Overflow: width-1 SCALAR solve over the SoA (scatter-safe) ------
        //
        // Phase C, Task 5: the overflow set is m_overflowRefs -- the emitted
        // constraints whose PERSISTENT contact color is kInvalidColor (an overflow
        // contact whose source Contact found no free color among kColorCount at
        // create -- a hub body in > kColorCount contacts -- OR a transient span with
        // no pool home). They CANNOT solve lane-wide (an overflow hub shares a body
        // with a colored contact), so we solve them one-at-a-time, scalar, over the
        // SAME BodyStateStore -- sequential => no scatter conflict. They read/write the
        // SoA velocity + dp/dq (so overflow composes with the colored result in the
        // same velocity store). One contact per call iter.
        //
        // LOCKSTEP: this overflow trio (OverflowWarmStart / OverflowSolve /
        // OverflowRestitution) is the WIDTH-1 SEQUENTIAL version of the same TGS
        // step the lane-wide SimdSolve::WarmStart / SolveNormalAndFriction /
        // ApplyRestitution passes run (ContactConstraintSimd.hpp). The two paths
        // share one m_bodyState and compose in one velocity store, so they MUST stay
        // numerically in lockstep lane-for-scalar. Keep the math here identical to
        // the SimdSolve::* passes (and to the plain-float ScalarRef oracle in
        // PhysicsSimdSolverTest.cpp); change one and you change all three.

        namespace
        {
            // Per-overflow-constraint scalar helpers (SoA-resident). cc is the
            // emitted ContactConstraint; bs holds vel + dp/dq by DENSE solverIndex
            // (Phase C, Task 2).
            //   dynB    : B is a real DYNAMIC body whose velocity is MUTATED.
            //   ia      : DENSE solverIndex of A (always an awake dynamic, >= 0).
            //   ib      : DENSE solverIndex of B, or kNullBodyIndex (-1) for a
            //             static body / tile span (read vB=0, skip the B write) --
            //             the scalar twin of the lane-wide null-index gather/scatter.
            struct OverflowBodies
            {
                std::uint32_t ia;
                std::int32_t  ib;
                bool dynB;
                Real iMa, iIa, iMb, iIb;
            };

            // Map a contact B-endpoint world slot to its dense solverIndex, matching
            // the packer's kinematic-vs-static gate exactly: a dynamic B -> its awake
            // row; a kinematic B -> awakeCount + its kinematic row (real velocity ->
            // push); a static B or a span -> kNullBodyIndex (the scalar twin of the
            // lane-wide null-index branch: read vB=0, never write B).
            inline std::int32_t DenseB(const ContactConstraint& cc, const PhysicsWorld& w,
                                       std::uint32_t awakeCount, bool dynB)
            {
                if (!cc.bodyBIsBody)
                {
                    return kNullBodyIndex;              // tile span -> null
                }
                if (dynB)
                {
                    return static_cast<std::int32_t>(w.AwakeIndexOf(cc.bodyB)); // dynamic B -> awake row
                }
                const std::uint32_t ki = w.KinematicIndexOf(cc.bodyB);
                if (ki != kNotKinematic)
                {
                    return static_cast<std::int32_t>(awakeCount + ki); // kinematic B -> its dense row
                }
                // static B -> null. NOTE: a non-idiomatic moving ZERO-invMass
                // *dynamic* B (BodyType::Dynamic with invMass==0) is not dynB and not
                // in the kinematic set, so it falls HERE -> the zero identity; its
                // velocity is NOT gathered and its push is dropped BY DESIGN -- use a
                // Kinematic body for an infinite-mass mover.
                return kNullBodyIndex;
            }

            // ia is a DENSE solverIndex; ib is a DENSE solverIndex or kNullBodyIndex.
            // awakeCount comes from the world's per-step dense index space; A is
            // ALWAYS an awake dynamic.
            inline OverflowBodies OverflowSetup(const ContactConstraint& cc, const PhysicsWorld& w,
                                                std::uint32_t awakeCount)
            {
                OverflowBodies ob;
                ob.dynB    = cc.bodyBIsBody && cc.invMassB > Real(0);
                ob.ia      = w.AwakeIndexOf(cc.bodyA);
                ob.ib      = DenseB(cc, w, awakeCount, ob.dynB);
                ob.iMa     = cc.invMassA; ob.iIa = cc.invInertiaA;
                ob.iMb     = cc.invMassB; ob.iIb = cc.invInertiaB;
                return ob;
            }
        } // namespace

        void SoftStep::OverflowWarmStart(SolverContext& ctx)
        {
            const PhysicsWorld& w = *ctx.world;
            const std::uint32_t awakeCount = w.AwakeCount();
            for (std::uint32_t ref : m_overflowRefs)
            {
                ContactConstraint& cc = ctx.contacts[ref];
                const OverflowBodies ob = OverflowSetup(cc, w, awakeCount);
                const Vec2 n = cc.normal;
                const Vec2 tangent(-n.y, n.x);

                Vec2 vA(m_bodyState[ob.ia].vx, m_bodyState[ob.ia].vy);
                Real wA = m_bodyState[ob.ia].w;
                // B read mirrors the lane-wide null-index gather: a real row (>=0)
                // is read for its velocity (dynamic OR kinematic push), a -1 (static/
                // span) reads the zero identity -> vB=0, wB=0.
                Vec2 vB(Real(0), Real(0)); Real wB = Real(0);
                if (ob.ib >= 0)
                {
                    const std::size_t ib = static_cast<std::size_t>(ob.ib);
                    vB = Vec2(m_bodyState[ib].vx, m_bodyState[ib].vy); wB = m_bodyState[ib].w;
                }

                for (int p = 0; p < cc.pointCount; ++p)
                {
                    const ContactConstraintPoint& cp = cc.points[p];
                    const Vec2 P = n * cp.normalImpulse + tangent * cp.tangentImpulse;
                    vA += P * ob.iMa;
                    wA += ob.iIa * CrossRP(cp.anchorA, P);
                    if (ob.dynB) { vB -= P * ob.iMb; wB -= ob.iIb * CrossRP(cp.anchorB, P); }
                }

                m_bodyState[ob.ia].vx = static_cast<float>(vA.x);
                m_bodyState[ob.ia].vy = static_cast<float>(vA.y);
                m_bodyState[ob.ia].w  = static_cast<float>(wA);
                // dynB => ib >= 0 (a dynamic B always has a real awake row).
                if (ob.dynB)
                {
                    const std::size_t ib = static_cast<std::size_t>(ob.ib);
                    m_bodyState[ib].vx = static_cast<float>(vB.x);
                    m_bodyState[ib].vy = static_cast<float>(vB.y);
                    m_bodyState[ib].w  = static_cast<float>(wB);
                }
            }
        }

        void SoftStep::OverflowSolve(SolverContext& ctx, Real h, bool useBias)
        {
            const Real invH = (h > Real(0)) ? (Real(1) / h) : Real(0);
            const Real maxBiasVel = ctx.world->ContactPushMaxVelocity();
            const PhysicsWorld& w = *ctx.world;
            const std::uint32_t awakeCount = w.AwakeCount();

            for (std::uint32_t ref : m_overflowRefs)
            {
                ContactConstraint& cc = ctx.contacts[ref];
                const OverflowBodies ob = OverflowSetup(cc, w, awakeCount);
                const Vec2 n = cc.normal;
                const Vec2 tangent(-n.y, n.x);

                Vec2 vA(m_bodyState[ob.ia].vx, m_bodyState[ob.ia].vy);
                Real wA = m_bodyState[ob.ia].w;
                // B velocity read: real row (>=0) -> its velocity; -1 -> zero identity.
                Vec2 vB(Real(0), Real(0)); Real wB = Real(0);
                if (ob.ib >= 0)
                {
                    const std::size_t ib = static_cast<std::size_t>(ob.ib);
                    vB = Vec2(m_bodyState[ib].vx, m_bodyState[ib].vy); wB = m_bodyState[ib].w;
                }

                const Vec2 dpA(m_bodyState[ob.ia].dpx, m_bodyState[ob.ia].dpy);
                const Real drA = m_bodyState[ob.ia].dq;
                // Only a dynamic B integrates dp/dq; a kinematic/static B's deltas are
                // zero (matching the lane-wide gather of a zero kinematic row / identity).
                Vec2 dpB(Real(0), Real(0)); Real drB = Real(0);
                if (ob.dynB)
                {
                    const std::size_t ib = static_cast<std::size_t>(ob.ib);
                    dpB = Vec2(m_bodyState[ib].dpx, m_bodyState[ib].dpy); drB = m_bodyState[ib].dq;
                }

                for (int p = 0; p < cc.pointCount; ++p)
                {
                    ContactConstraintPoint& cp = cc.points[p];
                    const Vec2 rA = cp.anchorA, rB = cp.anchorB;
                    const Vec2 prA = dpA + CrossWR(drA, rA);
                    const Vec2 prB = dpB + CrossWR(drB, rB);
                    const Real s = cp.baseSeparation + Dot(prA - prB, n);

                    Real bias = Real(0), massScale = Real(1), impulseScale = Real(0);
                    if (s > Real(0))            { bias = s * invH; }
                    else if (useBias)           { bias = std::max(cc.biasRate * s, -maxBiasVel); massScale = cc.massScale; impulseScale = cc.impulseScale; }

                    const Vec2 dv = (vA + CrossWR(wA, rA)) - (vB + CrossWR(wB, rB));
                    const Real vn = Dot(dv, n);
                    Real impulse = -cp.normalMass * massScale * (vn + bias) - impulseScale * cp.normalImpulse;
                    const Real newImpulse = std::max(cp.normalImpulse + impulse, Real(0));
                    impulse = newImpulse - cp.normalImpulse;
                    cp.normalImpulse = newImpulse;

                    const Vec2 P = n * impulse;
                    vA += P * ob.iMa; wA += ob.iIa * CrossRP(rA, P);
                    if (ob.dynB) { vB -= P * ob.iMb; wB -= ob.iIb * CrossRP(rB, P); }
                }

                for (int p = 0; p < cc.pointCount; ++p)
                {
                    ContactConstraintPoint& cp = cc.points[p];
                    const Vec2 rA = cp.anchorA, rB = cp.anchorB;
                    const Vec2 dv = (vA + CrossWR(wA, rA)) - (vB + CrossWR(wB, rB));
                    const Real vt = Dot(dv, tangent);
                    Real impulse = -cp.tangentMass * vt;
                    const Real maxFriction = cc.friction * cp.normalImpulse;
                    const Real newImpulse = std::clamp(cp.tangentImpulse + impulse, -maxFriction, maxFriction);
                    impulse = newImpulse - cp.tangentImpulse;
                    cp.tangentImpulse = newImpulse;

                    const Vec2 P = tangent * impulse;
                    vA += P * ob.iMa; wA += ob.iIa * CrossRP(rA, P);
                    if (ob.dynB) { vB -= P * ob.iMb; wB -= ob.iIb * CrossRP(rB, P); }
                }

                m_bodyState[ob.ia].vx = static_cast<float>(vA.x);
                m_bodyState[ob.ia].vy = static_cast<float>(vA.y);
                m_bodyState[ob.ia].w  = static_cast<float>(wA);
                if (ob.dynB)   // dynB => ib >= 0
                {
                    const std::size_t ib = static_cast<std::size_t>(ob.ib);
                    m_bodyState[ib].vx = static_cast<float>(vB.x);
                    m_bodyState[ib].vy = static_cast<float>(vB.y);
                    m_bodyState[ib].w  = static_cast<float>(wB);
                }
            }
        }

        void SoftStep::OverflowRestitution(SolverContext& ctx)
        {
            const Real threshold = ctx.world->RestitutionThreshold();
            const PhysicsWorld& w = *ctx.world;
            const std::uint32_t awakeCount = w.AwakeCount();
            for (std::uint32_t ref : m_overflowRefs)
            {
                ContactConstraint& cc = ctx.contacts[ref];
                if (cc.restitution <= Real(0)) { continue; }
                const OverflowBodies ob = OverflowSetup(cc, w, awakeCount);
                const Vec2 n = cc.normal;

                Vec2 vA(m_bodyState[ob.ia].vx, m_bodyState[ob.ia].vy);
                Real wA = m_bodyState[ob.ia].w;
                // B velocity read: real row (>=0) -> its velocity; -1 -> zero identity.
                Vec2 vB(Real(0), Real(0)); Real wB = Real(0);
                if (ob.ib >= 0)
                {
                    const std::size_t ib = static_cast<std::size_t>(ob.ib);
                    vB = Vec2(m_bodyState[ib].vx, m_bodyState[ib].vy); wB = m_bodyState[ib].w;
                }

                for (int p = 0; p < cc.pointCount; ++p)
                {
                    ContactConstraintPoint& cp = cc.points[p];
                    if (cp.relativeVelocity > -threshold || cp.normalImpulse <= Real(0)) { continue; }
                    const Vec2 rA = cp.anchorA, rB = cp.anchorB;
                    const Vec2 dv = (vA + CrossWR(wA, rA)) - (vB + CrossWR(wB, rB));
                    const Real vn = Dot(dv, n);
                    Real impulse = -cp.normalMass * (vn + cc.restitution * cp.relativeVelocity);
                    const Real newImpulse = std::max(cp.normalImpulse + impulse, Real(0));
                    impulse = newImpulse - cp.normalImpulse;
                    cp.normalImpulse = newImpulse;

                    const Vec2 P = n * impulse;
                    vA += P * ob.iMa; wA += ob.iIa * CrossRP(rA, P);
                    if (ob.dynB) { vB -= P * ob.iMb; wB -= ob.iIb * CrossRP(rB, P); }
                }

                m_bodyState[ob.ia].vx = static_cast<float>(vA.x);
                m_bodyState[ob.ia].vy = static_cast<float>(vA.y);
                m_bodyState[ob.ia].w  = static_cast<float>(wA);
                if (ob.dynB)   // dynB => ib >= 0
                {
                    const std::size_t ib = static_cast<std::size_t>(ob.ib);
                    m_bodyState[ib].vx = static_cast<float>(vB.x);
                    m_bodyState[ib].vy = static_cast<float>(vB.y);
                    m_bodyState[ib].w  = static_cast<float>(wB);
                }
            }
        }

        // ===================================================================
        // Stage-list build (Gap 1.1) -- the flat per-step SolverStage/SolverBlock
        // list the persistent worker region walks (replaces the per-color ParallelFor
        // dispatch storm). Reused scratch (m_stages/m_blocks): resize-not-realloc ->
        // zero steady-state alloc.
        // ===================================================================

        int SoftStep::BuildStages(std::uint32_t awakeCount, std::uint32_t sizingWorkers)
        {
            // Box2D b2ComputeBlockCount sizing: contacts min ~8 SIMD batches per block
            // (tuned 4->8 to reduce ring-CAS overhead), bodies min ~32, target ~=
            // blocksPerWorker * workerCount.
            constexpr int kContactMinBlock = 8;   // SIMD batches per contact block
            constexpr int kBodyMinBlock    = 32;  // awake bodies per body block
            const int target = 4 * static_cast<int>(sizingWorkers > 0u ? sizingWorkers : 1u);

            // Active (non-empty) colors in ASCENDING index order. The old per-color
            // loop walked colors 0..kColorCount-1 (empty ones no-op); colors are
            // Gauss-Seidel-ordered (adjacent colors share bodies), so preserving the
            // ascending order is required for byte-identity.
            int activeIdx[kColorCount];
            int C = 0;
            for (int k = 0; k < kColorCount; ++k)
            {
                if (!m_colorBatches[static_cast<std::size_t>(k)].empty())
                {
                    activeIdx[C++] = k;
                }
            }

            const int bodyItems  = static_cast<int>(awakeCount);
            const int bodyBlocks = ComputeBlockCount(bodyItems, kBodyMinBlock, target);

            // Total stages = 2 body (integrateVel + integratePos) + 5 colored phases
            // (warmStart/solve/relax/restitution/storeImpulses) x C active colors.
            const int stageCount = 2 + 5 * C;

            // Total blocks = integrateVel + integratePos (bodyBlocks each) + each
            // colored stage's per-color block count, x5 colored phases.
            int totalBlocks = 2 * bodyBlocks;
            for (int c = 0; c < C; ++c)
            {
                const int batches = static_cast<int>(
                    m_colorBatches[static_cast<std::size_t>(activeIdx[c])].size());
                totalBlocks += 5 * ComputeBlockCount(batches, kContactMinBlock, target);
            }

            // Reuse scratch (grows only on a new high-water; SolverStage/SolverBlock
            // carry a relaxed copy ctor so resize/realloc is well-formed -- copies
            // happen only here, single-threaded, never inside the region).
            m_blocks.resize(static_cast<std::size_t>(totalBlocks));
            m_stages.resize(static_cast<std::size_t>(stageCount));

            SolverBlock* const blockBase = m_blocks.data();
            int blockOffset = 0;

            auto setupStage = [&](int stageIdx, StageType type, int colorIndex,
                                  int itemCount, int minBlock)
            {
                SolverStage& st = m_stages[static_cast<std::size_t>(stageIdx)];
                st.type = type;
                st.colorIndex = colorIndex;
                st.completionCount.store(0, std::memory_order_relaxed);
                st.syncIndex = 0;
                const int bc = ComputeBlockCount(itemCount, minBlock, target);
                st.blockCount = bc;
                st.blocks = (bc > 0) ? (blockBase + blockOffset) : nullptr;
                if (bc > 0)
                {
                    PartitionBlocks(st.blocks, bc, itemCount);
                    blockOffset += bc;
                }
            };

            // Lay out the stages in the order SolverWorker walks (see SolverStages.hpp).
            setupStage(0, StageType::IntegrateVelocities, 0, bodyItems, kBodyMinBlock);
            for (int c = 0; c < C; ++c)
            {
                const int color = activeIdx[c];
                const int batches = static_cast<int>(m_colorBatches[static_cast<std::size_t>(color)].size());
                setupStage(1 + c, StageType::WarmStart, color, batches, kContactMinBlock);
            }
            for (int c = 0; c < C; ++c)
            {
                const int color = activeIdx[c];
                const int batches = static_cast<int>(m_colorBatches[static_cast<std::size_t>(color)].size());
                setupStage(1 + C + c, StageType::Solve, color, batches, kContactMinBlock);
            }
            setupStage(1 + 2 * C, StageType::IntegratePositions, 0, bodyItems, kBodyMinBlock);
            for (int c = 0; c < C; ++c)
            {
                const int color = activeIdx[c];
                const int batches = static_cast<int>(m_colorBatches[static_cast<std::size_t>(color)].size());
                setupStage(2 + 2 * C + c, StageType::Relax, color, batches, kContactMinBlock);
            }
            for (int c = 0; c < C; ++c)
            {
                const int color = activeIdx[c];
                const int batches = static_cast<int>(m_colorBatches[static_cast<std::size_t>(color)].size());
                setupStage(2 + 3 * C + c, StageType::Restitution, color, batches, kContactMinBlock);
            }
            for (int c = 0; c < C; ++c)
            {
                const int color = activeIdx[c];
                const int batches = static_cast<int>(m_colorBatches[static_cast<std::size_t>(color)].size());
                setupStage(2 + 4 * C + c, StageType::StoreImpulses, color, batches, kContactMinBlock);
            }

            return C;
        }

        // ===================================================================
        // Whole-Step driver (lane-wide colored SoA contact solve, Part 1)
        // ===================================================================

        void SoftStep::Solve(SolverContext& ctx)
        {
            assert(ctx.executor && "SolverContext::executor must be set (PhysicsWorld resolves a serial default)");
            PhysicsWorld& w = *ctx.world;

            // Phase C, Task 2: the body-state scratch is now DENSE, sized by the
            // per-step solver index space (awake dynamics + kinematics), NOT the
            // sparse world slot count. awakeCount = the awake-dynamic count
            // (solverIndex [0, awakeCount)); solverCount = awakeCount + kinematicCount
            // (kinematics occupy [awakeCount, solverCount)). No dummy tail: statics,
            // overflow spans, and padding lanes use null-index (-1), which the gather
            // resolves to the shared kIdentityRow zero state (no real-body memory touch)
            // and the scatter skips entirely.
            const std::uint32_t awakeCount  = w.AwakeCount();
            const std::uint32_t solverCount = awakeCount + w.KinematicCount();

            const Real h = ctx.subDt;
            const int substeps = static_cast<int>(ctx.substepCount);
            const bool hasJoints = ctx.jointCount > 0;
            const float maxBiasVel = static_cast<float>(w.ContactPushMaxVelocity());
            const float threshold = static_cast<float>(w.RestitutionThreshold());

            // 1) Prepare (once), SCALAR on ctx.contacts -- effective masses, soft
            //    coefficients, rest-relative velocity. Warm-start impulses arrive
            //    already seeded on each ContactConstraintPoint (emit). We run Prepare
            //    BEFORE building the SoA batches so Build packs the PREPARED values
            //    (lower churn than a lane-wide Prepare; the per-step b2MakeSoft is
            //    scalar regardless). Joints prepare as before (Part 2 leaves joints
            //    scalar).
            PrepareContacts(ctx);
            PrepareJoints(ctx);

            // 2) Body-state SoA: size to solverCount (NO dummy tail -- Task 3 dropped
            //    the scatter-safe dummy slot for the null-index branch) and sync world
            //    velocities into the DENSE rows (zeroes dp/dq for awake dynamics;
            //    kinematics get a real velocity row). A read-only-B / padding lane now
            //    packs kNullBodyIndex (-1): the gather injects a shared zero identity
            //    row (no real-body memory touch) and the scatter skips it.
            m_bodyState.Resize(solverCount);
            m_bodyState.SyncInCompacted(w);

            // 3) Group the emitted constraints by their PERSISTENT contact color
            //    (Phase C, Task 4/5). The per-step greedy recolor is GONE: every
            //    solver-relevant body-body contact was colored once at create
            //    (ConstraintGraph::AssignContactColor) and EmitContactConstraints copied
            //    that color onto cc.color. A kInvalidColor constraint -- an OVERFLOW
            //    contact (the source Contact found no free color at create) OR a span
            //    (no pool home) -- goes to the scalar overflow path; everything else
            //    buckets into its color. The colored solve is within-color order-
            //    independent (a valid coloring has one contact per dynamic body per
            //    color), so bucket order does not affect the result.
            //
            //    WHY THIS IS SCATTER-SAFE: the persistent coloring is valid w.r.t.
            //    WORLD SLOTS (no two same-color contacts share a dynamic world-slot
            //    body). Each awake dynamic world slot maps to a UNIQUE awake index
            //    (AwakeIndexOf is a bijection), so world-slot-disjoint <=> awake-index-
            //    disjoint -> the coloring is also valid for the dense awake-index space
            //    PackLane/ScatterBody rely on. A color's EMITTED subset (awake +
            //    touching + solver-relevant) is a SUBSET of its persistent members, so
            //    it is still a valid (disjoint) coloring.
            for (auto& bucket : m_colorRefs) { bucket.clear(); }
            m_overflowRefs.clear();
            for (std::uint32_t c = 0; c < ctx.contactCount; ++c)
            {
                const std::uint8_t col = ctx.contacts[c].color;
                if (col < kColorCount) { m_colorRefs[col].push_back(c); }
                else                   { m_overflowRefs.push_back(c); }
            }

#ifndef NDEBUG
            // Debug-only: PROVE the active emitted subset of the persistent coloring is
            // still a valid coloring -- within each color no awake-dynamic body index
            // appears twice. This is WHY grouping-by-persistent-color is scatter-safe:
            // PackLane/ScatterBody require no two constraints in a lane-wide batch to
            // share a dynamic awake-index. A is always an awake dynamic; B is dynamic
            // iff it is a real body with positive inverse mass (statics/kinematics/spans
            // are read-only, never block a color). Reuses awakeCount as the index bound.
            {
                std::vector<std::uint8_t> seen(static_cast<std::size_t>(awakeCount), 0u);
                for (std::uint32_t k = 0; k < static_cast<std::uint32_t>(kColorCount); ++k)
                {
                    std::fill(seen.begin(), seen.end(), 0u);
                    for (const std::uint32_t ref : m_colorRefs[k])
                    {
                        const ContactConstraint& cc = ctx.contacts[ref];
                        const std::uint32_t ia = w.AwakeIndexOf(cc.bodyA);
                        assert(ia < awakeCount && seen[ia] == 0u &&
                               "within-color clash: awake-dynamic A twice in one color");
                        seen[ia] = 1u;
                        if (cc.bodyBIsBody && cc.invMassB > Real(0))
                        {
                            const std::uint32_t ib = w.AwakeIndexOf(cc.bodyB);
                            assert(ib < awakeCount && seen[ib] == 0u &&
                                   "within-color clash: awake-dynamic B twice in one color");
                            seen[ib] = 1u;
                        }
                    }
                }
            }
#endif

            // 4) Build one SoA batch list per color (warm-start already on the
            //    prepared ctx.contacts points). Build re-homes each body index onto
            //    the dense solverIndex space via the world's awake/kinematic index
            //    maps; padding + static/span B lanes pack kNullBodyIndex (null-index
            //    branch: zero-identity gather, no scatter), and a kinematic B reads
            //    its real dense row.
            m_colorBatches.assign(static_cast<std::size_t>(kColorCount), {});
            for (std::size_t k = 0; k < static_cast<std::size_t>(kColorCount); ++k)
            {
                const std::vector<std::uint32_t>& color = m_colorRefs[k];
                if (color.empty()) { continue; }
                m_colorBatches[k] = ContactConstraintSimd::Build(
                    ctx.contacts, color.data(), static_cast<int>(color.size()),
                    w.AwakeIndexData(), w.KinematicIndexData(), awakeCount,
                    kNotKinematic);
            }

            // 5) Build the flat per-step stage list and run the WHOLE substep loop
            //    inside ONE persistent worker region (Gap 1.2) -- replacing the ~190
            //    per-color ParallelFor dispatches/step (the D1 dispatch storm) with a
            //    SINGLE ParallelFor(workerCount, 1, ...) whose body is the stage walk +
            //    atomic stage-sync barriers + ring-CAS block stealing. The worker index
            //    is the partition start `begin` (Box2D b2SolverTask style): begin == 0
            //    is the main/orchestrator (drives the stages, runs overflow/joints
            //    main-serial inline) and every begin > 0 is a thief that steals blocks.
            //    The stage walk reproduces D1's EXACT order per substep --
            //      integrate-vel -> warm-start[colors] -> (overflow) -> [joints] ->
            //      solve[colors](bias) -> (overflow) -> integrate-pos -> [joints] ->
            //      relax[colors] -> (overflow)
            //    then ONCE: restitution[colors] -> (overflow) -> store[colors] -- so
            //    the result is BYTE-IDENTICAL to the per-color loop it replaces AT ANY
            //    WORKER COUNT (serial == enki(1) == enki(N)): within a color the batches
            //    are body-disjoint, so the block partition -- hence the worker count --
            //    is order-independent; color boundaries are hard barriers, so the
            //    Gauss-Seidel order is preserved. Overflow + joints run MAIN-SERIAL
            //    inline between stages (the existing scalar methods, bound as
            //    FunctionRef lambdas below; thieves spin while the main runs them).
            const std::uint32_t workerCount = ctx.executor->WorkerCount();
            const int activeColors = BuildStages(awakeCount, workerCount);

            SolverStageContext sc;
            sc.ctx               = &ctx;
            sc.stages            = m_stages.data();
            sc.substepCount      = substeps;
            sc.activeColorStages = activeColors;
            sc.stageSync.store(0, std::memory_order_relaxed);
            sc.colorBatches      = &m_colorBatches;
            sc.colorRefs         = m_colorRefs.data();
            sc.bodyState         = m_bodyState.data();
            sc.h                 = static_cast<float>(h);
            sc.maxBiasVel        = maxBiasVel;
            sc.threshold         = threshold;

            // Bind the body-integrate range + the main-serial overflow/joint passes as
            // non-escaping lambdas (FunctionRef views -- they MUST outlive SolverWorker,
            // which they do: locals of this scope). The joint bridge no-ops when there
            // are no joints (keeps the pure-SoA hot path branch-free of joint work).
            auto integrateVelFn = [&](std::size_t b, std::size_t e) { IntegrateVelocitiesRange(ctx, h, b, e); };
            auto integratePosFn = [&](std::size_t b, std::size_t e) { IntegratePositionsRange(ctx, h, b, e); };
            auto overflowWarmStartFn   = [&]() { OverflowWarmStart(ctx); };
            auto overflowSolveFn       = [&](bool useBias) { OverflowSolve(ctx, h, useBias); };
            auto overflowRestitutionFn = [&]() { OverflowRestitution(ctx); };
            auto jointBridgeFn = [&]()
            {
                if (hasJoints) { SyncVelToWorld(ctx); SolveJoints(ctx); SyncVelFromWorld(ctx); }
            };
            sc.integrateVel        = integrateVelFn;
            sc.integratePos        = integratePosFn;
            sc.overflowWarmStart   = overflowWarmStartFn;
            sc.overflowSolve       = overflowSolveFn;
            sc.overflowRestitution = overflowRestitutionFn;
            sc.jointBridge         = jointBridgeFn;

            // Dispatch the persistent region ONCE. Worker index = the partition start
            // `begin` (unique per partition; begin == 0 always present -> exactly one
            // main). ParallelFor BLOCKS until every partition's SolverWorker returns:
            // the main returns after publishing the terminal sentinel, and every thief
            // returns the moment it observes that sentinel -> the join always completes.
            // workerCount == 1 (SerialWorkScheduler or JobSystem(1)) -> a single
            // partition fn(0,1,0) -> main only -> the deterministic serial reference.
            ctx.executor->ParallelFor(workerCount, /*minBatch=*/1,
                [&](std::size_t begin, std::size_t /*end*/, std::uint32_t /*enkiThread*/)
                {
                    SolverWorker(sc, static_cast<std::uint32_t>(begin));
                });

            // 8) Push final velocities to the world (DENSE write-back), then commit
            //    positions (compound-COM) from the SoA dp/dq.
            m_bodyState.SyncOutCompacted(w);
            FinalizePositionsSoA(ctx);
        }

    } // namespace Physics
} // namespace Manifold2D
