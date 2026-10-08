// PhysicsWorld.cpp -- the 2D-physics orchestration core, KINEMATIC subset
// (port of the kinematic path of PhysicsWorld.lua).
//
// See PhysicsWorld.hpp for the contract + the PORT BOUNDARY (what P1.8 ports
// vs what P2/P3 add). This TU implements Create/AddBody/RemoveBody/IsValid,
// the kinematic Step (prev snapshot + kinematic integration + broadphase
// update, then the event arrays), QueryAABB, the world event gate
// (SetEventsEnabled), and _staticCandidates.
//
// PRESENTATION-FREE + C++20-clean: Geometry::Vec2 + std + sibling Physics headers only.

// MOSAIC_LOG_CATEGORY must be defined before the FIRST transitive include of
// Mosaic/Log.hpp in this TU -- PhysicsWorld.hpp itself pulls it in (via
// Contact.hpp -> Mosaic/Assert.hpp -> Mosaic/Log.hpp), so this has to lead,
// ahead of even this TU's own header.
#define MOSAIC_LOG_CATEGORY "Manifold2D.World"
#include <Mosaic/Log.hpp>

#include <Manifold2D/Physics/PhysicsWorld.hpp>

#include <algorithm>
#include <Mosaic/Assert.hpp>
#include <cmath>

#include <Manifold2D/Physics/Body.hpp>
#include <Manifold2D/Physics/Broadphase/DynamicTree.hpp>
#include <Manifold2D/Physics/Broadphase/SpatialHash.hpp>
#include <Manifold2D/Physics/Broadphase/SweepAndPrune.hpp>
#include <Manifold2D/Physics/Narrowphase/GeometryKernel.hpp> // AabbOverlap (QueryAABB)
#include <Manifold2D/Physics/Narrowphase/Gjk.hpp>          // ShapeCast / ShapeCastPoly (bullet CCD)
#include <Manifold2D/Physics/Narrowphase/Collide.hpp>        // Collide (rotation-aware fixture-pair narrowphase: contacts T5, events/overlap T7)
#include <Manifold2D/Physics/Solver/SoftStep.hpp>            // SoftStep solver impl (THE solver)
#include <Manifold2D/Physics/Island.hpp>                     // island sleep pass (stage 4)
#include <Manifold2D/Physics/Joints/Joints.hpp>              // joint set + MakeJoint factory (P2.5)
#include <Manifold2D/Physics/StepProf.hpp>                   // opt-in per-Step-phase timing (zero-cost when ARCANE_STEPPROF==0)

namespace Manifold2D
{
    namespace Physics
    {

        // Compose a fixture's WORLD transform = body transform ∘ fixture local
        // transform.  SlotAabb, the contact narrowphase,
        // the fixture-aware queries, and BulletSweep all delegate
        // here so the rotate+offset formula lives in exactly ONE place. Promoted
        // from a TU-local static to a static member (T7) so Queries.cpp (a
        // separate TU) can compose fixture world transforms too.
        Transform PhysicsWorld::ComposeFixtureXf(Vec2 bodyPos, Real bodyAngle,
                                                 Vec2 localPos,
                                                 Real localAngle) noexcept
        {
            const Real bc = std::cos(bodyAngle);
            const Real bs = std::sin(bodyAngle);
            return Transform{
                Vec2(bodyPos.x + bc * localPos.x - bs * localPos.y,
                     bodyPos.y + bs * localPos.x + bc * localPos.y),
                bodyAngle + localAngle
            };
        }

        namespace
        {
            // Box2D v3 b2ComputeShapeExtent's minExtent: the smallest distance from
            // the shape's centroid to its surface (a polygon's nearest edge plane
            // plus its rounding radius; a circle's or capsule's radius).
            Real ShapeMinExtent(const Shape& s) noexcept
            {
                if (s.kind == ShapeKind::Aabb)
                {
                    return std::min(s.halfW, s.halfH) + s.radius;
                }
                if (s.kind == ShapeKind::Polygon && !s.normals.empty() && s.normals.size() == s.verts.size())
                {
                    Real m = Real(1e30);
                    for (std::size_t i = 0; i < s.verts.size(); ++i)
                    {
                        const Vec2 d(s.verts[i].x - s.polyCentroid.x, s.verts[i].y - s.polyCentroid.y);
                        m = std::min(m, s.normals[i].x * d.x + s.normals[i].y * d.y);
                    }
                    return std::max(Real(0), m) + s.radius;
                }
                return s.radius;
            }

            std::unique_ptr<IBroadphase> MakeBroadphase(const WorldDef& def)
            {
                // PORT + MODERNIZE: the Lua selected by string, defaulting to
                // SpatialHash; we default to DynamicTree (the P1.6 decision).
                switch (def.broadphase)
                {
                case BroadphaseKind::Hash:
                    return std::make_unique<SpatialHash>(def.hashCellSize);
                case BroadphaseKind::Sap:
                    return std::make_unique<SweepAndPrune>();
                case BroadphaseKind::Tree:
                default:
                    return std::make_unique<DynamicTree>();
                }
            }

            // Step 4 (Task B2): reject a degenerate collider (zero or
            // non-finite extent) at the AddFixture boundary. A zero-extent
            // shape silently registers a zero-area hitbox (finite, so no
            // downstream assert -- just a fixture that can never collide); a
            // non-finite one (e.g. NaN radius) produces a non-finite AABB that
            // trips DynamicTree::Update's finite-AABB assert. Both are input
            // errors, not simulation states, so they belong at the boundary,
            // not left to propagate.
            bool IsDegenerateShape(const Shape& s) noexcept
            {
                switch (s.kind)
                {
                    case ShapeKind::Circle:
                        return !std::isfinite(s.radius) || s.radius <= Real(0);
                    case ShapeKind::Capsule:
                        return !std::isfinite(s.radius) || s.radius <= Real(0) ||
                               !std::isfinite(s.halfLen) || s.halfLen < Real(0);
                    case ShapeKind::Aabb:
                        return !std::isfinite(s.halfW) || !std::isfinite(s.halfH) ||
                               s.halfW <= Real(0) || s.halfH <= Real(0);
                    case ShapeKind::Polygon:
                        if (s.verts.size() < 3)
                        {
                            return true;
                        }
                        for (const Vec2& v : s.verts)
                        {
                            if (!std::isfinite(v.x) || !std::isfinite(v.y))
                            {
                                return true;
                            }
                        }
                        return false;
                }
                return true; // unreachable; defensive
            }

            std::unique_ptr<ISolver> MakeSolver(const WorldDef& def)
            {
                // SoftStep is THE solver -- the P2.2 TGS Soft (Box2D-v3) impl.
                // The P2.3 Baumgarte A/B oracle was retired 2026-07-03 (MKS P2):
                // nothing selected it in production and SoftStep's own coverage
                // exceeds it. The ISolver seam stays for a future solver.
                (void)def;
                return std::make_unique<SoftStep>();
            }
        } // namespace

        PhysicsWorld::PhysicsWorld(const WorldDef& def)
            : m_fixtureBroadphase(MakeBroadphase(def))
            , m_gravityX(def.gravityX)
            , m_gravityY(def.gravityY)
            , m_substepCount(def.substepCount > 0u ? def.substepCount : 1u)
            , m_contactHertz(def.contactHertz)
            , m_contactDampingRatio(def.contactDampingRatio)
            , m_restitutionThreshold(def.restitutionThreshold)
            , m_contactPushMaxVelocity(def.contactPushMaxVelocity)
            , m_enableContinuous(def.enableContinuous)
            , m_maxLinearVelocity(def.maxLinearVelocity)
            , m_sleepThresholdDefault(def.sleepThreshold)
            , m_hitEventThreshold(def.hitEventThreshold)
            , m_contactEventsRequireBoth(def.contactEventsRequireBoth)
            , m_solver(MakeSolver(def))
        {
            // Optional tile statics: own a TileGrid over the passability seam if
            // one was provided (ports the Lua `tileGrid = map and TileGrid.new`).
            if (def.passability != nullptr)
            {
                m_tileGrid = std::make_unique<TileGrid>(*def.passability,
                                                        def.tileCellSize,
                                                        def.tileOrigin);
            }

            // Phase C, Task 4: the per-color contact-id lists are sized once in the
            // ConstraintGraph ctor (decomp step 2); the per-body color mask grows
            // lazily with the body SoA via m_graph.Grow in EnsureCapacity.

            // One-line world-config banner (the only INFO site in this TU --
            // everything else stays at WARN/DEBUG/TRACE or silent). Plain
            // literal: Mosaic's message seam is std::string_view with no
            // formatting layer, so a config-value banner is intentionally NOT
            // composed here.
            MOSAIC_LOG_INFO("PhysicsWorld created: SoftStep solver, MKS units");
        }

        PhysicsWorld::~PhysicsWorld() = default;

        void PhysicsWorld::EnsureCapacity(std::uint32_t n)
        {
            if (n <= m_posX.size())
            {
                return;
            }
            MOSAIC_LOG_DEBUG("storage grew");
            // Amortized growth: at least double the current capacity so that
            // a one-at-a-time fill (AddBody(count+1)) does not realloc all ~12
            // SoA vectors on every add.  Free-list / steady-state behaviour is
            // unchanged; only the initial fill is faster.
            const std::uint32_t next =
                std::max(n, static_cast<std::uint32_t>(m_posX.capacity() * 2));
            m_posX.resize(next);
            m_posY.resize(next);
            m_prevX.resize(next);
            m_prevY.resize(next);
            m_prevAngle.resize(next, Real(0));
            m_velX.resize(next);
            m_velY.resize(next);
            m_btype.resize(next);
            m_alive.resize(next);
            m_sensor.resize(next);
            m_gen.resize(next, 0u); // gen kept zero-filled (live slots start at 1)
            m_shape.resize(next);

            // Dynamics SoA (P2.1). Reals default-zero (no inverse mass/inertia ->
            // immovable like a static until AddBody fills a Dynamic slot); awake
            // defaults to 1 so an un-filled slot is never treated as a sleeper.
            // AddBody overwrites every field for the slot it returns, so these
            // fills only matter for the never-touched tail; keeping them
            // self-consistent avoids surprises if a future path scans the SoA.
            m_angle.resize(next, Real(0));
            m_angVel.resize(next, Real(0));
            m_invMass.resize(next, Real(0));
            m_invInertia.resize(next, Real(0));
            m_rest.resize(next, Real(0));
            m_fric.resize(next, Real(0));
            m_linDamp.resize(next, Real(0));
            m_gravityScale.resize(next, Real(1));
            m_sleepTimer.resize(next, Real(0));
            m_maxExtent.resize(next, Real(0));
            m_sleepThreshold.resize(next, m_sleepThresholdDefault);
            m_awake.resize(next, std::uint8_t(1));
            m_bullet.resize(next, std::uint8_t(0));
            m_minExtent.resize(next, Real(0));
            m_safetyFactor.resize(next, Real(0.5));
            // Island topology columns (m_islandId + SplitIsland scratch) now live in
            // IslandManager (decomp step 1); grow them through the seam.
            m_islandMgr.Grow(next);
            // Phase B: awake-set back-index column. kNotAwake = not in the set.
            // A tail slot is never in the set until AddBody writes it.
            m_awakeIndex.resize(next, kNotAwake);
            // Phase C: kinematic-set back-index column. kNotKinematic = not in the
            // set. A tail slot is never in the set until AddBody writes it.
            m_kinematicIndex.resize(next, kNotKinematic);
            // Phase C, Task 4: per-body color-occupancy bitmask -- now owned by
            // ConstraintGraph (decomp step 2); grow it through the seam. A fresh/
            // recycled slot starts with NO colors occupied (the RemoveBody
            // leak-detector asserts a removed body left mask 0).
            m_graph.Grow(next);
        }

        // ----------------------------------------------------------------
        // Fixture SoA helpers (v2 Task 4)
        // ----------------------------------------------------------------

        void PhysicsWorld::EnsureFxCapacity(std::uint32_t n)
        {
            if (n <= m_fxShape.size())
            {
                return;
            }
            MOSAIC_LOG_DEBUG("storage grew");
            // Amortised growth: at least double the current capacity.
            const std::uint32_t next =
                std::max(n, static_cast<std::uint32_t>(
                    m_fxShape.empty() ? 4u : m_fxShape.capacity() * 2u));

            m_fxShape.resize(next);
            m_fxLocalPosX.resize(next, Real(0));
            m_fxLocalPosY.resize(next, Real(0));
            m_fxLocalAngle.resize(next, Real(0));
            m_fxDensity.resize(next, Real(0));
            m_fxFriction.resize(next, Real(0));
            m_fxRestitution.resize(next, Real(0));
            m_fxFilterCat.resize(next, 1u);
            m_fxFilterMask.resize(next, 0xFFFFFFFFu);
            m_fxFilterGroup.resize(next, 0);
            m_fxSensor.resize(next, std::uint8_t(0));
            m_fxContactEvents.resize(next, std::uint8_t(0));
            m_fxSensorEvents.resize(next, std::uint8_t(0));
            m_fxHitEvents.resize(next, std::uint8_t(0));
            m_fxBody.resize(next, 0u);
            m_fxGen.resize(next, 0u); // 0 = dead; live starts at 1
        }

        void PhysicsWorld::EnsureBodyAuxCapacity(std::uint32_t n)
        {
            if (n <= m_bodyFixtures.size())
            {
                return;
            }
            MOSAIC_LOG_DEBUG("storage grew");
            // Grow to match body SoA capacity.
            const std::uint32_t next =
                std::max(n, static_cast<std::uint32_t>(
                    m_bodyFixtures.empty() ? 4u : m_bodyFixtures.capacity() * 2u));

            m_bodyFixtures.resize(next);
            m_islandMgr.GrowBodyContacts(next); // m_bodyContacts now IslandManager-owned
            m_localCenterX.resize(next, Real(0));
            m_localCenterY.resize(next, Real(0));
            m_bodyMass.resize(next, Real(0));
            m_bodyInertia.resize(next, Real(0));
            m_fixedRotation.resize(next, std::uint8_t(0));
            m_massOverride.resize(next, Real(0));
        }

        void PhysicsWorld::RecomputeBodyMass(std::uint32_t bodySlot)
        {
            // Static and Kinematic bodies never integrate or solve; keep their
            // inverse mass/inertia at zero (the existing convention).
            const BodyType bt = static_cast<BodyType>(m_btype[bodySlot]);
            if (bt != BodyType::Dynamic)
            {
                m_invMass[bodySlot]    = Real(0);
                m_invInertia[bodySlot] = Real(0);
                m_bodyMass[bodySlot]   = Real(0);
                m_bodyInertia[bodySlot]= Real(0);
                m_localCenterX[bodySlot] = Real(0);
                m_localCenterY[bodySlot] = Real(0);
                m_maxExtent[bodySlot]    = Real(0);
                return;
            }

            // Accumulate mass, density-weighted centroid, and inertia over
            // all fixtures attached to this body slot.
            //
            // Fix 2: collect each fixture's MassData ONCE into a local cache so
            // the two-pass aggregation (centroid + parallel-axis inertia) reads
            // each ComputeMass result from the cache rather than recomputing it.
            // This produces identical aggregation values + identical order; only
            // the per-fixture ComputeMass call is deduplicated.
            const std::vector<std::uint32_t>& fxList = m_bodyFixtures[bodySlot];

            // Cache: parallel arrays of (bodyLocalX, bodyLocalY, massData) for
            // each contributing fixture.  Stack-allocated vector (setup-time only).
            struct FxEntry
            {
                Real      bx, by;   // centroid in body-local space
                MassData  md;
            };
            std::vector<FxEntry> entries;
            entries.reserve(fxList.size());

            Real totalMass = Real(0);
            // Weighted centroid numerator (body-local space).
            Real cx = Real(0);
            Real cy = Real(0);

            // Single pass: compute each fixture's MassData once, accumulate mass
            // and density-weighted centroid, and store for the inertia pass.
            for (const std::uint32_t fi : fxList)
            {
                if (m_fxGen[fi] == 0u)
                {
                    continue; // dead slot in the list (should not happen; defensive)
                }
                const MassData md = m_fxShape[fi].ComputeMass(m_fxDensity[fi]);
                if (md.mass <= Real(0))
                {
                    continue; // degenerate / zero-density fixture
                }

                // The fixture's centroid in body-local space:
                //   bodyLocalCentroid = localPos + R(localAngle) * shapeCentroid
                // For our analytic test cases the shape centroid is (0,0) for
                // both circle and aabb, so localPos is the body-local centroid.
                // The general form is computed here for correctness.
                const Real la  = m_fxLocalAngle[fi];
                const Real lc  = std::cos(la);
                const Real ls  = std::sin(la);
                const Real scx = md.centroid.x;
                const Real scy = md.centroid.y;
                // Rotate shape centroid by localAngle, then offset by localPos.
                const Real bx = m_fxLocalPosX[fi] + lc * scx - ls * scy;
                const Real by = m_fxLocalPosY[fi] + ls * scx + lc * scy;

                totalMass += md.mass;
                cx += md.mass * bx;
                cy += md.mass * by;

                entries.push_back(FxEntry{ bx, by, md });
            }

            if (totalMass <= Real(0))
            {
                // No contributing fixtures (all degenerate).
                m_invMass[bodySlot]      = Real(0);
                m_invInertia[bodySlot]   = Real(0);
                m_bodyMass[bodySlot]     = Real(0);
                m_bodyInertia[bodySlot]  = Real(0);
                m_localCenterX[bodySlot] = Real(0);
                m_localCenterY[bodySlot] = Real(0);
                m_maxExtent[bodySlot]    = Real(0);
                return;
            }

            // Body center of mass in body-local frame.
            const Real comX = cx / totalMass;
            const Real comY = cy / totalMass;

            // Second pass: sum inertia about the COM via parallel-axis theorem
            // using the cached MassData (no second ComputeMass call per fixture).
            //   I_total = sum_i (I_i + m_i * |centroid_i - COM|^2)
            Real totalInertia = Real(0);
            for (const FxEntry& e : entries)
            {
                // Parallel-axis shift: distance^2 from fixture centroid to COM.
                const Real dx = e.bx - comX;
                const Real dy = e.by - comY;
                totalInertia += e.md.inertia + e.md.mass * (dx * dx + dy * dy);
            }

            // Fix 1: if the body has a mass override (set at AddBody when
            // def.mass > 0), apply it now: replace the density-derived totalMass
            // and scale totalInertia proportionally, exactly mirroring the
            // AddBody override semantics (scale = override / computedMass).
            // This preserves the override through subsequent AddFixture calls so
            // compound bodies with an explicit mass keep the right dynamics.
            // Static/Kinematic never reach this branch (early return above).
            const Real massOvr = (bodySlot < m_massOverride.size())
                                     ? m_massOverride[bodySlot]
                                     : Real(0);
            if (massOvr > Real(0) && totalMass > Real(0))
            {
                const Real scale = massOvr / totalMass;
                totalInertia    *= scale;
                totalMass        = massOvr;
            }

            // Store the aggregated body state.
            m_bodyMass[bodySlot]     = totalMass;
            m_localCenterX[bodySlot] = comX;
            m_localCenterY[bodySlot] = comY;
            m_bodyInertia[bodySlot]  = totalInertia;

            m_invMass[bodySlot]    = Real(1) / totalMass;
            const bool fixedRot    = m_fixedRotation[bodySlot] != 0u;
            m_invInertia[bodySlot] = (fixedRot || totalInertia <= Real(0))
                                         ? Real(0)
                                         : Real(1) / totalInertia;

            // COM is now final -> refresh the cached sleep-test extent.
            RecomputeMaxExtent(bodySlot);
        }

        void PhysicsWorld::RecomputeMaxExtent(std::uint32_t bodySlot)
        {
            // maxExtent = max over fixtures of ( max over core verts of
            // |vert_bodyLocal - COM| + shape.radius ). Body-local frame; COM is
            // m_localCenter. This is Box2D's sim->maxExtent, used by the sleep
            // velocity test ( |v| + |w|*maxExtent ). 0 for a body with no fixtures.
            Real maxExt = Real(0);
            Real minExt = Real(0);
            bool haveMin = false;
            const Real comX = m_localCenterX[bodySlot];
            const Real comY = m_localCenterY[bodySlot];
            for (const std::uint32_t fi : m_bodyFixtures[bodySlot])
            {
                if (fi >= m_fxCount || m_fxGen[fi] == 0u)
                {
                    continue; // dead slot (defensive)
                }
                const Shape& s = m_fxShape[fi];
                // minExtent (Box2D b2ComputeShapeExtent): the shape's smallest
                // centroid-to-surface distance; the body's is the min over shapes.
                const Real me = ShapeMinExtent(s);
                minExt = haveMin ? std::min(minExt, me) : me;
                haveMin = true;
                const Real la  = m_fxLocalAngle[fi];
                const Real lc  = std::cos(la), ls = std::sin(la);
                const Real lpx = m_fxLocalPosX[fi], lpy = m_fxLocalPosY[fi];
                for (const Vec2& v : s.verts)
                {
                    // Core vert in body-local frame: localPos + R(localAngle)*v.
                    const Real px = lpx + lc * v.x - ls * v.y;
                    const Real py = lpy + ls * v.x + lc * v.y;
                    const Real dx = px - comX, dy = py - comY;
                    const Real d  = std::sqrt(dx * dx + dy * dy) + s.radius;
                    if (d > maxExt) { maxExt = d; }
                }
            }
            m_maxExtent[bodySlot] = maxExt;
            m_minExtent[bodySlot] = minExt;
        }

        // ----------------------------------------------------------------
        // Fixture lifecycle: AddFixture / DropFixture / IsValid(FixtureHandle)
        // ----------------------------------------------------------------

        std::uint32_t PhysicsWorld::AllocFixtureSlot(std::uint32_t bodySlot,
                                                          const FixtureDef& def)
        {
            // Ensure per-body auxiliary arrays cover this body slot.
            EnsureBodyAuxCapacity(bodySlot + 1u);

            // Acquire a fixture slot (reuse from free-list or append).
            std::uint32_t fi;
            if (!m_fxFree.empty())
            {
                fi = m_fxFree.back();
                m_fxFree.pop_back();
            }
            else
            {
                fi = m_fxCount;
                EnsureFxCapacity(m_fxCount + 1u);
                ++m_fxCount;
            }

            // Populate the fixture slot fields.
            m_fxShape[fi]      = def.shape;
            m_fxLocalPosX[fi]  = def.localPos.x;
            m_fxLocalPosY[fi]  = def.localPos.y;
            m_fxLocalAngle[fi] = def.localAngle;
            m_fxDensity[fi]    = def.density;
            m_fxFriction[fi]   = def.friction;
            m_fxRestitution[fi]= def.restitution;
            m_fxFilterCat[fi]  = def.categoryBits;
            m_fxFilterMask[fi] = def.maskBits;
            m_fxFilterGroup[fi] = def.groupIndex;
            m_fxSensor[fi]     = def.isSensor ? std::uint8_t(1) : std::uint8_t(0);
            m_fxContactEvents[fi] = def.contactEvents ? std::uint8_t(1) : std::uint8_t(0);
            m_fxSensorEvents[fi]  = def.sensorEvents  ? std::uint8_t(1) : std::uint8_t(0);
            m_fxHitEvents[fi]     = def.hitEvents     ? std::uint8_t(1) : std::uint8_t(0);
            m_fxBody[fi]       = bodySlot;
            m_fxGen[fi]       += 1u; // bump generation (dead=0, live starts at 1)

            // Link the slot to the body.
            m_bodyFixtures[bodySlot].push_back(fi);

            return fi;
        }

        // AddFixture: compound shapes + mass-override now compose correctly.
        // After allocating and linking the fixture slot, RecomputeBodyMass
        // re-aggregates the body mass from all fixtures (respecting any
        // m_massOverride set at AddBody), so a previously set mass override
        // is preserved rather than silently discarded.
        FixtureHandle PhysicsWorld::AddFixture(BodyHandle bh, const FixtureDef& def)
        {
            if (!IsValid(bh))
            {
                return kInvalidFixture;
            }
            if (IsDegenerateShape(def.shape))
            {
                MOSAIC_LOG_WARN("rejected degenerate AddFixture shape");
                return kInvalidFixture;
            }
            const std::uint32_t bodySlot = bh.index;

            const std::uint32_t fi = AllocFixtureSlot(bodySlot, def);

            // Re-aggregate the body's mass / COM / inertia (mass override respected).
            RecomputeBodyMass(bodySlot);

            // A fixture added to a STATIC body grows its union AABB; movers
            // self-correct via the per-step broadphase Update, but statics never
            // re-register, so refresh the static tree here.
            if (static_cast<BodyType>(m_btype[bodySlot]) == BodyType::Static)
                m_staticTree.Update(bodySlot, SlotAabb(bodySlot));

            // Register the new fixture in the per-fixture mover broadphase
            // (Phase 2, Task 1). AddFixtureProxy skips Static bodies.
            AddFixtureProxy(fi);

            MOSAIC_LOG_TRACE("fixture added");
            return FixtureHandle{ fi, m_fxGen[fi] };
        }

        void PhysicsWorld::DropFixture(FixtureHandle fh)
        {
            if (!IsValid(fh))
            {
                MOSAIC_LOG_WARN("operation on a stale/invalid FixtureHandle ignored");
                return;
            }
            MOSAIC_LOG_TRACE("fixture removed");
            const std::uint32_t fi       = fh.index;
            const std::uint32_t bodySlot = m_fxBody[fi];

            // Unlink from the body's fixture list (swap-and-pop).
            std::vector<std::uint32_t>& list = m_bodyFixtures[bodySlot];
            for (std::size_t k = 0; k < list.size(); ++k)
            {
                if (list[k] == fi)
                {
                    list[k] = list.back();
                    list.pop_back();
                    break;
                }
            }

            // Remove the fixture proxy from the per-fixture mover broadphase
            // BEFORE bumping the generation (Phase 2, Task 1). The Remove call
            // only needs the slot id -- it does not read gen or body fields.
            RemoveFixtureProxy(fi);

            // Invalidate the slot (bump generation + clear shape storage).
            m_fxGen[fi] += 1u;
            m_fxShape[fi] = Shape{}; // release polygon vertex/normal storage

            // Immediately destroy any persistent contacts on this fixture (Task 5)
            // so a recycled slot never leaves a stale contact for the single Step
            // before UpdateContacts' guard would reap it. The walk matches by slot
            // index, so it runs correctly after the generation bump above.
            m_graph.DestroyContactsForFixture(*this, fi);

            // Recycle.
            m_fxFree.push_back(fi);

            // Re-aggregate the body's mass.
            RecomputeBodyMass(bodySlot);

            // Symmetric with AddFixture: dropping a fixture from a STATIC shrinks
            // its union AABB. Statics never re-register via Step, so refresh the
            // static tree here to keep its AABB tight (movers self-correct each step).
            if (static_cast<BodyType>(m_btype[bodySlot]) == BodyType::Static)
                m_staticTree.Update(bodySlot, SlotAabb(bodySlot));
        }

        bool PhysicsWorld::IsValid(FixtureHandle fh) const noexcept
        {
            if (fh.generation == 0u)
            {
                return false;
            }
            if (fh.index >= m_fxCount)
            {
                return false;
            }
            return m_fxGen[fh.index] == fh.generation;
        }

        std::uint32_t PhysicsWorld::FixtureCount(BodyHandle bh) const noexcept
        {
            if (!IsValid(bh))
            {
                return 0u;
            }
            const std::uint32_t bodySlot = bh.index;
            if (bodySlot >= m_bodyFixtures.size())
            {
                return 0u;
            }
            return static_cast<std::uint32_t>(m_bodyFixtures[bodySlot].size());
        }

        Vec2 PhysicsWorld::GetFixtureWorldPos(FixtureHandle fh) const noexcept
        {
            if (!IsValid(fh))
            {
                return Vec2(Real(0), Real(0));
            }
            const std::uint32_t fi       = fh.index;
            const std::uint32_t bodySlot = m_fxBody[fi];

            const Real bodyAngle = m_angle[bodySlot];
            const Real c = std::cos(bodyAngle);
            const Real s = std::sin(bodyAngle);
            const Real lx = m_fxLocalPosX[fi];
            const Real ly = m_fxLocalPosY[fi];

            return Vec2(
                m_posX[bodySlot] + c * lx - s * ly,
                m_posY[bodySlot] + s * lx + c * ly);
        }

        Real PhysicsWorld::GetFixtureWorldAngle(FixtureHandle fh) const noexcept
        {
            if (!IsValid(fh))
            {
                return Real(0);
            }
            const std::uint32_t fi       = fh.index;
            const std::uint32_t bodySlot = m_fxBody[fi];
            return m_angle[bodySlot] + m_fxLocalAngle[fi];
        }

        const Shape& PhysicsWorld::GetFixtureShape(FixtureHandle fh) const noexcept
        {
            // Function-local empty shape for stale/out-of-range handles: a
            // reference return has to name something, and a default Shape has
            // empty verts (the outline draws nothing for it).
            static const Shape kEmpty{};
            if (!IsValid(fh))
            {
                return kEmpty;
            }
            return m_fxShape[fh.index];
        }

        Vec2 PhysicsWorld::GetFixtureLocalPos(FixtureHandle fh) const noexcept
        {
            if (!IsValid(fh))
            {
                return Vec2(Real(0), Real(0));
            }
            const std::uint32_t fi = fh.index;
            return Vec2(m_fxLocalPosX[fi], m_fxLocalPosY[fi]);
        }

        Real PhysicsWorld::GetFixtureLocalAngle(FixtureHandle fh) const noexcept
        {
            if (!IsValid(fh))
            {
                return Real(0);
            }
            return m_fxLocalAngle[fh.index];
        }

        std::uint32_t PhysicsWorld::GetFixtureCategory(FixtureHandle fh) const noexcept
        {
            if (!IsValid(fh))
            {
                return 1u; // default
            }
            return m_fxFilterCat[fh.index];
        }

        std::uint32_t PhysicsWorld::GetFixtureMask(FixtureHandle fh) const noexcept
        {
            if (!IsValid(fh))
            {
                return 0xFFFFFFFFu; // default
            }
            return m_fxFilterMask[fh.index];
        }

        FixtureHandle PhysicsWorld::GetBodyFixture(BodyHandle bh,
                                                    std::uint32_t n) const noexcept
        {
            if (!IsValid(bh))
            {
                return kInvalidFixture;
            }
            const std::uint32_t bodySlot = bh.index;
            if (bodySlot >= m_bodyFixtures.size())
            {
                return kInvalidFixture;
            }
            const std::vector<std::uint32_t>& list = m_bodyFixtures[bodySlot];
            if (n >= static_cast<std::uint32_t>(list.size()))
            {
                return kInvalidFixture;
            }
            const std::uint32_t fi = list[n];
            if (fi >= m_fxCount || m_fxGen[fi] == 0u)
            {
                return kInvalidFixture;
            }
            return FixtureHandle{ fi, m_fxGen[fi] };
        }

        Real PhysicsWorld::GetBodyMass(BodyHandle bh) const noexcept
        {
            if (!IsValid(bh))
            {
                return Real(0);
            }
            const std::uint32_t bodySlot = bh.index;
            if (bodySlot >= m_bodyMass.size())
            {
                return Real(0);
            }
            return m_bodyMass[bodySlot];
        }

        Vec2 PhysicsWorld::GetLocalCenter(BodyHandle bh) const noexcept
        {
            if (!IsValid(bh))
            {
                return Vec2(Real(0), Real(0));
            }
            const std::uint32_t bodySlot = bh.index;
            if (bodySlot >= m_localCenterX.size())
            {
                return Vec2(Real(0), Real(0));
            }
            return Vec2(m_localCenterX[bodySlot], m_localCenterY[bodySlot]);
        }

        Real PhysicsWorld::GetBodyInertia(BodyHandle bh) const noexcept
        {
            if (!IsValid(bh))
            {
                return Real(0);
            }
            const std::uint32_t bodySlot = bh.index;
            if (bodySlot >= m_bodyInertia.size())
            {
                return Real(0);
            }
            return m_bodyInertia[bodySlot];
        }

        Aabb2 PhysicsWorld::SlotAabb(std::uint32_t i) const noexcept
        {
            // T5: body AABB = union of its fixtures' rotation-aware world AABBs.
            // Each fixture's world transform is composed from the body transform +
            // the fixture's local transform:
            //   worldAngle = bodyAngle + fxLocalAngle
            //   worldPos   = bodyPos + R(bodyAngle) * fxLocalPos
            // (via GetFixtureWorldPos / GetFixtureWorldAngle).
            //
            // The mover broadphase is PER-FIXTURE (m_fixtureBroadphase; one proxy
            // per live fixture of a Dynamic/Kinematic body). SlotAabb returns the
            // union over all fixtures and is used for the static grid + residency
            // grid only. A body with zero live fixtures falls back to the legacy
            // single-shape AABB (back-compat for any body that bypasses AddFixture).
            //
            // For a body with no fixture list or an empty fixture list, fall
            // back to the legacy single-shape path (same as pre-T5 behaviour).
            if (i < m_bodyFixtures.size() && !m_bodyFixtures[i].empty())
            {
                const Real bodyAngle = m_angle[i];
                const Vec2 bodyPos(m_posX[i], m_posY[i]);

                // Sentinel: empty AABB that we expand on first fixture.
                Aabb2 unionAabb;
                unionAabb.min = Vec2( Real(1e30f),  Real(1e30f));
                unionAabb.max = Vec2(-Real(1e30f), -Real(1e30f));

                for (const std::uint32_t fi : m_bodyFixtures[i])
                {
                    if (fi >= m_fxCount || m_fxGen[fi] == 0u)
                    {
                        continue; // dead slot (defensive)
                    }

                    // Compose body transform + fixture local transform (shared helper).
                    const Transform xf = ComposeFixtureXf(
                        bodyPos, bodyAngle,
                        Vec2(m_fxLocalPosX[fi], m_fxLocalPosY[fi]),
                        m_fxLocalAngle[fi]);
                    const Aabb2 fxAabb = m_fxShape[fi].ComputeAABB(xf);

                    // Expand the union AABB.
                    unionAabb.min.x = std::min(unionAabb.min.x, fxAabb.min.x);
                    unionAabb.min.y = std::min(unionAabb.min.y, fxAabb.min.y);
                    unionAabb.max.x = std::max(unionAabb.max.x, fxAabb.max.x);
                    unionAabb.max.y = std::max(unionAabb.max.y, fxAabb.max.y);
                }

                // If we actually expanded the sentinel (at least one live fixture),
                // return the union; otherwise fall through to the legacy path.
                if (unionAabb.min.x <= unionAabb.max.x)
                {
                    return unionAabb;
                }
            }

            // Legacy fallback: single-shape AABB with the body's real angle.
            // (Also correct for bodies that have fixtures but they all died.)
            const Transform xf{ Vec2(m_posX[i], m_posY[i]), m_angle[i] };
            return m_shape[i].ComputeAABB(xf);
        }

        // ----------------------------------------------------------------
        // Per-fixture broadphase helpers (Phase 2, Task 1)
        // ----------------------------------------------------------------

        Aabb2 PhysicsWorld::FixtureAabb(std::uint32_t fi) const noexcept
        {
            const std::uint32_t b = m_fxBody[fi];
            const Transform xf = ComposeFixtureXf(
                Vec2(m_posX[b], m_posY[b]), m_angle[b],
                Vec2(m_fxLocalPosX[fi], m_fxLocalPosY[fi]), m_fxLocalAngle[fi]);
            return m_fxShape[fi].ComputeAABB(xf);
        }

        void PhysicsWorld::AddFixtureProxy(std::uint32_t fi)
        {
            const std::uint32_t b = m_fxBody[fi];
            if (static_cast<BodyType>(m_btype[b]) == BodyType::Static)
            {
                return; // statics are not in the mover fixture broadphase
            }
            m_fixtureBroadphase->Update(fi, FixtureAabb(fi));
        }

        void PhysicsWorld::RemoveFixtureProxy(std::uint32_t fi)
        {
            m_fixtureBroadphase->Remove(fi);
        }

        void PhysicsWorld::UpdateMoverProxies(std::uint32_t b)
        {
            // Residency index (combat-sphere seam): keep body-union AABB in sync.
            const Aabb2 bodyBox = SlotAabb(b);
            m_residencyGrid.Move(b, bodyBox);

            // Per-fixture proxies: refresh every live fixture of this body.
            const Vec2 pos(m_posX[b], m_posY[b]);
            const Real ang = m_angle[b];
            if (b < m_bodyFixtures.size())
            {
                for (const std::uint32_t fi : m_bodyFixtures[b])
                {
                    if (fi >= m_fxCount || m_fxGen[fi] == 0u)
                    {
                        continue; // dead slot (defensive)
                    }
                    const Transform xf = ComposeFixtureXf(
                        pos, ang,
                        Vec2(m_fxLocalPosX[fi], m_fxLocalPosY[fi]),
                        m_fxLocalAngle[fi]);
                    m_fixtureBroadphase->Update(fi, m_fxShape[fi].ComputeAABB(xf));
                }
            }
        }

        void PhysicsWorld::LiveFixtureAabbs(std::vector<std::uint32_t>& fxOut,
                                            std::vector<Aabb2>& boxOut) const
        {
            fxOut.clear(); boxOut.clear();
            // Enumerate the AUTHORITATIVE live mover-fixture set: each live
            // Dynamic/Kinematic body's m_bodyFixtures list (exactly what the
            // fixture broadphase indexes; dropped fixtures are unlinked from it).
            for (std::uint32_t b = 0; b < m_count; ++b)
            {
                if (m_alive[b] == 0u) continue;
                if (static_cast<BodyType>(m_btype[b]) == BodyType::Static) continue;
                for (const std::uint32_t fi : m_bodyFixtures[b])
                {
                    if (fi >= m_fxCount || m_fxGen[fi] == 0u) continue; // defensive
                    fxOut.push_back(fi);
                    boxOut.push_back(FixtureAabb(fi));
                }
            }
        }

        // --------------------------------------------------------------------
        // DebugCollide -- re-run the REAL narrowphase on two fixtures + record a
        // NarrowphaseTrace (debug-viz Slice B inspector seam, Task 3).
        //
        // PURE: reads the SoA + fixture arrays, writes only `out`. Composes each
        // fixture's world transform with the SAME ComposeFixtureXf the Step
        // path's GenerateContacts/FixtureWorldXf uses (so the reproduced
        // manifold matches the Step's manifold exactly), Clears the trace, then
        // calls the unified Collide with a non-null recorder. Hard-contact (no
        // speculative margin), matching how the inspector re-runs a settled
        // contact. Stale/invalid handles -> empty manifold + Cleared trace.
        // --------------------------------------------------------------------
        Manifold PhysicsWorld::DebugCollide(FixtureHandle a, FixtureHandle b,
                                            NarrowphaseTrace& out) const
        {
            out.Clear();

            if (!IsValid(a) || !IsValid(b))
            {
                return Manifold{};
            }

            const std::uint32_t fa = a.index;
            const std::uint32_t fb = b.index;
            const std::uint32_t ba = m_fxBody[fa];
            const std::uint32_t bb = m_fxBody[fb];

            // Compose each fixture's world transform exactly as the Step path
            // does (ComposeFixtureXf == the one copy of the rotate+offset form).
            const Transform xfA = ComposeFixtureXf(
                Vec2(m_posX[ba], m_posY[ba]), m_angle[ba],
                Vec2(m_fxLocalPosX[fa], m_fxLocalPosY[fa]), m_fxLocalAngle[fa]);
            const Transform xfB = ComposeFixtureXf(
                Vec2(m_posX[bb], m_posY[bb]), m_angle[bb],
                Vec2(m_fxLocalPosX[fb], m_fxLocalPosY[fb]), m_fxLocalAngle[fb]);

            // Re-run the real narrowphase with the recorder attached. The Step
            // path passes NO trace to this same Collide (so it stays
            // byte-identical); here the &out recorder captures the intermediate
            // geometry while the returned manifold reproduces the Step's.
            return Collide(m_fxShape[fa], xfA, m_fxShape[fb], xfB,
                           /*speculativeMargin*/ Real(0), &out);
        }

        BodyHandle PhysicsWorld::AddBody(const BodyDef& def)
        {
            // Reject a non-finite position before it consumes a slot: it would
            // otherwise flow into SlotAabb/broadphase Update unchecked, which
            // either poisons the tree with a garbage finite-but-huge AABB (a
            // NaN position -- min/max accumulation silently drops NaN because
            // every NaN comparison is false) or trips DynamicTree::Update's
            // finite-AABB assert outright (an Inf position produces a genuinely
            // infinite AABB component). No slot is consumed either way.
            if (!std::isfinite(def.position.x) || !std::isfinite(def.position.y))
            {
                MOSAIC_LOG_WARN("rejected non-finite AddBody position");
                return kInvalidBody;
            }

            // Reuse a free slot or append (ports addBody's free-list pop).
            std::uint32_t idx;
            if (!m_free.empty())
            {
                idx = m_free.back();
                m_free.pop_back();
            }
            else
            {
                idx = m_count;
                EnsureCapacity(m_count + 1);
                ++m_count;
            }

            m_posX[idx]  = def.position.x;
            m_posY[idx]  = def.position.y;
            m_prevX[idx] = def.position.x;
            m_prevY[idx] = def.position.y;
            m_velX[idx]  = Real(0);
            m_velY[idx]  = Real(0);
            m_btype[idx] = static_cast<std::uint8_t>(def.type);
            m_sensor[idx] = def.isSensor ? std::uint8_t(1) : std::uint8_t(0);
            m_alive[idx]  = 1;
            // Bump generation (live slots start at 1; bump on BOTH add + remove
            // so stale handles never match). Ports gen[idx] = (gen[idx] or 0)+1.
            m_gen[idx] += 1u;
            m_shape[idx] = def.shape;

            // ---- dynamics state (P2.1) -- ports addBody (PhysicsWorld.lua:
            // 229-250). Static/Kinematic get the zero defaults (no inverse
            // mass/inertia -> never integrated/solved). A Dynamic body derives
            // mass + rotational inertia from Shape::ComputeMass(density).
            m_angle[idx]      = Real(0);
            m_prevAngle[idx]  = Real(0);
            m_angVel[idx]     = Real(0);
            m_invMass[idx]    = Real(0);
            m_invInertia[idx] = Real(0);
            m_rest[idx]       = def.restitution;
            m_fric[idx]       = def.friction;
            m_linDamp[idx]    = def.linearDamping;
            m_gravityScale[idx] = std::isfinite(def.gravityScale) ? def.gravityScale : Real(1);
            m_sleepTimer[idx] = Real(0);
            // Per-body sleep gate: explicit BodyDef override (>= 0) or inherit world.
            m_sleepThreshold[idx] = (def.sleepThreshold >= Real(0))
                                        ? def.sleepThreshold
                                        : m_sleepThresholdDefault;
            m_awake[idx]      = 1;
            m_bullet[idx]     = def.bullet ? std::uint8_t(1) : std::uint8_t(0);
            m_safetyFactor[idx] = def.safetyFactor;

            // ---- persistent island assignment (Phase A) ---------------------
            // A new DYNAMIC body is its own 1-body island; static/kinematic are
            // not island members (they anchor). The slot may be recycled, so
            // overwrite unconditionally.
            if (def.type == BodyType::Dynamic)
            {
                m_islandMgr.CreateSingletonIsland(idx);
                // Phase B: a recycled slot may carry a stale awakeIndex; clear it
                // BEFORE AddToAwakeSet so the idempotency guard does not wrongly
                // skip re-adding a slot that belonged to a prior body.
                m_awakeIndex[idx] = kNotAwake;
                AddToAwakeSet(idx); // new dynamic body starts awake (m_awake[idx]=1 above)
            }
            else
            {
                m_islandMgr.ClearIsland(idx);
                // Phase B: static/kinematic never enter the awake-set; clear any
                // stale index left by a recycled slot's prior dynamic lifetime.
                m_awakeIndex[idx] = kNotAwake;
            }

            // Phase C: maintain the kinematic solver-set. A KINEMATIC body enters
            // the set; static/dynamic do not. Reset the back-index BEFORE the add
            // so a recycled slot carrying a stale m_kinematicIndex cannot make the
            // idempotency guard wrongly skip the re-add (mirrors the awake-set's
            // reset-before-add). For a non-kinematic recycled slot, clear any stale
            // index left by a prior kinematic lifetime so RemoveFromKinematicSet
            // stays a no-op and the invariant holds.
            m_kinematicIndex[idx] = kNotKinematic;
            if (def.type == BodyType::Kinematic)
            {
                AddToKinematicSet(idx); // m_btype[idx] is already set to Kinematic above
            }

            if (def.type == BodyType::Dynamic)
            {
                // A dynamic AABB must be fixedRotation: an axis-aligned box has
                // no meaningful orientation in this fixed-rotation engine
                // (ports the Lua assert, PhysicsWorld.lua:238).
                MOSAIC_ASSERT((def.shape.kind != ShapeKind::Aabb || def.fixedRotation),
                              "dynamic AABB shapes must be fixedRotation "
                              "(axis-aligned by definition)");

                // Mass + rotational inertia from Shape::ComputeMass(density)
                // (the P1.1 MassData -- verified equivalent to the Lua massProps,
                // so we do NOT re-port massProps). md.inertia is about the
                // shape's centroid, which matches the Lua's per-shape inertia.
                const MassData md = def.shape.ComputeMass(def.density);
                Real m       = md.mass;
                Real inertia = md.inertia;

                // Optional mass override: scale inertia by (mass/computedMass)
                // then use the override mass (ports lines 241-243). Guard against
                // a zero computed mass (degenerate shape) so the scale is safe.
                if (def.mass > Real(0))
                {
                    if (m > Real(0))
                    {
                        inertia = inertia * (def.mass / m);
                    }
                    m = def.mass;
                }

                m_invMass[idx]    = m > Real(0) ? Real(1) / m : Real(0);
                m_invInertia[idx] = (def.fixedRotation || inertia <= Real(0))
                                        ? Real(0)
                                        : Real(1) / inertia;
            }

            if (def.type == BodyType::Static)
            {
                m_staticList.push_back(idx);
                // SlotAabb uses the single-shape fallback here: m_bodyFixtures[idx]
                // is empty at this point (a fresh slot, or cleared by RemoveBody on
                // recycle), and the back-compat fixture created later in AddBody has
                // the SAME shape, so the registered AABB stays consistent.
                m_staticTree.Update(idx, SlotAabb(idx));
            }
            else
            {
                // Kinematic + Dynamic: register body-union AABB in residency index
                // (combat-sphere seam). Fixture proxies are added via AddFixtureProxy
                // below; the per-fixture broadphase (m_fixtureBroadphase) is the sole
                // mover broadphase as of Phase 2, Task 3.
                const Aabb2 moverBox = SlotAabb(idx);
                m_residencyGrid.Insert(idx, moverBox);
            }

            // ---- Back-compat fixture creation (v2 Task 4) -------------------
            //
            // AddBody creates exactly ONE fixture from the BodyDef's shape +
            // material so the fixture data model is always populated.
            // The legacy single-shape contact-gen path (GenerateContacts) reads
            // m_shape[] + m_rest[] + m_fric[] DIRECTLY and is UNCHANGED (Task 5
            // wires fixture-pair contact gen and retires that path).  So we
            // create the fixture record WITHOUT calling RecomputeBodyMass --
            // the already-correct m_invMass/m_invInertia derived above (which
            // correctly handles the optional mass override) must be preserved
            // byte-for-byte.
            //
            // EnsureBodyAuxCapacity + store fixedRotation + mass-override flags
            // (needed by RecomputeBodyMass if AddFixture is called later).
            EnsureBodyAuxCapacity(idx + 1u);
            m_bodyFixtures[idx].clear(); // fresh slot (may be recycled)
            m_islandMgr.ClearBodyContacts(idx); // fresh slot (may be recycled)
            m_fixedRotation[idx] = def.fixedRotation ? std::uint8_t(1) : std::uint8_t(0);
            // Record the mass override so RecomputeBodyMass (called by a later
            // AddFixture) honours it and does not silently drop it (Fix 1).
            m_massOverride[idx] = (def.mass > Real(0)) ? def.mass : Real(0);

            // Create the auto fixture record via AllocFixtureSlot (shared with
            // AddFixture -- single field-population site, safe for T5 to extend).
            // We do NOT call RecomputeBodyMass here; the legacy invMass/invInertia
            // path above is the one source of truth for the single-shape case.
            //
            // T6 fix: populate the auto-fixture from the new BodyDef filter +
            // local-transform fields so authored categoryBits / maskBits /
            // localPos / localAngle on the primary fixture flow through.
            // The mass/invMass path above is NOT changed: compound-COM for an
            // offset primary fixture is deferred; legacy dynamics tests stay
            // byte-for-byte green because invMass/invInertia are derived from
            // def.shape + def.density only (above), never from localPos/localAngle.
            {
                FixtureDef autoFd;
                autoFd.shape        = def.shape;
                autoFd.localPos     = def.localPos;
                autoFd.localAngle   = def.localAngle;
                autoFd.density      = def.density;
                autoFd.friction     = def.friction;
                autoFd.restitution  = def.restitution;
                autoFd.categoryBits = def.categoryBits;
                autoFd.maskBits     = def.maskBits;
                autoFd.groupIndex   = def.groupIndex;
                autoFd.isSensor     = def.isSensor;
                autoFd.contactEvents = def.contactEvents;
                autoFd.sensorEvents  = def.sensorEvents;
                autoFd.hitEvents     = def.hitEvents;
                const std::uint32_t autoFi = AllocFixtureSlot(idx, autoFd); // no RecomputeBodyMass

                // Register the auto-fixture in the per-fixture mover broadphase
                // (Phase 2, Task 1). AddFixtureProxy skips Static bodies.
                // Static fixtures are not registered here (covered by m_staticTree).
                AddFixtureProxy(autoFi);

                // Populate the body-mass accessors consistently with the legacy
                // path.  For Dynamic bodies the correct mass was computed above
                // (possibly with a mass override); for Static/Kinematic it is 0.
                // This keeps GetBodyMass/GetBodyInertia/GetLocalCenter correct
                // for the single-fixture case without going through RecomputeBodyMass.
                if (def.type == BodyType::Dynamic)
                {
                    const MassData md = def.shape.ComputeMass(def.density);
                    Real totalMass    = md.mass;
                    Real totalInertia = md.inertia;
                    if (def.mass > Real(0) && totalMass > Real(0))
                    {
                        const Real scale = def.mass / totalMass;
                        totalInertia    *= scale;
                        totalMass        = def.mass;
                    }
                    m_bodyMass[idx]      = totalMass;
                    m_bodyInertia[idx]   = totalInertia;
                    m_localCenterX[idx]  = md.centroid.x;
                    m_localCenterY[idx]  = md.centroid.y;
                }
                else
                {
                    m_bodyMass[idx]     = Real(0);
                    m_bodyInertia[idx]  = Real(0);
                    m_localCenterX[idx] = Real(0);
                    m_localCenterY[idx] = Real(0);
                }
            }

            // COM + the auto-fixture are now in place -> cache the sleep-test extent
            // (Box2D sim->maxExtent). A later AddFixture re-runs this via RecomputeBodyMass.
            RecomputeMaxExtent(idx);

            MOSAIC_LOG_TRACE("body added");
            return BodyHandle{ idx, m_gen[idx] };
        }

        void PhysicsWorld::RemoveBody(BodyHandle h)
        {
            if (!IsValid(h))
            {
                MOSAIC_LOG_WARN("operation on a stale/invalid BodyHandle ignored");
                return;
            }
            MOSAIC_LOG_TRACE("body removed");
            const std::uint32_t idx = h.index;
            m_alive[idx] = 0;
            m_gen[idx] += 1u; // invalidates the handle (stale-handle invariant)

            if (static_cast<BodyType>(m_btype[idx]) == BodyType::Static)
            {
                // staticList is an unsorted no-duplicate index bag -- swap-and-pop
                // is O(1) and order-independent (pair keys are canonically keyed
                // (min,max) so static-loop order does not affect determinism).
                for (std::size_t i = 0; i < m_staticList.size(); ++i)
                {
                    if (m_staticList[i] == idx)
                    {
                        m_staticList[i] = m_staticList.back();
                        m_staticList.pop_back();
                        break;
                    }
                }
                m_staticTree.Remove(idx);
            }
            else
            {
                m_residencyGrid.Remove(idx);
            }

            m_solver->DropBody(idx); // drop warm-start state for the recycled slot

            // Immediately destroy every persistent-pool contact referencing this
            // body (Task 5). RemoveBody drops ALL the body's fixtures, so the
            // by-body walk catches every one of them in a single pass -- a removed
            // body never leaves a stale contact even for the one Step before
            // UpdateContacts' stale-handle guard would reap it.
            m_graph.DestroyContactsForBody(*this, idx);

            // Phase C, Task 4 leak-detector: every contact referencing this body
            // released its color bits above (DestroyContactsForBody -> ReleaseContactColor
            // per contact), so the removed slot's color mask MUST be 0. If this
            // fires, a destroy path is missing its ReleaseContactColor call (a leaked
            // bit would mis-color a future body recycled into this slot). Debug-only;
            // the recycle path also defaults the mask to 0 in EnsureCapacity, so a
            // leak here is a real bug, not a benign stale value.
            MOSAIC_ASSERT(m_graph.DebugBodyMaskClear(idx),
                          "RemoveBody: body left a non-zero color mask -- a Destroy site is missing ReleaseContactColor");

            // Phase B: remove from the awake-set while the slot is still typed
            // Dynamic (btype has not been touched yet; RemoveFromAwakeSet checks
            // the btype to guard non-dynamics -- though a sleeping body is already
            // absent, being idempotent makes it unconditional and safe).
            RemoveFromAwakeSet(idx);

            // Phase C: remove from the kinematic-set while the slot is still typed
            // Kinematic (btype is untouched here; RemoveFromKinematicSet is
            // idempotent, so a non-kinematic slot -- already absent -- is a no-op).
            RemoveFromKinematicSet(idx);

            // Release the removed body's island membership (Phase A). Erase the
            // slot from its island's member list; if the island is now empty, free
            // it; otherwise mark it a split candidate (losing a member can fracture
            // the remaining pile). A recycled slot is reassigned a fresh island in
            // AddBody. Order-stable enough: SplitIsland re-derives membership from
            // contacts, not member order, so the swap-erase is determinism-safe.
            m_islandMgr.RemoveBodyFromIsland(idx);

            // Drop joints referencing the destroyed body (ports PhysicsWorld.lua
            // 281-286: `j.a.idx == idx or j.b.idx == idx`). Match by HANDLE index
            // (stable from construction, independent of Prepare). Iterate in
            // reverse so the swap-free erase keeps the surviving joints' order.
            for (std::size_t i = m_joints.size(); i-- > 0;)
            {
                const Joint* j = m_joints[i].get();
                const BodyHandle ha = j->HandleA();
                const BodyHandle hb = j->HandleB();
                const bool aIsIdx = (ha.generation != 0u && ha.index == idx);
                const bool bIsIdx = (hb.generation != 0u && hb.index == idx);
                if (aIsIdx || bIsIdx)
                {
                    // If this joint bridged two Dynamic bodies, dropping idx removes
                    // the joint's island edge: wake + mark the SURVIVING endpoint's
                    // island a split candidate so it re-derives its components
                    // (mirrors the contact-removal pattern). idx already left its own
                    // island above (IslandOf(idx) == kInvalidIsland here).
                    const BodyHandle other = aIsIdx ? hb : ha;
                    if (other.generation != 0u && other.index < m_count &&
                        TypeSlot(idx) == BodyType::Dynamic &&
                        TypeSlot(other.index) == BodyType::Dynamic)
                    {
                        WakeIsland(other.index);
                        MarkSplitCandidate(IslandOf(other.index));
                    }
                    m_joints.erase(m_joints.begin() + static_cast<std::ptrdiff_t>(i));
                }
            }

            m_shape[idx] = Shape{}; // release polygon storage

            // ---- Drop all fixtures belonging to this body (v2 Task 4) -------
            if (idx < m_bodyFixtures.size())
            {
                // Recycle each fixture slot (bump generation; clear shape).
                // We iterate a COPY because DropFixture would modify the list.
                const std::vector<std::uint32_t> fxCopy = m_bodyFixtures[idx];
                for (const std::uint32_t fi : fxCopy)
                {
                    if (fi < m_fxCount && m_fxGen[fi] != 0u)
                    {
                        // Remove the fixture proxy BEFORE bumping gen (Phase 2,
                        // Task 1). Remove only needs the slot id.
                        RemoveFixtureProxy(fi);
                        m_fxGen[fi] += 1u;           // invalidate
                        m_fxShape[fi] = Shape{};     // release storage
                        m_fxFree.push_back(fi);
                    }
                }
                m_bodyFixtures[idx].clear();
                m_islandMgr.ClearBodyContacts(idx); // DestroyContactsForBody already drained it; defensive
                m_bodyMass[idx]     = Real(0);
                m_bodyInertia[idx]  = Real(0);
                m_localCenterX[idx] = Real(0);
                m_localCenterY[idx] = Real(0);
            }

            m_free.push_back(idx);
        }

        Joint* PhysicsWorld::AddJoint(const JointDef& def)
        {
            // Build the concrete joint (the Lua Joints.make factory). Unknown
            // kinds return nullptr (never thrown).
            std::unique_ptr<Joint> j = MakeJoint(*this, def);
            if (!j)
            {
                return nullptr;
            }
            Joint* raw = j.get();
            m_joints.push_back(std::move(j));
            MOSAIC_LOG_TRACE("joint added");

            // Wake the jointed bodies so a sleeping captive rejoins the solve
            // (ports addJoint's def.a:wake() / def.b:wake()).
            Wake(def.a);
            Wake(def.b);

            // Union the two bodies' islands so a jointed dynamic construct sleeps
            // as ONE unit (Box2D treats a joint as an island edge). Only DYNAMIC
            // bodies are island members, so a joint to a static/kinematic anchor
            // unions nothing. Resolve the def's body HANDLES to slots directly:
            // BodyA()/BodyB() are not resolved until the joint's first Prepare, but
            // the handle indices are stable from construction. Both bodies were just
            // woken above, so the merged island is uniformly awake.
            const BodyHandle ha = def.a;
            const BodyHandle hb = def.b;
            if (ha.generation != 0u && hb.generation != 0u &&
                ha.index < m_count && hb.index < m_count &&
                TypeSlot(ha.index) == BodyType::Dynamic &&
                TypeSlot(hb.index) == BodyType::Dynamic)
            {
                const std::uint32_t ia = IslandOf(ha.index);
                const std::uint32_t ib = IslandOf(hb.index);
                if (ia != ib &&
                    ia != Island::kInvalidIsland &&
                    ib != Island::kInvalidIsland)
                {
                    MergeIslands(ia, ib);
                }
            }
            return raw;
        }

        Vec2 PhysicsWorld::JointReactionForce(const Joint* j) const noexcept
        {
            // Pointer membership, as RemoveJoint finds a joint: a removed joint's
            // stale pointer matches nothing and reads zero.
            for (const std::unique_ptr<Joint>& owned : m_joints)
            {
                if (owned.get() == j && j != nullptr) { return j->ReactionForce(); }
            }
            return Vec2(Real(0), Real(0));
        }

        Real PhysicsWorld::JointReactionTorque(const Joint* j) const noexcept
        {
            for (const std::unique_ptr<Joint>& owned : m_joints)
            {
                if (owned.get() == j && j != nullptr) { return j->ReactionTorque(); }
            }
            return Real(0);
        }

        void PhysicsWorld::RemoveJoint(Joint* j)
        {
            if (j == nullptr)
            {
                return;
            }
            for (std::size_t i = 0; i < m_joints.size(); ++i)
            {
                if (m_joints[i].get() == j)
                {
                    // If this joint bridged two Dynamic bodies, its island edge is
                    // gone: wake the island + mark it a split candidate so the
                    // deferred pass re-derives its connected components (mirrors the
                    // contact-removal pattern in DestroyContactsForBody). Both
                    // endpoints share the island, so marking/waking one covers both.
                    const BodyHandle ha = j->HandleA();
                    const BodyHandle hb = j->HandleB();
                    if (ha.generation != 0u && hb.generation != 0u &&
                        ha.index < m_count && hb.index < m_count &&
                        TypeSlot(ha.index) == BodyType::Dynamic &&
                        TypeSlot(hb.index) == BodyType::Dynamic)
                    {
                        WakeIsland(ha.index);
                        MarkSplitCandidate(IslandOf(ha.index));
                    }
                    m_joints.erase(m_joints.begin() + static_cast<std::ptrdiff_t>(i));
                    MOSAIC_LOG_TRACE("joint removed");
                    return;
                }
            }
        }

        bool PhysicsWorld::IsValid(BodyHandle h) const noexcept
        {
            // Ports handleValid: in-range index, matching generation, alive.
            // gen==0 is NEVER a live slot (AddBody bumps 0->1 on first use), so
            // the sentinel kInvalidBody{0,0} can never match a live body.
            return h.index < m_count && m_gen[h.index] == h.generation &&
                   m_alive[h.index] != 0;
        }

        Vec2 PhysicsWorld::Position(BodyHandle h) const noexcept
        {
            if (!IsValid(h))
            {
                return Vec2(Real(0), Real(0));
            }
            return Vec2(m_posX[h.index], m_posY[h.index]);
        }

        void PhysicsWorld::SetPosition(BodyHandle h, Vec2 p)
        {
            if (!IsValid(h))
            {
                MOSAIC_LOG_WARN("operation on a stale/invalid BodyHandle ignored");
                return;
            }
            if (!std::isfinite(p.x) || !std::isfinite(p.y))
            {
                MOSAIC_LOG_WARN("rejected non-finite SetPosition");
                return;
            }
            const std::uint32_t i = h.index;
            // Teleport: prev snaps too (no lerp smear) -- ports setPosition.
            m_posX[i]  = p.x;
            m_posY[i]  = p.y;
            m_prevX[i] = p.x;
            m_prevY[i] = p.y;
            if (static_cast<BodyType>(m_btype[i]) != BodyType::Static)
            {
                UpdateMoverProxies(i);
            }
        }

        void PhysicsWorld::MovePosition(BodyHandle h, Vec2 p)
        {
            if (!IsValid(h))
            {
                MOSAIC_LOG_WARN("operation on a stale/invalid BodyHandle ignored");
                return;
            }
            const std::uint32_t i = h.index;
            // Step-managed-prev move: write pos + the mover-broadphase AABB but
            // leave prev untouched (Step owns prev). Ports the Lua slideMove
            // write-back: w.posX[i] = x; w.posY[i] = y; moverHash:update(...).
            m_posX[i] = p.x;
            m_posY[i] = p.y;
            if (static_cast<BodyType>(m_btype[i]) != BodyType::Static)
            {
                UpdateMoverProxies(i);
            }
        }

        Vec2 PhysicsWorld::Velocity(BodyHandle h) const noexcept
        {
            if (!IsValid(h))
            {
                return Vec2(Real(0), Real(0));
            }
            return Vec2(m_velX[h.index], m_velY[h.index]);
        }

        Real PhysicsWorld::AngularVelocity(BodyHandle h) const noexcept
        {
            if (!IsValid(h))
            {
                return Real(0);
            }
            return m_angVel[h.index];
        }

        void PhysicsWorld::SetVelocity(BodyHandle h, Vec2 v)
        {
            if (!IsValid(h))
            {
                MOSAIC_LOG_WARN("operation on a stale/invalid BodyHandle ignored");
                return;
            }
            if (!std::isfinite(v.x) || !std::isfinite(v.y))
            {
                MOSAIC_LOG_WARN("rejected non-finite SetVelocity");
                return;
            }
            const std::uint32_t i = h.index;
            const BodyType bt = static_cast<BodyType>(m_btype[i]);
            // Kinematic + Dynamic accept velocity; Static ignores (ports
            // setVelocity). Setting a Dynamic body's velocity WAKES it (line
            // 102) so a sleeping body re-enters integration next Step.
            if (bt == BodyType::Kinematic || bt == BodyType::Dynamic)
            {
                m_velX[i] = v.x;
                m_velY[i] = v.y;
                if (bt == BodyType::Dynamic)
                {
                    m_awake[i]      = 1;
                    m_sleepTimer[i] = Real(0);
                    AddToAwakeSet(i); // Phase B: kInvalidIsland safety net (WakeIsland also adds, but may not run for isolated body)
                    WakeIsland(i); // wake the whole island, not just this body
                }
            }
        }

        void PhysicsWorld::SetAngularVelocity(BodyHandle h, Real w)
        {
            if (!IsValid(h))
            {
                MOSAIC_LOG_WARN("operation on a stale/invalid BodyHandle ignored");
                return;
            }
            const std::uint32_t i = h.index;
            const BodyType bt = static_cast<BodyType>(m_btype[i]);
            // Mirror SetVelocity: Kinematic + Dynamic accept it; Static ignores.
            // Setting a Dynamic body's angular velocity WAKES it (and its island).
            if (bt == BodyType::Kinematic || bt == BodyType::Dynamic)
            {
                m_angVel[i] = w;
                if (bt == BodyType::Dynamic)
                {
                    m_awake[i]      = 1;
                    m_sleepTimer[i] = Real(0);
                    AddToAwakeSet(i);
                    WakeIsland(i);
                }
            }
        }

        void PhysicsWorld::ApplyImpulse(BodyHandle h, Vec2 impulse)
        {
            if (!IsValid(h))
            {
                MOSAIC_LOG_WARN("operation on a stale/invalid BodyHandle ignored");
                return;
            }
            const std::uint32_t i = h.index;
            // Dynamic only; wakes (ports applyImpulse, the no-point branch).
            if (static_cast<BodyType>(m_btype[i]) != BodyType::Dynamic)
            {
                return;
            }
            m_awake[i]      = 1;
            m_sleepTimer[i] = Real(0);
            AddToAwakeSet(i); // Phase B: kInvalidIsland safety net
            WakeIsland(i); // wake the whole island, not just this body
            m_velX[i] += impulse.x * m_invMass[i];
            m_velY[i] += impulse.y * m_invMass[i];
        }

        void PhysicsWorld::ApplyImpulse(BodyHandle h, Vec2 impulse, Vec2 worldPoint)
        {
            if (!IsValid(h))
            {
                MOSAIC_LOG_WARN("operation on a stale/invalid BodyHandle ignored");
                return;
            }
            const std::uint32_t i = h.index;
            // Dynamic only; wakes (ports applyImpulse, the px,py branch). Linear
            // part as above, plus an angular term from the lever arm about the
            // body's CENTER OF MASS: angVel += cross(p - com, i) * invInertia.
            // The lever MUST be measured from the COM, not the origin: invInertia
            // is the inertia about the COM and the solver integrates rotation
            // about the COM (FinalizePositions). For a single-fixture / COM==origin
            // body WorldCom() == origin, so this is byte-identical there; for an
            // off-COM compound body the origin lever was wrong (it dropped the
            // origin->COM offset, under-rotating or mis-signing the spin).
            if (static_cast<BodyType>(m_btype[i]) != BodyType::Dynamic)
            {
                return;
            }
            m_awake[i]      = 1;
            m_sleepTimer[i] = Real(0);
            AddToAwakeSet(i); // Phase B: kInvalidIsland safety net
            WakeIsland(i); // wake the whole island, not just this body
            m_velX[i] += impulse.x * m_invMass[i];
            m_velY[i] += impulse.y * m_invMass[i];
            const Vec2 com = WorldCom(Vec2(m_posX[i], m_posY[i]), m_angle[i],
                                      Vec2(m_localCenterX[i], m_localCenterY[i]));
            const Real rx = worldPoint.x - com.x;
            const Real ry = worldPoint.y - com.y;
            m_angVel[i] += (rx * impulse.y - ry * impulse.x) * m_invInertia[i];
        }

        void PhysicsWorld::Wake(BodyHandle h)
        {
            if (!IsValid(h))
            {
                MOSAIC_LOG_WARN("operation on a stale/invalid BodyHandle ignored");
                return;
            }
            const std::uint32_t i = h.index;
            // Only Dynamic bodies have a sleep state to clear (ports Body:wake).
            if (static_cast<BodyType>(m_btype[i]) == BodyType::Dynamic)
            {
                m_awake[i]      = 1;
                m_sleepTimer[i] = Real(0);
                AddToAwakeSet(i); // Phase B: kInvalidIsland safety net
                WakeIsland(i); // wake the whole island, not just this body
            }
        }

        bool PhysicsWorld::IsAwake(BodyHandle h) const noexcept
        {
            if (!IsValid(h))
            {
                return false;
            }
            // Static/Kinematic never sleep -> always report awake (their m_awake
            // stays 1). Ports Body:isAwake.
            return m_awake[h.index] != 0;
        }

        Real PhysicsWorld::GetAngle(BodyHandle h) const noexcept
        {
            if (!IsValid(h))
            {
                return Real(0);
            }
            return m_angle[h.index];
        }

        // ---- gravity well + per-body gravity scale ---------------------------

        void PhysicsWorld::SetGravityWell(const GravityWell& well)
        {
            if (!well.enabled)
            {
                m_gravityWell.enabled = false;
                return;
            }
            const bool finite = std::isfinite(well.center.x) && std::isfinite(well.center.y)
                             && std::isfinite(well.surfaceRadius) && std::isfinite(well.surfaceGravity)
                             && std::isfinite(well.fadeStart) && std::isfinite(well.fadeEnd);
            const bool fadeOk = well.falloff != GravityFalloff::Fade
                             || (well.fadeStart >= Real(0) && well.fadeEnd > well.fadeStart);
            if (!finite || !(well.surfaceRadius > Real(0)) || !fadeOk)
            {
                MOSAIC_LOG_WARN("SetGravityWell: invalid well ignored (non-finite field, surfaceRadius <= 0, or fadeEnd <= fadeStart)");
                return;
            }
            m_gravityWell = well;
        }

        Vec2 PhysicsWorld::GravityWellAccel(Vec2 p) const noexcept
        {
            const GravityWell& gw = m_gravityWell;
            if (!gw.enabled)
            {
                return Vec2(Real(0), Real(0));
            }
            const Real dx = gw.center.x - p.x;
            const Real dy = gw.center.y - p.y;
            const Real r2 = dx * dx + dy * dy;
            if (!(r2 > Real(0)))
            {
                return Vec2(Real(0), Real(0)); // at the centre: no direction, no pull
            }
            const Real r = std::sqrt(r2);
            const Real R = gw.surfaceRadius;
            Real g;
            if (r < R)
            {
                g = gw.surfaceGravity * r / R;
            }
            else if (gw.falloff == GravityFalloff::InverseSquare)
            {
                g = gw.surfaceGravity * (R * R) / r2;
            }
            else if (r <= gw.fadeStart)
            {
                g = gw.surfaceGravity;
            }
            else if (r >= gw.fadeEnd)
            {
                g = Real(0);
            }
            else
            {
                const Real t = (r - gw.fadeStart) / (gw.fadeEnd - gw.fadeStart);
                g = gw.surfaceGravity * (Real(1) - t * t * (Real(3) - Real(2) * t));
            }
            const Real k = g / r; // unit direction (dx, dy) / r, times g
            return Vec2(dx * k, dy * k);
        }

        void PhysicsWorld::SetGravityScale(BodyHandle h, Real scale)
        {
            if (!IsValid(h))
            {
                MOSAIC_LOG_WARN("operation on a stale/invalid BodyHandle ignored");
                return;
            }
            if (!std::isfinite(scale))
            {
                MOSAIC_LOG_WARN("SetGravityScale: non-finite scale ignored");
                return;
            }
            const std::uint32_t i = h.index;
            m_gravityScale[i] = scale;
            if (static_cast<BodyType>(m_btype[i]) == BodyType::Dynamic)
            {
                Wake(h); // a sleeping body would otherwise never feel the change
            }
        }

        void PhysicsWorld::SetBullet(BodyHandle h, bool bullet)
        {
            if (!IsValid(h))
            {
                MOSAIC_LOG_WARN("operation on a stale/invalid BodyHandle ignored");
                return;
            }
            m_bullet[h.index] = bullet ? std::uint8_t(1) : std::uint8_t(0);
        }

        bool PhysicsWorld::IsBullet(BodyHandle h) const noexcept
        {
            return IsValid(h) && m_bullet[h.index] != 0;
        }

        void PhysicsWorld::SetBodyFilter(BodyHandle h, std::uint32_t categoryBits, std::uint32_t maskBits)
        {
            if (!IsValid(h))
            {
                MOSAIC_LOG_WARN("operation on a stale/invalid BodyHandle ignored");
                return;
            }
            const std::uint32_t b = h.index;
            if (b >= m_bodyFixtures.size())
            {
                return;
            }
            const bool isStatic = static_cast<BodyType>(m_btype[b]) == BodyType::Static;
            bool changed = false;
            for (const std::uint32_t fi : m_bodyFixtures[b])
            {
                if (fi >= m_fxCount || m_fxGen[fi] == 0u)
                {
                    continue;
                }
                if (m_fxFilterCat[fi] == categoryBits && m_fxFilterMask[fi] == maskBits)
                {
                    continue;
                }
                m_fxFilterCat[fi]  = categoryBits;
                m_fxFilterMask[fi] = maskBits;
                changed = true;
                // b2ResetProxy: drop the fixture's contacts (waking what they
                // touched), then re-insert its mover proxy so the pair set is
                // rebuilt against the new filter.
                m_graph.DestroyContactsForFixture(*this, fi);
                if (!isStatic)
                {
                    RemoveFixtureProxy(fi);
                    AddFixtureProxy(fi);
                }
            }
            if (changed && static_cast<BodyType>(m_btype[b]) == BodyType::Dynamic)
            {
                Wake(h);
            }
        }

        void PhysicsWorld::SetFixtureEvents(FixtureHandle fh, bool contact, bool sensor, bool hit)
        {
            if (!IsValid(fh))
            {
                MOSAIC_LOG_WARN("operation on a stale/invalid FixtureHandle ignored");
                return;
            }
            // contact/hit apply to contacts CREATED afterwards (captured at create,
            // Box2D contact.c:253); sensor is read every sensor pass (spec s6.1, A3).
            m_fxContactEvents[fh.index] = contact ? std::uint8_t(1) : std::uint8_t(0);
            m_fxSensorEvents[fh.index]  = sensor  ? std::uint8_t(1) : std::uint8_t(0);
            m_fxHitEvents[fh.index]     = hit     ? std::uint8_t(1) : std::uint8_t(0);
        }

        void PhysicsWorld::SetHitEventThreshold(Real threshold)
        {
            if (!std::isfinite(threshold) || threshold < Real(0))
            {
                MOSAIC_LOG_WARN("SetHitEventThreshold: negative or non-finite threshold ignored");
                return;
            }
            m_hitEventThreshold = threshold;
        }

        std::uint8_t PhysicsWorld::DebugContactEventFlags(FixtureHandle a, FixtureHandle b) const
        {
            const Contact* c = m_graph.FindContact(a, b);
            return c != nullptr ? c->eventFlags : std::uint8_t(0xFF);
        }

        Real PhysicsWorld::GravityScale(BodyHandle h) const noexcept
        {
            return IsValid(h) ? m_gravityScale[h.index] : Real(1);
        }

        void PhysicsWorld::SetAngle(BodyHandle h, Real angle)
        {
            if (!IsValid(h))
            {
                MOSAIC_LOG_WARN("operation on a stale/invalid BodyHandle ignored");
                return;
            }
            const std::uint32_t i = h.index;
            m_prevAngle[i] = angle; // a set angle is a teleport: no swept rotation through it
            m_angle[i] = angle;
            // A non-circle body's world AABB is rotation-aware. Statics never
            // re-register via Step, so refresh the static tree; movers refresh
            // their per-fixture proxies + residency immediately (mirrors SetPosition).
            if (static_cast<BodyType>(m_btype[i]) == BodyType::Static)
                m_staticTree.Update(i, SlotAabb(i));
            else
                UpdateMoverProxies(i);
        }

        Vec2 PhysicsWorld::DrawPosition(BodyHandle h, Real alpha) const noexcept
        {
            if (!IsValid(h))
            {
                return Vec2(Real(0), Real(0));
            }
            const std::uint32_t i = h.index;
            // Render-boundary lerp prev..curr (ports drawPosition).
            return Vec2(m_prevX[i] + (m_posX[i] - m_prevX[i]) * alpha,
                        m_prevY[i] + (m_posY[i] - m_prevY[i]) * alpha);
        }

        const Shape* PhysicsWorld::GetShape(BodyHandle h) const noexcept
        {
            if (!IsValid(h))
            {
                return nullptr;
            }
            return &m_shape[h.index];
        }

        BodyType PhysicsWorld::GetType(BodyHandle h) const noexcept
        {
            if (!IsValid(h))
            {
                return BodyType::Static;
            }
            return static_cast<BodyType>(m_btype[h.index]);
        }

        bool PhysicsWorld::IsSensor(BodyHandle h) const noexcept
        {
            return IsValid(h) && m_sensor[h.index] != 0;
        }

        void PhysicsWorld::SetEventsEnabled(bool on)
        {
            // Plain world gate (spec s6.2). Begin and hit pushes read it.
            // An End is delivered iff its Begin was (R10, R12), even after the
            // gate closes. Turning it back on does not emit a burst.
            m_eventGate = on;
        }

        void PhysicsWorld::Step(Real dt)
        {
            StepImpl(dt, nullptr);
        }

        void PhysicsWorld::StepTraced(Real dt, StepTrace& trace)
        {
            StepImpl(dt, &trace);
        }

        void PhysicsWorld::StepImpl(Real dt, StepTrace* trace)
        {
            // STEP ORDER (P2.2 -- Box2D v3 TGS Soft restructure of the P1.8/P2.1
            // pipeline). The P2.1 INLINE dynamic gravity-integrate (old stage 1)
            // and dynamic position-integrate (old stage 4) are SUPERSEDED: the
            // Soft Step solver now OWNS dynamic velocity + position integration,
            // folded INTO its sub-step loop (the proven-stable v3 form). Dynamics
            // are therefore integrated EXACTLY ONCE per Step -- by the solver --
            // with no double-integration.
            //
            //   stage 1: prev snapshot (ALL) + KINEMATIC integrate (once/step) +
            //            kinematic mover-broadphase update.
            //   stage 2: solver contact generation (Part A) -- manifolds for
            //            awake-dynamic pairs (vs tile spans + static bodies via
            //            StaticCandidates; vs mover-mover pairs involving a
            //            dynamic, narrowed). Builds m_contactConstraints.
            //   stage 3: SOFT STEP SOLVE (Part B) -- the solver runs Prepare ->
            //            [per sub-step: integrate vel, warm-start, solve(bias),
            //            integrate pos, relax] -> restitution -> store impulses
            //            -> commit dynamic positions + mover-broadphase update.
            //   stage 4: bullet GJK-TOI clamp (P3.1 CCD) -- sweep each isBullet
            //            body prev->curr vs statics; clamp to time-of-impact so a
            //            thin static wall cannot be tunneled. (The speculative
            //            margin in stage 2 is the inline CCD for fast dynamics;
            //            this is the discrete backup, primarily for kinematics.)
            //            CCD runs after the solver commits positions, before
            //            island/events.
            //   stage 5: island sleep bookkeeping            (P2.4)
            //   stage 6: contact / hit / sensor event arrays (spec s6)
            //
            // Free-fall parity: with NO contacts the solver's sub-step loop is a
            // pure semi-implicit integrate (gravity per sub-step, position per
            // sub-step). Summed over substepCount sub-steps of length h = dt/N
            // this equals the P2.1 single-step semi-implicit Euler to f32
            // tolerance (the PhysicsDynamics free-fall test's margins absorb the
            // sub-step regrouping). Index-ordered, no wall-clock, no fast-math.

            // Begin/hit arrays are this step's only (Box2D world.c:710-712).
            // End buffers stay: this step writes m_contactEndEvents[m_endEventIndex]
            // and m_sensorEndEvents[m_endEventIndex], and stage 6 flips that
            // buffer into view.
            m_contactBeginEvents.clear();
            m_contactHitEvents.clear();
            m_sensorBeginEvents.clear();

            // ---- stage 1: prev snapshot + kinematic integrate ----------------
            //
            // WHY two passes instead of one (Phase B, Task 4):
            //
            // Pass A (0..m_count, non-dynamics): statics + kinematics.
            //   * Statics: snapping prev is harmless (AddBody already sets
            //     prev==pos and statics never move), but it keeps the old
            //     invariant for any teleport/manual-edit that touched pos.
            //   * Kinematics: snap prev then integrate pos; update broadphase.
            //   Kinematics are NOT in the awake-set (YAGNI a kinematic list), so
            //   they stay on the 0..m_count path.
            //
            // Pass B (ForEachAwake): AWAKE dynamic slots only.
            //   Sleeping dynamics are snapped by NEITHER pass -- their prev was
            //   set to pos at the moment they fell asleep (SnapPrevToPos in
            //   IslandManager::UpdateSleep), and their pos is frozen thereafter, so
            //   prev==pos persists without any per-step work.
            //
            // WIN: sleeping dynamics (potentially the MAJORITY in a settled
            // scene) are entirely skipped by Stage 1 after warmup.
            {
                ARCANE_STEPPROF_SCOPE(Stage1Snapshot);

                // Pass A: statics + kinematics (0..m_count; no dynamic slots).
                for (std::uint32_t i = 0; i < m_count; ++i)
                {
                    if (m_alive[i] == 0)
                    {
                        continue;
                    }
                    const BodyType t = static_cast<BodyType>(m_btype[i]);
                    if (t == BodyType::Dynamic)
                    {
                        continue; // handled in Pass B below
                    }
                    // Static prev==pos always (AddBody seeds it; statics never move).
                    // Snap harmlessly keeps the invariant for manual teleports.
                    m_prevX[i] = m_posX[i];
                    m_prevY[i] = m_posY[i];
                    m_prevAngle[i] = m_angle[i];

                    // Kinematic bodies integrate ONCE per Step (unchanged from P2.1).
                    // Dynamic gravity/damping + position now live in the solver's
                    // sub-step loop (stage 3); a dynamic body with no contacts still
                    // gets the same semi-implicit fall, just regrouped over sub-steps.
                    if (t == BodyType::Kinematic)
                    {
                        // Only a MOVING kinematic integrates + refreshes its broadphase
                        // proxy.  A stationary kinematic's proxy is already current (its
                        // last mutation -- SetPosition/SetAngle/SetVelocity -- refreshed
                        // it at the call site), so the per-step UpdateMoverProxies +
                        // residency-grid Move would be pure waste.  Stage-1 integration
                        // is LINEAR only (angle is not integrated here; SetAngle refreshes
                        // at its call site), so "moved this step" == nonzero linear vel.
                        if (m_velX[i] != Real(0) || m_velY[i] != Real(0))
                        {
                            m_posX[i] += m_velX[i] * dt;
                            m_posY[i] += m_velY[i] * dt;
                            // UpdateMoverProxies refreshes all per-fixture mover-broadphase
                            // proxies and the body's residency grid (Phase 2, Task 3).
                            UpdateMoverProxies(i);
                        }
                    }
                }

                // Pass B: snap prev for every AWAKE dynamic slot.
                // Sleeping dynamics are covered by SnapPrevToPos at sleep-time.
                ForEachAwake([&](std::uint32_t i)
                {
                    m_prevX[i] = m_posX[i];
                    m_prevY[i] = m_posY[i];
                    m_prevAngle[i] = m_angle[i];
                });
            }

            // ---- stage 2: solver contact generation (Part A) -----------------
            // Collision-rebuild Phase 3, Task 4 -- THE SWAP. The persistent-contact
            // path is now the SOLE solver feed; GenerateContacts' per-step rebuild
            // is RETIRED. UpdateContacts does ALL the narrowphase this Step:
            //   * fixture<->fixture (mover<->mover + mover<->static-body) into the
            //     graph-owned PERSISTENT pool (created/updated/destroyed across steps),
            //   * tile spans into the graph's TRANSIENT span scratch (virtual
            //     fixtures, refilled each Step).
            // EmitContactConstraints then walks BOTH into m_contactConstraints in
            // the canonical (bodyA, bodyB, fixtureA, fixtureB) order; the ids are
            // stable (MixContactId) so warm-start impulses (now carried on each
            // persistent Contact's manifold point, seeded into the emitted
            // constraint here and written back after Solve -- see stage 3b) line up
            // with the same physical contact across steps.
            { ARCANE_STEPPROF_SCOPE(Narrowphase); m_graph.UpdateContacts(*this, dt); }
            m_contactConstraints.clear();
            { ARCANE_STEPPROF_SCOPE(EmitConstraints); m_graph.EmitContactConstraints(*this, m_contactConstraints); }

            // ---- stage 3: Soft Step solve (Part B) ---------------------------
            // The solver integrates dynamic velocity + position across sub-steps,
            // solves soft contacts, applies restitution, and commits positions +
            // the mover broadphase. Run it whenever there are dynamics to move
            // (constraints OR free-falling bodies); the solver is a no-op for a
            // scene with no awake dynamics.
            // Rebuild the pooled JointConstraint array (Joint* views into the
            // world-owned m_joints) so the SolverContext keeps its P2.1 shape.
            // clear() preserves capacity -> zero steady-state alloc.
            m_jointConstraints.clear();
            for (std::size_t i = 0; i < m_joints.size(); ++i)
            {
                Joint* jt = m_joints[i].get();
                if (jt == nullptr) { continue; }
                // Skip a joint whose Dynamic endpoints are ALL asleep: the solver's
                // global joint passes must not integrate a sleeping construct (the
                // perf win + the sleeping-frozen contract, mirroring the contact
                // awake-gate). Since a jointed dynamic pair shares one island (Step 1)
                // and islands sleep/wake atomically, this never mixes awake+asleep
                // endpoints. A joint anchored to a static/kinematic body is driven by
                // its one Dynamic endpoint. A woken island re-enters its joints next
                // Step (this array is rebuilt every Step).
                const BodyHandle ha = jt->HandleA();
                const BodyHandle hb = jt->HandleB();
                bool anyAwakeDynamic = false;
                if (ha.generation != 0u && ha.index < m_count &&
                    TypeSlot(ha.index) == BodyType::Dynamic && AwakeSlot(ha.index))
                {
                    anyAwakeDynamic = true;
                }
                if (hb.generation != 0u && hb.index < m_count &&
                    TypeSlot(hb.index) == BodyType::Dynamic && AwakeSlot(hb.index))
                {
                    anyAwakeDynamic = true;
                }
                if (!anyAwakeDynamic) { continue; }
                // Open this step's reaction window (Joint::ReactionForce): the
                // joint sums what it delivers to B through the solve below.
                jt->BeginReactionWindow(dt);
                JointConstraint jc;
                jc.joint = jt;
                m_jointConstraints.push_back(jc);
            }

            // ---- stage 3 (traced only): start poses + the per-stage snapshot --
            //
            // The solver keeps its mid-step pose deltas in its own BodyState rows
            // (dpx/dpy/dq, zeroed by SyncInCompacted) and commits them to the
            // world ONCE, after the stage walk, in SoftStep::FinalizePositionsSoA.
            // A snapshot therefore composes each row's START pose with the deltas
            // accumulated so far. Indexed by DENSE solverIndex to match
            // BodyState: awake dynamics at AwakeIndexOf(slot) in [0, AwakeCount()),
            // kinematics at AwakeCount() + KinematicIndexOf(slot). Recording the
            // poses here (rather than reading PosSlot inside the hook) makes the
            // snapshot independent of anything the solver region might commit in
            // future. Both stay empty -- no allocation -- on the Step(dt) path.
            struct TraceStartPose
            {
                std::uint32_t slot = kInvalidSlot;
                Vec2          pos{ Real(0), Real(0) };
                Real          angle = Real(0);
                Vec2          localCenter{ Real(0), Real(0) };
            };
            std::vector<TraceStartPose> traceStart;
            if (trace != nullptr)
            {
                const std::uint32_t awakeRows = AwakeCount();
                traceStart.assign(
                    static_cast<std::size_t>(awakeRows + KinematicCount()),
                    TraceStartPose{});
                ForEachAwake([&](std::uint32_t s)
                {
                    TraceStartPose& sp = traceStart[AwakeIndexOf(s)];
                    sp.slot        = s;
                    sp.pos         = PosSlot(s);
                    sp.angle       = AngleSlot(s);
                    sp.localCenter = LocalCenterSlot(s);
                });
                ForEachKinematic([&](std::uint32_t s)
                {
                    TraceStartPose& sp = traceStart[awakeRows + KinematicIndexOf(s)];
                    sp.slot        = s;
                    sp.pos         = PosSlot(s);
                    sp.angle       = AngleSlot(s);
                    sp.localCenter = LocalCenterSlot(s);
                });
            }

            auto traceHookFn = [&](StageType stage, int substep,
                                   const SolverStageContext& sc)
            {
                StepTraceSnapshot snap;
                snap.stage   = stage;
                snap.substep = static_cast<std::uint8_t>(substep);

                // ---- bodies, in dense solver-row order --------------------
                snap.bodies.reserve(traceStart.size());
                for (std::size_t row = 0; row < traceStart.size(); ++row)
                {
                    const TraceStartPose& sp = traceStart[row];
                    const BodyState&      bs = sc.bodyState[row];
                    // Compound-COM composition, verbatim SoftStep::
                    // FinalizePositionsSoA: c = (p0 + R(a0)*lc) + dp;
                    // a = a0 + dq; p = c - R(a)*lc. For localCenter == (0,0)
                    // (every single-fixture body) this reduces exactly to
                    // p0 + dp / a0 + dq, and matching the commit expression
                    // is what makes the LAST snapshot bit-identical to the
                    // pose the Step leaves behind.
                    const Vec2 dp(static_cast<Real>(bs.dpx),
                                  static_cast<Real>(bs.dpy));
                    const Real dr = static_cast<Real>(bs.dq);
                    const Vec2 c0 = sp.pos + RotateVec(sp.angle, sp.localCenter);
                    const Vec2 c  = c0 + dp;
                    const Real a  = sp.angle + dr;

                    StepTraceBody tb;
                    tb.body            = sp.slot;
                    tb.position        = c - RotateVec(a, sp.localCenter);
                    tb.angle           = a;
                    tb.velocity        = Vec2(static_cast<Real>(bs.vx),
                                              static_cast<Real>(bs.vy));
                    tb.angularVelocity = static_cast<Real>(bs.w);
                    snap.bodies.push_back(tb);
                }

                // ---- force the lane -> constraint impulse copy-out ---------
                //
                // The colored solve accumulates each point's normal/tangent
                // impulse in the per-color ContactConstraintSimd SoA batches;
                // ctx.contacts[].points[].normalImpulse/tangentImpulse are only
                // refreshed by the StoreImpulses STAGE at the very end of the
                // step. Running the SAME routine that stage runs
                // (SimdSolve::StoreImpulses, ContactConstraintSimd.hpp:1009-1030
                // -- a pure lane->point copy of normalImpulse/tangentImpulse and
                // nothing else) gives every snapshot the ACCUMULATED impulses as
                // of that stage.
                //
                // BEHAVIOURALLY NEUTRAL, and pinned by the "bit-identical to
                // Step" case: between here and the real StoreImpulses stage
                // nothing reads a COLORED constraint's point impulses -- the
                // lane-wide SimdSolve::WarmStart / SolveNormalAndFriction /
                // ApplyRestitution passes read the batches, and the scalar
                // Overflow* trio reads only m_overflowRefs, which is disjoint
                // from every color bucket (SoftStep.cpp:929-931 buckets each
                // constraint into exactly one of the two). The overflow path
                // already mutates ctx.contacts[ref].points[] in place
                // (SoftStep::OverflowSolve), so overflow contacts are already
                // current with no help from us.
                for (std::size_t k = 0; k < static_cast<std::size_t>(kColorCount); ++k)
                {
                    const std::vector<ContactConstraintSimd>& batches =
                        (*sc.colorBatches)[k];
                    if (batches.empty()) { continue; }
                    SimdSolve::StoreImpulses(batches, sc.ctx->contacts,
                                             sc.colorRefs[k].data());
                }

                // ---- contacts: one record per manifold point ---------------
                std::size_t pointTotal = 0;
                for (std::uint32_t ci = 0; ci < sc.ctx->contactCount; ++ci)
                {
                    pointTotal += static_cast<std::size_t>(sc.ctx->contacts[ci].pointCount);
                }
                snap.contacts.reserve(pointTotal);

                for (std::uint32_t ci = 0; ci < sc.ctx->contactCount; ++ci)
                {
                    const ContactConstraint& cc = sc.ctx->contacts[ci];

                    // TGS pose deltas. A is ALWAYS an awake dynamic, so it has a
                    // row. B carries deltas only when it is a real DYNAMIC body
                    // -- the exact `dynB` gate SoftStep's OverflowSetup uses
                    // (SoftStep.cpp:552); a kinematic, static or span B never
                    // integrates, so its dp/dq are the zero identity.
                    const std::uint32_t rowA = AwakeIndexOf(cc.bodyA);
                    const Vec2 dpA(static_cast<Real>(sc.bodyState[rowA].dpx),
                                   static_cast<Real>(sc.bodyState[rowA].dpy));
                    const Real dqA = static_cast<Real>(sc.bodyState[rowA].dq);
                    Vec2 dpB(Real(0), Real(0));
                    Real dqB = Real(0);
                    const bool dynB = cc.bodyBIsBody && cc.invMassB > Real(0);
                    if (dynB)
                    {
                        const std::uint32_t rowB = AwakeIndexOf(cc.bodyB);
                        dpB = Vec2(static_cast<Real>(sc.bodyState[rowB].dpx),
                                   static_cast<Real>(sc.bodyState[rowB].dpy));
                        dqB = static_cast<Real>(sc.bodyState[rowB].dq);
                    }

                    // Fixture ids through the PERSISTENT pool. A transient tile-
                    // span constraint has no pool home (sourceContactId ==
                    // kNoContact) and keeps kInvalidSlot. m_graph is a private
                    // member of this class, so no friendship is needed.
                    std::uint32_t fixA = kInvalidSlot;
                    std::uint32_t fixB = kInvalidSlot;
                    if (cc.sourceContactId != ContactConstraint::kNoContact &&
                        m_graph.Pool().Alive(cc.sourceContactId))
                    {
                        const Contact& src = m_graph.Pool().Get(cc.sourceContactId);
                        fixA = src.a.index;
                        fixB = src.b.index;
                    }

                    // anchorA is (contact point - A's CENTRE OF MASS) as of
                    // Prepare (ConstraintGraph.cpp:1167), so the world point is
                    // A's mid-step COM + anchorA, and the mid-step COM is the
                    // start COM plus the accumulated dp.
                    const TraceStartPose& spA = traceStart[rowA];
                    const Vec2 comA = spA.pos
                                    + RotateVec(spA.angle, spA.localCenter) + dpA;

                    for (int p = 0; p < cc.pointCount; ++p)
                    {
                        const ContactConstraintPoint& cp = cc.points[p];
                        // TGS CURRENT separation, verbatim the solver's own form
                        // (ContactConstraintSimd.hpp:801-807, mirrored scalar in
                        // SoftStep::OverflowSolve):
                        //   s = baseSeparation
                        //       + dot((dpA + dqA x rA) - (dpB + dqB x rB), n)
                        // with CrossWR(w, r) == (-w*r.y, w*r.x) inlined. s > 0 is
                        // a speculative gap; s < 0 is penetration.
                        const Vec2 prA(dpA.x - dqA * cp.anchorA.y,
                                       dpA.y + dqA * cp.anchorA.x);
                        const Vec2 prB(dpB.x - dqB * cp.anchorB.y,
                                       dpB.y + dqB * cp.anchorB.x);

                        StepTraceContact tc;
                        tc.bodyA          = cc.bodyA;
                        tc.bodyB          = cc.bodyBIsBody ? cc.bodyB : kInvalidSlot;
                        tc.fixtureA       = fixA;
                        tc.fixtureB       = fixB;
                        tc.point          = comA + cp.anchorA;
                        tc.normal         = cc.normal;
                        tc.separation     = cp.baseSeparation
                                          + Geometry::Dot(prA - prB, cc.normal);
                        tc.normalImpulse  = cp.normalImpulse;
                        tc.tangentImpulse = cp.tangentImpulse;
                        snap.contacts.push_back(tc);
                    }
                }

                trace->snapshots.push_back(std::move(snap));
            };

            {
                ARCANE_STEPPROF_SCOPE(Solve);
                SolverContext ctx;
                ctx.world        = this;
                ctx.contacts     = m_contactConstraints.empty()
                                       ? nullptr
                                       : m_contactConstraints.data();
                ctx.contactCount = static_cast<std::uint32_t>(m_contactConstraints.size());
                ctx.joints       = m_jointConstraints.empty()
                                       ? nullptr
                                       : m_jointConstraints.data();
                ctx.jointCount   = static_cast<std::uint32_t>(m_jointConstraints.size());
                ctx.dt           = dt;
                ctx.substepCount = m_substepCount;
                ctx.invDt        = dt > Real(0) ? Real(1) / dt : Real(0);
                ctx.subDt        = dt / static_cast<Real>(m_substepCount);
                ctx.invSubDt     = ctx.subDt > Real(0) ? Real(1) / ctx.subDt : Real(0);
                ctx.gravity      = Vec2(m_gravityX, m_gravityY);
                ctx.executor     = Executor();   // Phase D1: always non-null (serial default)
                if (trace != nullptr) { ctx.traceHook = traceHookFn; }
                m_solver->Solve(ctx);
            }

            // ---- stage 3b: warm-start write-back (warm-start-on-Contact) -----
            // The solver no longer owns a warm-start cache: the converged per-point
            // impulses live on the persistent Contact. After Solve() (POST-
            // restitution) the graph writes each emitted point's impulses back onto
            // its source Contact's manifold (spans carry kNoContact and cold-start).
            // Decoupled: the solver never touches the pool; only the graph does the
            // round-trip (ConstraintGraph::WritebackImpulses, decomp step 2 Task 3).
            {
                ARCANE_STEPPROF_SCOPE(WarmStartWriteback);
                m_graph.WritebackImpulses(m_contactConstraints);
            }

            // ---- stage 4: bullet GJK-TOI clamp (P3.1 CCD) --------------------
            // The discrete CCD backup for `isBullet` bodies vs statics. Runs
            // AFTER the solver commits dynamic positions (so a dynamic bullet's
            // post-solve sweep is correct) and BEFORE island sleep + events (so
            // they observe the clamped position). Kinematic bullets -- which the
            // solver never touches -- are clamped from their stage-1-integrated
            // position; the speculative margin (stage 2) handles fast dynamics
            // inline, with this as the safety net. CCD runs after the solver
            // commits positions, before island/events.
            { ARCANE_STEPPROF_SCOPE(Bullet); BulletSweep(dt); }

            // ---- stage 5: island sleep bookkeeping (P2.4) --------------------
            // Build the per-Step constraint graph (bodies = nodes, THIS step's
            // contacts = edges), advance per-body sleep timers, and sleep any
            // island whose every member is idle past the threshold. Runs AFTER
            // the solve using this step's contacts; sleeping bodies are then
            // skipped by the next Step's contact feed + solver (the awake gate in
            // EmitContactConstraints), freezing their positions. Ports PhysicsWorld.lua:403-452.
            {
                ARCANE_STEPPROF_SCOPE(IslandSleep);
                // ---- deferred island split (quota-limited, Phase A) -----------------
                // Process AT MOST kMaxSplitsPerStep split-candidate islands per Step
                // (Box2D's amortization). Collect candidates ascending-id (determinism),
                // then split the first quota; the rest carry their flag to the next Step.
                // A single big pile that fractures resolves over a few steps.
                m_islandMgr.CollectSplitCandidates(m_splitCandidates);
                {
                    std::uint32_t processed = 0;
                    for (const std::uint32_t id : m_splitCandidates)
                    {
                        if (processed >= Island::kMaxSplitsPerStep) { break; }
                        SplitIsland(id);
                        ++processed;
                    }
                }

                m_islandMgr.UpdateSleep(*this, dt);
            }

            // ---- stage 6: event arrays (spec 2026-10-08 s6) -------------------
            // Hits, then the sensor pass (overlap state updates even when the
            // world gate is off), then sort for determinism (s6.3), then flip
            // the end buffers exactly as Box2D world.c:807-810: the buffer this
            // step wrote becomes readable, the other is cleared for the next
            // step and for destroys before it.
            {
                // Hits (Box2D solver.c:1758-1814): per solver contact that opted in,
                // the point with the largest approach speed above the threshold among
                // points that took normal impulse. Approach speed is
                // -ContactConstraintPoint::relativeVelocity, captured at Prepare
                // (SoftStep.cpp:270-274). Box2D copies that field onto the manifold
                // point as normalVelocity (contact_solver.c:502, :2097-2115) and the
                // hit loop reads -normalVelocity; Manifold2D does not store it, so
                // the constraint point is read directly. The impulse test is the
                // post-solve ManifoldPoint::normalImpulse > 0 -- there is no
                // totalNormalImpulse (spec amendment A4). Emit binds
                // `const Manifold& m = c.manifold` on the pool contact
                // (ConstraintGraph.cpp:1192) and copies manifold point p into
                // constraint point p (ConstraintGraph.cpp:1243-1261).
                // WritebackImpulses writes the impulse back by that same index
                // (ConstraintGraph.cpp:1378-1383), so p indexes both. Gated on
                // m_eventGate at push time; hits are not begin/end pairs (R10).
                if (m_eventGate)
                {
                    for (const ContactConstraint& cc : m_contactConstraints)
                    {
                        if (cc.sourceContactId == ContactConstraint::kNoContact) continue; // tile span
                        const Contact& c = m_graph.PoolContact(cc.sourceContactId);
                        if ((c.eventFlags & kEvHit) == 0u) continue;
                        Real best = m_hitEventThreshold;
                        int  bestP = -1;
                        for (int p = 0; p < cc.pointCount; ++p)
                        {
                            const Real approach = -cc.points[p].relativeVelocity;
                            if (approach > best && c.manifold.points[p].normalImpulse > Real(0))
                            {
                                best = approach;
                                bestP = p;
                            }
                        }
                        if (bestP < 0) continue;
                        ContactHitEvent e;
                        e.a = c.a; e.b = c.b;
                        e.bodyA = BodyHandle{ c.bodyA, c.genA };
                        e.bodyB = BodyHandle{ c.bodyB, c.genB };
                        e.point = c.manifold.points[bestP].point;
                        e.normal = Vec2(-c.manifold.normal.x, -c.manifold.normal.y); // B->A stored; A->B reported (A8, Manifold.hpp:33)
                        e.approachSpeed = best;
                        m_contactHitEvents.push_back(e);
                    }
                }

                // Sensor pass writes no simulation state. Begin pushes gate on
                // m_eventGate; Ends follow SensorOverlap::beginReported (R12).
                // Overlap memory always updates.
                RunSensorPass();

                const auto pairLess = [](const auto& l, const auto& r) noexcept
                {
                    if (l.a != r.a) return FixtureLess(l.a, r.a);
                    return FixtureLess(l.b, r.b);
                };
                const auto sensorLess = [](const auto& l, const auto& r) noexcept
                {
                    if (l.sensor != r.sensor) return FixtureLess(l.sensor, r.sensor);
                    return FixtureLess(l.visitor, r.visitor);
                };
                std::sort(m_contactBeginEvents.begin(), m_contactBeginEvents.end(), pairLess);
                std::sort(m_contactEndEvents[m_endEventIndex].begin(), m_contactEndEvents[m_endEventIndex].end(), pairLess);
                std::sort(m_contactHitEvents.begin(), m_contactHitEvents.end(), pairLess);
                std::sort(m_sensorBeginEvents.begin(), m_sensorBeginEvents.end(), sensorLess);
                std::sort(m_sensorEndEvents[m_endEventIndex].begin(), m_sensorEndEvents[m_endEventIndex].end(), sensorLess);
                m_endEventIndex = 1u - m_endEventIndex;
                m_contactEndEvents[m_endEventIndex].clear();
                m_sensorEndEvents[m_endEventIndex].clear();
            }
        }

        void PhysicsWorld::PushContactBegin(Contact& c)
        {
            if (!m_eventGate) return;
            m_contactBeginEvents.push_back(ContactBeginEvent{ c.a, c.b,
                BodyHandle{ c.bodyA, c.genA }, BodyHandle{ c.bodyB, c.genB } });
            c.beginReported = true; // R10: an End is delivered iff this Begin was
        }

        void PhysicsWorld::PushContactEnd(Contact& c)
        {
            // R10: iff a Begin was actually pushed, regardless of the gate now.
            // Into the CURRENT end buffer (Box2D world.c:668 / contact.c:364): a step
            // writes it, then flips at its end; a destroy between steps writes the
            // buffer the NEXT step will flip and deliver. Clear only after the push
            // so a throwing allocation does not drop the pairing bit.
            if (!c.beginReported) return;
            m_contactEndEvents[m_endEventIndex].push_back(ContactEndEvent{ c.a, c.b,
                BodyHandle{ c.bodyA, c.genA }, BodyHandle{ c.bodyB, c.genB } });
            c.beginReported = false;
        }

        ContactEvents PhysicsWorld::GetContactEvents() const noexcept
        {
            return ContactEvents{ m_contactBeginEvents,
                                  m_contactEndEvents[1u - m_endEventIndex],
                                  m_contactHitEvents };
        }

        void PhysicsWorld::PushSensorBegin(FixtureHandle sensor, BodyHandle sensorBody,
                                           SensorOverlap& overlap)
        {
            // R12: record delivery on the overlap RunSensorPass keeps (in `now`,
            // then swapped into m_sensorState). A closed gate drops the push and
            // leaves the bit clear, so a later End is not emitted.
            if (!m_eventGate) return;
            m_sensorBeginEvents.push_back(SensorBeginEvent{
                sensor, overlap.visitor, sensorBody, overlap.visitorBody });
            overlap.beginReported = true;
        }

        void PhysicsWorld::PushSensorEnd(FixtureHandle sensor, BodyHandle sensorBody,
                                         SensorOverlap& overlap)
        {
            // R12: iff a Begin was actually pushed, regardless of the gate now.
            // Into the CURRENT end buffer (Box2D world.c:668): this step's pass
            // writes it, then stage 6b flips. A sensor removed between steps is
            // ended by the next step's pass, which writes the buffer that step flips.
            // Clear only after the push so a throwing allocation does not drop the bit.
            if (!overlap.beginReported) return;
            m_sensorEndEvents[m_endEventIndex].push_back(SensorEndEvent{
                sensor, overlap.visitor, sensorBody, overlap.visitorBody });
            overlap.beginReported = false;
        }

        SensorEvents PhysicsWorld::GetSensorEvents() const noexcept
        {
            return SensorEvents{ m_sensorBeginEvents,
                                 m_sensorEndEvents[1u - m_endEventIndex] };
        }

        bool PhysicsWorld::FixturesOverlapExact(std::uint32_t fa, std::uint32_t fb) const
        {
            // Same world-transform composition as DebugCollide (this file, the
            // ComposeFixtureXf call there): body pose composed with the fixture local pose.
            const std::uint32_t ba = m_fxBody[fa];
            const std::uint32_t bb = m_fxBody[fb];
            const Transform xfA = ComposeFixtureXf(
                Vec2(m_posX[ba], m_posY[ba]), m_angle[ba],
                Vec2(m_fxLocalPosX[fa], m_fxLocalPosY[fa]), m_fxLocalAngle[fa]);
            const Transform xfB = ComposeFixtureXf(
                Vec2(m_posX[bb], m_posY[bb]), m_angle[bb],
                Vec2(m_fxLocalPosX[fb], m_fxLocalPosY[fb]), m_fxLocalAngle[fb]);
            // Margin 0, no trace: a point with separation > 0 is strict penetration
            // (Manifold.hpp:30-32). Matches ConstraintGraph.cpp exactlyOverlapping
            // (the loop over manifold points). An exact edge touch (separation == 0)
            // is not an overlap.
            const Manifold manifold = Collide(m_fxShape[fa], xfA, m_fxShape[fb], xfB,
                                              /*speculativeMargin*/ Real(0), nullptr);
            for (int p = 0; p < manifold.pointCount; ++p)
            {
                if (manifold.points[p].separation > Real(0))
                {
                    return true;
                }
            }
            return false;
        }

        // End-of-step sensor pass (Box2D 3.1.1 sensor.c b2OverlapSensors, spec s6.1):
        // every live sensor fixture with sensorEvents tests fixtures on bodies of
        // EVERY type (sensor.c:179-181 queries all trees). Skip: same body (:72),
        // filter (:77), visitor sensorEvents off (:66), visitor is a sensor
        // (amendment A1 -- not in 3.1.1). Overlap = a narrowphase point with
        // separation > 0 (margin 0, Collide). Overlaps are kept sorted per sensor and
        // diffed against last pass: new -> Begin, missing -> End (sensor.c:270-340).
        // A dead or recycled sensor slot (gen mismatch, or absent from the live set)
        // ends everything it held (sensor.c:158-165 for the flag; destroy ends the
        // overlaps the slot still holds). Ascending fixture-slot order + sorted
        // visitors = deterministic arrays. Writes event buffers and m_sensorState only.
        void PhysicsWorld::RunSensorPass()
        {
            // Liveness: a freed fixture slot keeps a NON-zero generation. RemoveBody
            // sets m_alive[idx] = 0 and bumps m_fxGen (this file, RemoveBody). The
            // authoritative live set is each alive body's m_bodyFixtures list.
            // m_count is the body-slot high-water (ConstraintGraph reads w.m_count
            // the same way). m_fxGen != 0 is NOT a live test.
            if (m_sensorState.size() < m_fxCount) m_sensorState.resize(m_fxCount);
            std::vector<std::uint32_t>& liveSensors = m_sensorSlotScratch;
            std::vector<std::uint8_t>&  visited     = m_sensorVisitedScratch;
            std::vector<BodyHandle>&    candidates  = m_sensorBodyScratch;
            liveSensors.clear();
            visited.assign(m_fxCount, std::uint8_t(0));
            for (std::uint32_t b = 0; b < m_count; ++b)
            {
                if (m_alive[b] == 0) continue;
                for (const std::uint32_t fi : m_bodyFixtures[b])
                {
                    if (m_fxSensor[fi] != 0u) liveSensors.push_back(fi);
                }
            }
            std::sort(liveSensors.begin(), liveSensors.end());
            std::vector<SensorOverlap> now;
            for (const std::uint32_t fi : liveSensors)
            {
                visited[fi] = 1;
                SensorState& st = m_sensorState[fi];
                if (st.gen != m_fxGen[fi] && !st.overlaps.empty()) // slot recycled into a new sensor
                {
                    for (SensorOverlap& o : st.overlaps)
                    {
                        PushSensorEnd(FixtureHandle{ fi, st.gen }, st.body, o);
                    }
                    st.overlaps.clear();
                }
                const std::uint32_t sb = m_fxBody[fi];
                st.gen  = m_fxGen[fi];
                st.body = HandleOf(sb);
                if (m_fxSensorEvents[fi] == 0u) // flag off: end what it held (sensor.c:158)
                {
                    for (SensorOverlap& o : st.overlaps)
                    {
                        PushSensorEnd(FixtureHandle{ fi, st.gen }, st.body, o);
                    }
                    st.overlaps.clear();
                    continue;
                }
                now.clear();
                QueryAABB(FixtureAabb(fi), candidates); // index-ordered, all body types
                for (const BodyHandle bh : candidates)
                {
                    if (bh.index == sb) continue; // same body (sensor.c:72)
                    for (const std::uint32_t vf : m_bodyFixtures[bh.index]) // live fixtures only
                    {
                        if (m_fxSensor[vf] != 0u || m_fxSensorEvents[vf] == 0u) continue;
                        if (!FixturesCollide(fi, vf)) continue; // filter (sensor.c:77)
                        if (!FixturesOverlapExact(fi, vf)) continue;
                        now.push_back(SensorOverlap{ FixtureHandle{ vf, m_fxGen[vf] }, bh });
                    }
                }
                std::sort(now.begin(), now.end(), [](const SensorOverlap& l, const SensorOverlap& r) noexcept
                          { return FixtureLess(l.visitor, r.visitor); });
                // Diff two sorted lists (sensor.c:270-340). A handle differs when
                // its index OR its generation does (FixtureLess).
                std::size_t i = 0, j = 0;
                while (i < st.overlaps.size() || j < now.size())
                {
                    if (j == now.size() || (i < st.overlaps.size() && FixtureLess(st.overlaps[i].visitor, now[j].visitor)))
                    {
                        PushSensorEnd(FixtureHandle{ fi, st.gen }, st.body, st.overlaps[i++]);
                    }
                    else if (i == st.overlaps.size() || FixtureLess(now[j].visitor, st.overlaps[i].visitor))
                    {
                        PushSensorBegin(FixtureHandle{ fi, st.gen }, st.body, now[j++]);
                    }
                    else
                    {
                        // Same visitor. `now[j]` is a fresh overlap; keep the bit so
                        // a Begin delivered on an earlier pass still closes (R12).
                        now[j].beginReported = st.overlaps[i].beginReported;
                        ++i;
                        ++j;
                    }
                }
                st.overlaps.swap(now);
            }
            // Dead sensors (body removed, fixture dropped, or the slot recycled into
            // a non-sensor): not in the live set, but still holding overlaps.
            // End each with the OLD handles, then drop the memory.
            for (std::uint32_t fi = 0; fi < m_fxCount; ++fi)
            {
                if (visited[fi] != 0u || m_sensorState[fi].overlaps.empty()) continue;
                SensorState& st = m_sensorState[fi];
                for (SensorOverlap& o : st.overlaps)
                {
                    PushSensorEnd(FixtureHandle{ fi, st.gen }, st.body, o);
                }
                st.overlaps.clear();
            }
        }

        // ----------------------------------------------------------------
        // Persistent-contact helpers (collision-rebuild Phase 3, Task 2).
        // ----------------------------------------------------------------

        bool PhysicsWorld::FixtureSlotLive(FixtureHandle h) const noexcept
        {
            return h.index < m_fxGen.size() &&
                   m_fxGen[h.index] == h.generation &&
                   m_fxGen[h.index] != 0u;
        }


        // Persistent incremental contact coloring (Phase C, Stage 2, Tasks 4-5)
        // MOVED to ConstraintGraph (decomp step 2 Task 2): AssignContactColor /
        // ReleaseContactColor / ContactColorOf / ValidatePersistentColoring /
        // ColoredContactCount + m_bodyColorMask / m_colorContacts now live there,
        // reached through the inline probe forwarders (hpp) and the internal
        // m_graph calls at the create/destroy/grow seams.


        Real PhysicsWorld::CcdFixtureToi(std::uint32_t body, std::uint32_t fi, bool movers)
        {
            // Box2D v3 b2SolveContinuous + b2ContinuousQueryCallback, for one fixture:
            //   * the sweep: the body's centre of mass moves from its start-of-step
            //     to its end-of-step position while its angle turns from the start
            //     to the end angle (Box2D's b2Sweep, linear in both); obstacles stand
            //     at their end-of-step poses, as Box2D's do.
            //   * time of impact by conservative advancement with the engine's own
            //     ShapeDistance: from the current separation d, advance by
            //     (d - target) / (|dc| + |da| * r), r the fixture's farthest point from
            //     the centre of mass -- no point of the fixture can close faster, so the
            //     advance never steps through contact. Stops within linearSlop.
            //   * obstacles: statics (and tile spans) always; kinematic and non-bullet
            //     dynamic bodies when `movers` (a dynamic bullet sweeps the kinematic
            //     and dynamic trees). Other bullets, the body itself and sensors are
            //     skipped; the collision filter decides exactly as for contacts.
            //   * only 0 < t counts for a DYNAMIC body (Box2D: 0 < fraction); a sweep
            //     that starts touching is retried with a small circle (B2_CORE_FRACTION
            //     0.25 * minExtent) about the shape's centroid -- sliding along a
            //     surface never touches it, but the body's core cannot pass through. A
            //     KINEMATIC bullet (Manifold2D's own; no solver stops it) is held by a
            //     start-touch when the step moves INTO it.
            const Shape& shape = m_fxShape[fi];
            const bool kinematic = static_cast<BodyType>(m_btype[body]) == BodyType::Kinematic;
            const Vec2 lc(m_localCenterX[body], m_localCenterY[body]);
            const Real a0 = m_prevAngle[body], a1 = m_angle[body];
            const auto rot = [](Real ang, Vec2 v) { const Real c = std::cos(ang), sn = std::sin(ang); return Vec2(c * v.x - sn * v.y, sn * v.x + c * v.y); };
            const Vec2 c0 = Vec2(m_prevX[body], m_prevY[body]) + rot(a0, lc);
            const Vec2 c1 = Vec2(m_posX[body], m_posY[body]) + rot(a1, lc);
            const Vec2 dc = c1 - c0;
            const Real da = a1 - a0;
            const Vec2 fxPos(m_fxLocalPosX[fi], m_fxLocalPosY[fi]);
            const Real fxAng = m_fxLocalAngle[fi];
            // the fixture's transform at sweep time t (local point `at`, in the fixture frame, as origin)
            const auto poseAt = [&](Real t, Vec2 at) -> Transform
            {
                const Vec2 c(c0.x + dc.x * t, c0.y + dc.y * t);
                const Real ang = a0 + da * t;
                const Vec2 origin = c - rot(ang, lc);
                const Transform fx = ComposeFixtureXf(origin, ang, fxPos, fxAng);
                return Transform{ fx.position + rot(fx.rotation, at), fx.rotation };
            };
            // the farthest any point of `sh` (placed at `at` in the fixture) gets from the COM
            const auto reachOf = [&](const Shape& sh, Vec2 at)
            {
                Real r = Real(0);
                for (const Vec2& v : sh.verts)
                {
                    const Vec2 q = fxPos + rot(fxAng, at + v) - lc;
                    r = std::max(r, std::sqrt(q.x * q.x + q.y * q.y));
                }
                return r + sh.radius;
            };
            const Real dcLen = std::sqrt(dc.x * dc.x + dc.y * dc.y);
            constexpr Real kTarget = kLinearSlop;
            constexpr Real kTol = Real(0.25) * kLinearSlop;
            // conservative advancement of `sh` at `at`; dist(Transform) -> ShapeDistanceResult.
            // Returns t in [0, 1] (1 = clear) and whether it was touching at the start.
            const auto advance = [&](const Shape& sh, Vec2 at, const auto& dist, bool& startTouch, Vec2& startNormal) -> Real
            {
                const Real bound = dcLen + std::abs(da) * reachOf(sh, at);
                Real t = Real(0);
                for (int it = 0; it < 40; ++it)
                {
                    const ShapeDistanceResult r = dist(sh, poseAt(t, at));
                    if (r.distance < kTarget + kTol)
                    {
                        if (it == 0) { startTouch = true; startNormal = r.normal; }
                        return t;
                    }
                    if (!(bound > Real(0))) { return Real(1); }
                    t += (r.distance - kTarget) / bound;
                    if (t >= Real(1)) { return Real(1); }
                }
                return Real(1);
            };
            const Real coreR = kinematic ? Real(0) : Real(0.25) * ShapeMinExtent(shape);
            const Shape core = MakeCircle(coreR > Real(0) ? coreR : Real(0.001));
            const Vec2 centroid = shape.kind == ShapeKind::Polygon ? shape.polyCentroid : Vec2(Real(0), Real(0));
            Real best = Real(1);
            const auto test = [&](const auto& dist)
            {
                bool touch = false;
                Vec2 n(Real(0), Real(0));
                const Real t = advance(shape, Vec2(Real(0), Real(0)), dist, touch, n);
                if (t >= Real(1)) { return; }
                if (!touch)
                {
                    best = std::min(best, t);
                    return;
                }
                if (kinematic)
                {
                    if (dc.x * n.x + dc.y * n.y < Real(0)) { best = Real(0); } // moving into it
                    return;
                }
                if (coreR > Real(0))
                {
                    bool coreTouch = false;
                    Vec2 cn(Real(0), Real(0));
                    const Real tc = advance(core, centroid, dist, coreTouch, cn);
                    if (!coreTouch && tc < best) { best = tc; }
                }
            };

            // candidates: the fixture's swept box over the step
            const Aabb2 boxA = shape.ComputeAABB(poseAt(Real(0), Vec2(Real(0), Real(0))));
            const Aabb2 boxB = shape.ComputeAABB(poseAt(Real(1), Vec2(Real(0), Real(0))));
            const Real spinPad = std::abs(da) * reachOf(shape, Vec2(Real(0), Real(0))); // mid-sweep corners
            Aabb2 swept;
            swept.min = Vec2(std::min(boxA.min.x, boxB.min.x) - kShapeCastTol - spinPad, std::min(boxA.min.y, boxB.min.y) - kShapeCastTol - spinPad);
            swept.max = Vec2(std::max(boxA.max.x, boxB.max.x) + kShapeCastTol + spinPad, std::max(boxA.max.y, boxB.max.y) + kShapeCastTol + spinPad);

            const auto admits = [&](std::uint32_t fj) { return FixturesCollide(fi, fj); };
            const auto testFixture = [&](std::uint32_t fj)
            {
                const std::uint32_t bj = m_fxBody[fj];
                const Transform xf = ComposeFixtureXf(
                    Vec2(m_posX[bj], m_posY[bj]), m_angle[bj],
                    Vec2(m_fxLocalPosX[fj], m_fxLocalPosY[fj]), m_fxLocalAngle[fj]);
                test([&](const Shape& sh, const Transform& sx) { return ShapeDistance(sh, sx, m_fxShape[fj], xf); });
            };

            StaticCandidates(swept, m_scratchSpans, m_scratchStatics, m_staticGridScratch);
            for (const Aabb2& span : m_scratchSpans) // tile terrain: no fixture, no filter
            {
                Vec2 poly[4];
                AabbToCorners(span, poly);
                test([&](const Shape& sh, const Transform& sx) { return ShapePolyDistance(sh, sx, poly, 4); });
            }
            for (const std::uint32_t idx : m_scratchStatics)
            {
                if (m_sensor[idx] != 0) { continue; }
                if (idx < m_bodyFixtures.size() && !m_bodyFixtures[idx].empty())
                {
                    for (const std::uint32_t fj : m_bodyFixtures[idx])
                    {
                        if (fj >= m_fxCount || m_fxGen[fj] == 0u || m_fxSensor[fj] != 0u || !admits(fj)) { continue; }
                        testFixture(fj);
                    }
                }
                else // a fixtureless legacy static: its single shape, unfiltered
                {
                    const Transform sx0{ Vec2(m_posX[idx], m_posY[idx]), m_angle[idx] };
                    test([&](const Shape& sh, const Transform& sx) { return ShapeDistance(sh, sx, m_shape[idx], sx0); });
                }
            }

            if (movers && !kinematic)
            {
                m_ccdMoverScratch.clear();
                m_fixtureBroadphase->QueryAABB(swept, m_ccdMoverScratch);
                for (const std::uint32_t fj : m_ccdMoverScratch)
                {
                    if (fj >= m_fxCount || m_fxGen[fj] == 0u || m_fxSensor[fj] != 0u) { continue; }
                    const std::uint32_t bj = m_fxBody[fj];
                    if (bj == body || m_alive[bj] == 0 || m_sensor[bj] != 0 || m_bullet[bj] != 0) { continue; }
                    if (!admits(fj)) { continue; }
                    testFixture(fj);
                }
            }
            return best;
        }

        void PhysicsWorld::BulletSweep(Real dt)
        {
            // P3.1 CCD bullet clamp (port of PhysicsWorld.lua:313-320). For each
            // alive `isBullet` body, cast the swept shape from its start-of-step
            // position (prev, snapshotted in Step stage 1) along this Step's net
            // displacement vs STATICS ONLY, and clamp the position to the time of
            // impact so a thin static wall cannot be tunneled.
            //
            // The Lua epsilon: clamp = max(0, hit.t - 0.001). Pulling the body a
            // hair short of the surface keeps it just OUTSIDE the wall (no
            // depenetration churn next Step).
            constexpr Real kBulletEpsilon = Real(0.001);

            //
            // Box2D v3 continuous for NON-bullets (b2FinalizeBodiesTask): an awake
            // dynamic body is FAST when its motion this step -- the larger of its
            // displacement plus spin times maxExtent and (|v| + |w| maxExtent) dt --
            // exceeds safetyFactor * minExtent; a fast body is swept like a bullet
            // but against STATIC geometry only (b2SolveContinuous queries only the
            // static tree for non-bullets). Without it a hard hit on a static wall
            // overshot the speculative gap in the impact step (the sequential
            // two-point solve spins the body) and sank in by ~0.1 m.
            for (std::uint32_t i = 0; i < m_count; ++i)
            {
                if (m_alive[i] == 0)
                {
                    continue;
                }
                const bool bullet = m_bullet[i] != 0;
                const bool dynamic = static_cast<BodyType>(m_btype[i]) == BodyType::Dynamic;
                if (!bullet && !(m_enableContinuous && dynamic && m_awakeIndex[i] != kNotAwake))
                {
                    continue;
                }

                const Vec2 prev(m_prevX[i], m_prevY[i]);
                const Vec2 curr(m_posX[i], m_posY[i]);
                const Vec2 delta = curr - prev;
                // No net travel AND no turn this Step -> nothing to sweep. (A body
                // spinning in place moves its extremities a long way while its origin
                // stays put; the sweep rotates, so it must still run.)
                if (delta.x == Real(0) && delta.y == Real(0) && m_angle[i] == m_prevAngle[i])
                {
                    continue;
                }
                if (!bullet)
                {
                    const Real spin = std::abs(m_angVel[i]) * m_maxExtent[i];
                    const Real vel = std::sqrt(m_velX[i] * m_velX[i] + m_velY[i] * m_velY[i]);
                    const Real moved = std::sqrt(delta.x * delta.x + delta.y * delta.y) + spin * dt;
                    const Real maxMotion = std::max(moved, (vel + spin) * dt);
                    if (!(maxMotion > m_safetyFactor[i] * m_minExtent[i]))
                    {
                        continue; // not fast: safe to advance
                    }
                }

                // INVARIANT: bullets are always movers (Kinematic or Dynamic);
                // a Static body is never flagged isBullet so this branch is
                // unreachable for statics.
                MOSAIC_ASSERT(static_cast<BodyType>(m_btype[i]) != BodyType::Static,
                              "bullets are never static");

                // Obstacles (CcdFixtureToi, after Box2D v3's b2SolveContinuous):
                // tile spans and static bodies for every bullet; kinematic and
                // non-bullet dynamic bodies too for a DYNAMIC bullet. Collision
                // filters and sensors are honoured, and a touch at the start of the
                // sweep is not an impact (a bullet resting on something moves on).
                //
                // T7 Part C (ROTATION + FIXTURE AWARE CCD): sweep EACH of the
                // bullet's fixtures with the body's REAL angle and the fixture's
                // local transform, keeping the EARLIEST TOI across all fixtures.
                // The fixture's start-of-step world pos uses `prev` and the body's
                // current angle (consistent with the existing positional-sweep
                // approximation -- the conservative-advancement holds the angle
                // fixed during the sweep; we use m_angle[i] rather than a per-step
                // angle history, matching the translational CCD model). A bullet
                // with no fixtures falls back to the legacy single m_shape at the
                // body's real angle.
                const Real bodyAngle = m_angle[i];
                bool haveHit = false;
                Real bestT = Real(1);

                const std::vector<std::uint32_t>* fxList = nullptr;
                if (i < m_bodyFixtures.size() && !m_bodyFixtures[i].empty())
                {
                    fxList = &m_bodyFixtures[i];
                }

                if (fxList != nullptr)
                {
                    for (const std::uint32_t fi : *fxList)
                    {
                        if (fi >= m_fxCount || m_fxGen[fi] == 0u)
                        {
                            continue;
                        }
                        // Fixture start-of-step world pos = prev + R(angle)*local;
                        // world angle = bodyAngle + fixtureLocalAngle.
                        const Real t = CcdFixtureToi(i, fi, /*movers=*/bullet);
                        if (t < bestT)
                        {
                            bestT   = t;
                            haveHit = true;
                        }
                    }
                }
                else
                {
                    // Legacy fallback: single m_shape at the body's real angle.
                    const std::optional<ShapeCastHit> hit =
                        ShapeCast(m_shape[i], prev, delta, ShapeCastOpts{}, bodyAngle);
                    // a touch at the start blocks only a kinematic bullet moving into it (see CcdFixtureToi)
                    const bool kin = static_cast<BodyType>(m_btype[i]) == BodyType::Kinematic;
                    const bool counts = hit && (hit->t > Real(0) ||
                        (kin && delta.x * hit->normal.x + delta.y * hit->normal.y < Real(0)));
                    if (counts && hit->t < bestT)
                    {
                        bestT   = hit->t;
                        haveHit = true;
                    }
                }

                if (!haveHit || bestT >= Real(1))
                {
                    continue; // clear sweep -> the integrated position stands
                }

                // Clamp to the EARLIEST TOI (a hair short, the Lua's hit.t-0.001).
                // The clamp is applied to the BODY position with the body's delta
                // (the fixture offsets ride along rigidly with the body).
                const Real clamp = std::max(Real(0), bestT - kBulletEpsilon);
                // the pose at the time of impact: the centre of mass and the angle both
                // interpolated (Box2D: c = lerp(c1, c2, t), q = nlerp(q1, q2, t))
                const Real ang = m_prevAngle[i] + (m_angle[i] - m_prevAngle[i]) * clamp;
                const Vec2 lcI(m_localCenterX[i], m_localCenterY[i]);
                const auto rotI = [](Real a, Vec2 v) { const Real c = std::cos(a), sn = std::sin(a); return Vec2(c * v.x - sn * v.y, sn * v.x + c * v.y); };
                const Vec2 cS = prev + rotI(m_prevAngle[i], lcI), cE = curr + rotI(m_angle[i], lcI);
                const Vec2 cT = cS + (cE - cS) * clamp;
                const Vec2 clamped = cT - rotI(ang, lcI);
                m_posX[i] = clamped.x;
                m_posY[i] = clamped.y;
                m_angle[i] = ang;
                // Time loss (Box2D b2SolveContinuous): the body keeps its velocity but
                // gives back the gravity it gained over the part of the step it lost
                // (world gravity plus the well, scaled like the integrator's).
                if (dynamic)
                {
                    Vec2 g(m_gravityX, m_gravityY);
                    if (m_gravityWell.enabled) { g = g + GravityWellAccel(clamped); }
                    const Real lost = (Real(1) - clamp) * dt * m_gravityScale[i];
                    m_velX[i] -= lost * g.x;
                    m_velY[i] -= lost * g.y;
                }
                // Unconditional: the assert above guarantees this is always a
                // mover (never Static), so the broadphase update always applies.
                // UpdateMoverProxies keeps all per-fixture mover-broadphase proxies
                // and the body's residency grid in sync (Phase 2, Task 3).
                UpdateMoverProxies(i);
            }
        }

        // ---- pull API for debug draw / inspection (P3.6) -------------------

        void PhysicsWorld::GetBodyContacts(BodyHandle h, std::vector<BodyContact>& out) const
        {
            out.clear();
            if (!IsValid(h)) return;
            // Touching solver contacts of this body, ascending pool id. Sleeping
            // bodies keep their pool contacts (ConstraintGraph.cpp UpdateOneContact
            // BothAsleep return, :591-596), so they are listed too. The stored
            // manifold normal points B -> A (Manifold.hpp:74; per-point :33).
            m_graph.ForEachPoolContact([&](std::uint32_t, const Contact& c)
            {
                if (!c.solverRelevant || !c.bIsBody || !c.touching) return;
                const Vec2 n = c.manifold.normal;
                if (c.bodyA == h.index && c.genA == h.generation)
                    out.push_back(BodyContact{ c.a, c.b, h, BodyHandle{ c.bodyB, c.genB },
                                               Vec2(-n.x, -n.y), c.manifold.pointCount });
                else if (c.bodyB == h.index && c.genB == h.generation)
                    out.push_back(BodyContact{ c.b, c.a, h, BodyHandle{ c.bodyA, c.genA },
                                               n, c.manifold.pointCount });
            });
        }

        void PhysicsWorld::ForEachContact(
            Mosaic::FunctionRef<void(std::uint32_t, std::uint32_t)> fn) const
        {
            // Every touching body-to-body pool contact, ascending id (spec s6.5).
            // Includes dynamic-vs-static.
            m_graph.ForEachPoolContact([&](std::uint32_t, const Contact& c)
            {
                if (c.bIsBody && c.touching && c.bodyB != kInvalidSlot) fn(c.bodyA, c.bodyB);
            });
        }


        // DebugValidateBodyContacts MOVED to IslandManager (decomp step 1 Task 3);
        // PhysicsWorld::DebugValidateBodyContacts() forwards to it (inline in .hpp).

        // ---- awake-set maintainers (Phase B, Task 2) ----------------------------
        //
        // AddToAwakeSet / RemoveFromAwakeSet keep m_awakeBodies + m_awakeIndex
        // in sync with the awake-dynamic population. Both are idempotent: a
        // double-add or double-remove is a no-op. The Step loops are NOT rerouted
        // here (Tasks 3/4 do that); these are PURE bookkeeping appended to every
        // existing wake/sleep seam so the invariant holds at all times.

        void PhysicsWorld::AddToAwakeSet(std::uint32_t slot) noexcept
        {
            // Only awake dynamics belong in the set; idempotent (already-present
            // or non-dynamic is a no-op). The size guard handles a slot index that
            // has not been grown into yet (should not occur in normal flow, but
            // guards against out-of-bound reads on a half-initialized world).
            if (slot >= m_awakeIndex.size()) { return; }
            if (static_cast<BodyType>(m_btype[slot]) != BodyType::Dynamic) { return; }
            if (m_awakeIndex[slot] != kNotAwake) { return; }  // already in the set
            m_awakeIndex[slot] = static_cast<std::uint32_t>(m_awakeBodies.size());
            m_awakeBodies.push_back(slot);
        }

        void PhysicsWorld::RemoveFromAwakeSet(std::uint32_t slot) noexcept
        {
            // Swap-remove: move the last element into the vacated position + patch
            // the moved element's back-index.  Idempotent: a slot that is already
            // absent (kNotAwake) or out of range is a no-op.
            if (slot >= m_awakeIndex.size()) { return; }
            const std::uint32_t pos = m_awakeIndex[slot];
            if (pos == kNotAwake) { return; }                  // not in the set
            const std::uint32_t last  = static_cast<std::uint32_t>(m_awakeBodies.size() - 1u);
            const std::uint32_t moved = m_awakeBodies[last];
            m_awakeBodies[pos]  = moved;   // fill the hole with the last element
            m_awakeIndex[moved] = pos;     // patch the moved element's back-index
            m_awakeBodies.pop_back();
            m_awakeIndex[slot]  = kNotAwake;
        }

        // ---- kinematic solver-set maintainers (Phase C, Task 1) -----------------
        //
        // AddToKinematicSet / RemoveFromKinematicSet keep m_kinematicBodies +
        // m_kinematicIndex in sync with the live-kinematic population. They mirror
        // AddToAwakeSet/RemoveFromAwakeSet exactly (idempotent swap-remove with a
        // back-index patch) but gate on BodyType::Kinematic instead of Dynamic, and
        // -- because kinematics never sleep -- they are wired ONLY at the AddBody /
        // RemoveBody seams (there is no sleep/wake path for this set). The solver is
        // NOT rerouted onto this list here (Task 2 does that); this is PURE
        // bookkeeping so behavior stays byte-identical.

        void PhysicsWorld::AddToKinematicSet(std::uint32_t slot) noexcept
        {
            // Only kinematics belong in the set; idempotent (already-present or
            // non-kinematic is a no-op). The size guard handles a slot index not
            // grown into yet (should not occur in normal flow, but guards against
            // out-of-bound reads on a half-initialized world).
            if (slot >= m_kinematicIndex.size()) { return; }
            if (static_cast<BodyType>(m_btype[slot]) != BodyType::Kinematic) { return; }
            if (m_kinematicIndex[slot] != kNotKinematic) { return; }  // already in the set
            m_kinematicIndex[slot] = static_cast<std::uint32_t>(m_kinematicBodies.size());
            m_kinematicBodies.push_back(slot);
        }

        void PhysicsWorld::RemoveFromKinematicSet(std::uint32_t slot) noexcept
        {
            // Swap-remove: move the last element into the vacated position + patch
            // the moved element's back-index. Idempotent: a slot that is already
            // absent (kNotKinematic) or out of range is a no-op.
            if (slot >= m_kinematicIndex.size()) { return; }
            const std::uint32_t pos = m_kinematicIndex[slot];
            if (pos == kNotKinematic) { return; }                  // not in the set
            const std::uint32_t last  = static_cast<std::uint32_t>(m_kinematicBodies.size() - 1u);
            const std::uint32_t moved = m_kinematicBodies[last];
            m_kinematicBodies[pos]  = moved;   // fill the hole with the last element
            m_kinematicIndex[moved] = pos;     // patch the moved element's back-index
            m_kinematicBodies.pop_back();
            m_kinematicIndex[slot]  = kNotKinematic;
        }

        int PhysicsWorld::QueryAABB(const Aabb2& box,
                                    std::vector<BodyHandle>& out) const
        {
            // Linear scan over alive slots, index-ordered (deterministic).
            // Ports queryAABB (the richer query suite is P1.9).
            out.clear();
            for (std::uint32_t i = 0; i < m_count; ++i)
            {
                if (m_alive[i] == 0)
                {
                    continue;
                }
                if (AabbOverlap(box, SlotAabb(i)))
                {
                    out.push_back(BodyHandle{ i, m_gen[i] });
                }
            }
            return static_cast<int>(out.size());
        }

        // NOTE: the spatial-query surface (Raycast / RayVsBody / LineOfSight /
        // ShapeCast / OverlapShape / StaticCandidates) lives in Queries.cpp
        // (same class, separate TU -- split out in P2.1 to keep this
        // orchestration TU readable once the dynamics stages landed). No API
        // change; PhysicsWorld.hpp still declares them.
        Body PhysicsWorld::GetBody(BodyHandle h) noexcept
        {
            return Body(this, h);
        }

    } // namespace Physics
} // namespace Manifold2D
