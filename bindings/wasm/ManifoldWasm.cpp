// Manifold2D -> WebAssembly bindings (embind).
//
// NOT part of the MSVC/premake build: premake's Manifold2D project globs
// include/** + src/** (premake5.lua:50) and this file is outside both, so only
// scripts/build-wasm.sh ever compiles it. That is also why <emscripten/*.h>
// here does not violate the repo's zero-external-dependency invariant, which
// scopes to include/ and src/.
//
// ONE class, ManifoldSim, wrapping one PhysicsWorld, plus two packed float
// buffers exposed as typed_memory_views: the live body array and the last
// StepTraced capture. Views, not embind value objects, so a whole 22-stop trace
// crosses the boundary in a single copy on the JS side.
//
// MEMORY GROWTH detaches views: JS must copy (slice()) what bodies()/trace()
// return before the next call into the module.
//
// Units are MKS, +Y is DOWN (the engine's default gravity is (0, 10)).

#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <Manifold2D/Physics/PhysicsTypes.hpp>
#include <Manifold2D/Physics/Shapes.hpp>
#include <Manifold2D/Physics/Fixture.hpp>
#include <Manifold2D/Physics/PhysicsWorld.hpp>
#include <Manifold2D/Physics/StepTrace.hpp>
#include <Manifold2D/Physics/Solver/SolverStages.hpp>   // StageType enumerators

#ifndef MANIFOLD_WASM_REV
    // scripts/build-wasm.sh passes -DMANIFOLD_WASM_REV="<short sha>"; this
    // fallback only keeps the TU compilable on its own.
    #define MANIFOLD_WASM_REV "unknown"
#endif

using namespace Manifold2D::Physics;

namespace
{
    // kPi comes from Manifold2D::Physics::kPi (PhysicsTypes.hpp) via the
    // using-directive above; a second local definition here would be
    // ambiguous at every unqualified use below.

    // The hero scene's dynamic-body material (contract section 3): the engine
    // default friction with a little bounce.
    constexpr Real kDynDensity     = Real(1);
    constexpr Real kDynFriction    = Real(0.4);
    constexpr Real kDynRestitution = Real(0.2);

    // The tumbler wall material: lower friction than the bodies so they slide
    // around the ring instead of sticking to it.
    constexpr Real kWallFriction    = Real(0.2);
    constexpr Real kWallRestitution = Real(0);

    // 0xFFFFFFFF ids go over the wire as -1 (contract section 4).
    float Id(std::uint32_t v)
    {
        return (v == kInvalidSlot) ? -1.0f : static_cast<float>(v);
    }

    // Hexagon wall k (k = 0..5). The ring's INNER faces sit at distance
    // `apothem` from the body origin. Edge k's outward normal is
    //   theta_k = k*pi/3 + pi/6,
    // so the wall box's centre is at (apothem + half) * (cos, sin)(theta_k) and
    // its LONG axis is perpendicular to that normal -- hence the +pi/2 on the
    // local angle, with MakeAabb's first half-extent (the local x axis) being
    // the long one.
    Real WallAngle(int k)
    {
        return static_cast<Real>(k) * kPi / Real(3) + kPi / Real(6) + kPi / Real(2);
    }
    Vec2 WallCentre(int k, Real apothem, Real half)
    {
        const Real theta = static_cast<Real>(k) * kPi / Real(3) + kPi / Real(6);
        const Real r     = apothem + half;
        return Vec2(r * std::cos(theta), r * std::sin(theta));
    }
} // namespace

class ManifoldSim
{
public:
    ManifoldSim(float gravityY, int substeps)
    {
        WorldDef wd;
        wd.gravityX     = Real(0);
        wd.gravityY     = static_cast<Real>(gravityY);
        wd.substepCount = static_cast<std::uint32_t>(substeps < 1 ? 1 : substeps);
        m_def   = wd;   // reported back by params(): the world was built from exactly this
        m_world = std::make_unique<PhysicsWorld>(wd);
        // The world's own SerialWorkScheduler: no threads, the deterministic
        // reference path, and the only option under -sENVIRONMENT=web without
        // pthreads.
        m_world->SetExecutor(nullptr);
    }

    // Six box fixtures on ONE kinematic body: the hexagonal tumbler ring.
    // apothem = R (the distance from the centre to an inner wall face);
    // wallThick = the walls' radial thickness. Returns the body slot.
    std::uint32_t addKinematicHexagon(float cx, float cy, float R, float wallThick)
    {
        const Real apothem = static_cast<Real>(R);
        const Real half    = static_cast<Real>(wallThick) * Real(0.5);
        // Half-length of a regular hexagon's edge at apothem a is
        // a * tan(pi/6) = a / sqrt(3). The extra `half` lengthens each wall by
        // its own half-thickness so neighbouring walls OVERLAP at the corners
        // instead of leaving a seam a small body could squeeze through.
        const Real edgeHalf = apothem * std::tan(kPi / Real(6)) + half;

        // Fixture 0 comes from the BodyDef (AddBody's back-compat auto-fixture);
        // BodyDef::localPos / localAngle flow through to it (PhysicsWorld.hpp:
        // 139-160, the T6 fix), so wall 0 needs no separate AddFixture.
        BodyDef bd;
        bd.type        = BodyType::Kinematic;
        bd.position    = Vec2(static_cast<Real>(cx), static_cast<Real>(cy));
        bd.shape       = MakeAabb(edgeHalf, half);
        bd.friction    = kWallFriction;
        bd.restitution = kWallRestitution;
        bd.localPos    = WallCentre(0, apothem, half);
        bd.localAngle  = WallAngle(0);
        const BodyHandle bh = m_world->AddBody(bd);

        for (int k = 1; k < 6; ++k)
        {
            FixtureDef fd;
            fd.shape       = MakeAabb(edgeHalf, half);
            fd.localPos    = WallCentre(k, apothem, half);
            fd.localAngle  = WallAngle(k);
            fd.friction    = kWallFriction;
            fd.restitution = kWallRestitution;
            m_world->AddFixture(bh, fd);
        }

        m_hex      = bh;
        m_hexAngle = Real(0);
        m_world->SetAngle(m_hex, m_hexAngle);
        // A spin set before the hexagon existed is applied now instead of lost.
        m_world->SetAngularVelocity(m_hex, m_hexOmega);
        return bh.index;
    }

    // A static box, e.g. the ground the hero's stack rests on. hw/hh are
    // half-extents. Returns the body slot, or kInvalidSlot for a non-positive
    // extent (a degenerate AABB would still reach the narrowphase).
    std::uint32_t addStaticBox(float x, float y, float hw, float hh, float angle)
    {
        if (!(hw > 0.0f) || !(hh > 0.0f)) { return kInvalidSlot; }
        BodyDef bd;
        bd.type        = BodyType::Static;
        bd.position    = Vec2(static_cast<Real>(x), static_cast<Real>(y));
        bd.shape       = MakeAabb(static_cast<Real>(hw), static_cast<Real>(hh));
        bd.friction    = kDynFriction;     // the engine default, same as the bodies on it
        bd.restitution = kWallRestitution;
        const BodyHandle bh = m_world->AddBody(bd);
        if (angle != 0.0f) { m_world->SetAngle(bh, static_cast<Real>(angle)); }
        return bh.index;
    }

    // The values this world was built with, for captions that quote them:
    //   [gravityY, substepCount, contactHertz, contactDampingRatio,
    //    restitutionThreshold, contactPushMaxVelocity, sleepThreshold,
    //    dynamicRestitution, dynamicFriction]
    // A typed_memory_view over wasm memory: copy it before the next call.
    emscripten::val params()
    {
        m_params = {
            static_cast<float>(m_def.gravityY),
            static_cast<float>(m_def.substepCount),
            static_cast<float>(m_def.contactHertz),
            static_cast<float>(m_def.contactDampingRatio),
            static_cast<float>(m_def.restitutionThreshold),
            static_cast<float>(m_def.contactPushMaxVelocity),
            static_cast<float>(m_def.sleepThreshold),
            static_cast<float>(kDynRestitution),
            static_cast<float>(kDynFriction)
        };
        return emscripten::val(emscripten::typed_memory_view(m_params.size(), m_params.data()));
    }

    // rad/s. The engine integrates a kinematic body's POSITION but never its
    // ANGLE (PhysicsWorld.cpp stage 1 is linear only), so step()/stepTraced()
    // advance m_hexAngle here and SetAngle before every Step. The angular
    // VELOCITY is still set on the body: the solver gathers a kinematic contact
    // endpoint's real velocity row (BodyStateStore::SyncInCompacted pass b) and
    // that is what pushes the tumbling bodies along the wall.
    void setHexagonSpin(float omega)
    {
        m_hexOmega = static_cast<Real>(omega);
        if (m_hex.generation != 0u)
        {
            m_world->SetAngularVelocity(m_hex, m_hexOmega);
        }
    }

    std::uint32_t addCircle(float x, float y, float r)
    {
        if (!(r > 0.0f)) { return kInvalidSlot; }
        BodyDef bd;
        bd.type        = BodyType::Dynamic;
        bd.position    = Vec2(static_cast<Real>(x), static_cast<Real>(y));
        bd.shape       = MakeCircle(static_cast<Real>(r));
        bd.density     = kDynDensity;
        bd.friction    = kDynFriction;
        bd.restitution = kDynRestitution;
        return m_world->AddBody(bd).index;
    }

    // hw/hh are HALF-EXTENTS (MakeAabb's units).
    std::uint32_t addBox(float x, float y, float hw, float hh, float angle)
    {
        if (!(hw > 0.0f) || !(hh > 0.0f)) { return kInvalidSlot; }
        BodyDef bd;
        bd.type        = BodyType::Dynamic;
        bd.position    = Vec2(static_cast<Real>(x), static_cast<Real>(y));
        bd.shape       = MakeAabb(static_cast<Real>(hw), static_cast<Real>(hh));
        bd.density     = kDynDensity;
        bd.friction    = kDynFriction;
        bd.restitution = kDynRestitution;
        // BodyDef has no angle field; the initial orientation is a SetAngle.
        const BodyHandle bh = m_world->AddBody(bd);
        m_world->SetAngle(bh, static_cast<Real>(angle));
        return bh.index;
    }

    // A regular `sides`-gon of circumradius `radius`, CCW, rotated by `angle`.
    std::uint32_t addPolygon(float x, float y, float radius, int sides, float angle)
    {
        // MakePolygon throws std::invalid_argument below 3 verts
        // (src/Physics/Shapes.cpp:332) and the wasm build has no exception
        // catching, so validate HERE: a bad call returns kInvalidSlot instead of
        // aborting the module. 8 is the hero's upper bound, well under
        // kMaxPolyVerts (128).
        if (sides < 3 || sides > 8 || !(radius > 0.0f))
        {
            return kInvalidSlot;
        }
        std::vector<Vec2> verts;
        verts.reserve(static_cast<std::size_t>(sides));
        for (int k = 0; k < sides; ++k)
        {
            // Unrotated local vertices; the orientation is the body's angle
            // (SetAngle below), exactly as addBox does it, so a consumer draws
            // vertex k at angle + 2*pi*k/sides with no correction.
            const Real t = Real(2) * kPi * static_cast<Real>(k) / static_cast<Real>(sides);
            verts.push_back(Vec2(static_cast<Real>(radius) * std::cos(t),
                                 static_cast<Real>(radius) * std::sin(t)));
        }

        BodyDef bd;
        bd.type        = BodyType::Dynamic;
        bd.position    = Vec2(static_cast<Real>(x), static_cast<Real>(y));
        bd.shape       = MakePolygon(verts);
        bd.density     = kDynDensity;
        bd.friction    = kDynFriction;
        bd.restitution = kDynRestitution;
        const BodyHandle bh = m_world->AddBody(bd);
        m_world->SetAngle(bh, static_cast<Real>(angle));
        return bh.index;
    }

    void step(float dt)
    {
        const Real h = static_cast<Real>(dt);
        advanceHexagon(h);
        m_world->Step(h);
    }

    // Step PLUS the 22-stop capture. The packed trace stays readable until the
    // next stepTraced().
    void stepTraced(float dt)
    {
        const Real h = static_cast<Real>(dt);
        advanceHexagon(h);
        m_trace.snapshots.clear();
        m_world->StepTraced(h, m_trace);
        packTrace();
    }

    std::uint32_t bodyCount() const
    {
        return m_world->Count();
    }

    // [x, y, angle, vx, vy, w] per SLOT, slot order, bodyCount * 6 floats. A
    // dead slot is six zeros so the indexing stays slot-addressable.
    emscripten::val bodies()
    {
        const std::uint32_t n = m_world->Count();
        m_bodies.assign(static_cast<std::size_t>(n) * 6u, 0.0f);
        for (std::uint32_t i = 0; i < n; ++i)
        {
            if (!m_world->Alive(i)) { continue; }
            const BodyHandle h = m_world->HandleOf(i);
            const Vec2 p = m_world->Position(h);
            const Vec2 v = m_world->Velocity(h);
            float* row = m_bodies.data() + static_cast<std::size_t>(i) * 6u;
            row[0] = static_cast<float>(p.x);
            row[1] = static_cast<float>(p.y);
            row[2] = static_cast<float>(m_world->GetAngle(h));
            row[3] = static_cast<float>(v.x);
            row[4] = static_cast<float>(v.y);
            row[5] = static_cast<float>(m_world->AngularVelocity(h));
        }
        return emscripten::val(
            emscripten::typed_memory_view(m_bodies.size(), m_bodies.data()));
    }

    emscripten::val trace()
    {
        return emscripten::val(
            emscripten::typed_memory_view(m_traceBuf.size(), m_traceBuf.data()));
    }

private:
    void advanceHexagon(Real dt)
    {
        if (m_hex.generation == 0u) { return; }   // no hexagon was added
        // Wrap so a tab left open for days never lets the float's ulp reach
        // the per-step increment.
        m_hexAngle = std::fmod(m_hexAngle + m_hexOmega * dt, Real(2) * kPi);
        m_world->SetAngle(m_hex, m_hexAngle);
    }

    // Contract section 4:
    //   [0] snapshotCount
    //   per snapshot: [stage, substep, bodyCount, contactCount]
    //     bodies:   [slot, x, y, angle, vx, vy, w]                      x 7
    //     contacts: [bodyA, bodyB, fixtureA, fixtureB, px, py, nx, ny,
    //                separation, normalImpulse, tangentImpulse]         x 11
    void packTrace()
    {
        std::size_t floats = 1u;
        for (const StepTraceSnapshot& s : m_trace.snapshots)
        {
            floats += 4u + s.bodies.size() * 7u + s.contacts.size() * 11u;
        }
        m_traceBuf.clear();
        m_traceBuf.reserve(floats);

        m_traceBuf.push_back(static_cast<float>(m_trace.snapshots.size()));
        for (const StepTraceSnapshot& s : m_trace.snapshots)
        {
            m_traceBuf.push_back(static_cast<float>(static_cast<std::uint8_t>(s.stage)));
            m_traceBuf.push_back(static_cast<float>(s.substep));
            m_traceBuf.push_back(static_cast<float>(s.bodies.size()));
            m_traceBuf.push_back(static_cast<float>(s.contacts.size()));

            for (const StepTraceBody& b : s.bodies)
            {
                m_traceBuf.push_back(Id(b.body));
                m_traceBuf.push_back(static_cast<float>(b.position.x));
                m_traceBuf.push_back(static_cast<float>(b.position.y));
                m_traceBuf.push_back(static_cast<float>(b.angle));
                m_traceBuf.push_back(static_cast<float>(b.velocity.x));
                m_traceBuf.push_back(static_cast<float>(b.velocity.y));
                m_traceBuf.push_back(static_cast<float>(b.angularVelocity));
            }
            for (const StepTraceContact& c : s.contacts)
            {
                m_traceBuf.push_back(Id(c.bodyA));
                m_traceBuf.push_back(Id(c.bodyB));
                m_traceBuf.push_back(Id(c.fixtureA));
                m_traceBuf.push_back(Id(c.fixtureB));
                m_traceBuf.push_back(static_cast<float>(c.point.x));
                m_traceBuf.push_back(static_cast<float>(c.point.y));
                m_traceBuf.push_back(static_cast<float>(c.normal.x));
                m_traceBuf.push_back(static_cast<float>(c.normal.y));
                m_traceBuf.push_back(static_cast<float>(c.separation));
                m_traceBuf.push_back(static_cast<float>(c.normalImpulse));
                m_traceBuf.push_back(static_cast<float>(c.tangentImpulse));
            }
        }
    }

    std::unique_ptr<PhysicsWorld> m_world;   // PhysicsWorld is non-copyable
    BodyHandle                    m_hex{};   // kInvalidBody until addKinematicHexagon
    Real                          m_hexAngle = Real(0);
    Real                          m_hexOmega = Real(0);
    StepTrace                     m_trace;
    std::vector<float>            m_bodies;
    std::vector<float>            m_traceBuf;
    WorldDef                      m_def;      // what the world was built from (params())
    std::array<float, 9>          m_params{}; // params()'s backing store
};

EMSCRIPTEN_BINDINGS(manifold)
{
    // The engine commit these artifacts were built from. embind puts the NAME in
    // the wasm data segment, not in manifold.js -- scripts/build-wasm.sh appends
    // a `// MANIFOLD_WASM_REV=<sha>` comment to the JS for the deploy check.
    emscripten::constant("MANIFOLD_WASM_REV", std::string(MANIFOLD_WASM_REV));

    emscripten::class_<ManifoldSim>("ManifoldSim")
        .constructor<float, int>()
        .function("addKinematicHexagon", &ManifoldSim::addKinematicHexagon)
        .function("addStaticBox",        &ManifoldSim::addStaticBox)
        .function("setHexagonSpin",      &ManifoldSim::setHexagonSpin)
        .function("addCircle",           &ManifoldSim::addCircle)
        .function("addBox",              &ManifoldSim::addBox)
        .function("addPolygon",          &ManifoldSim::addPolygon)
        .function("step",                &ManifoldSim::step)
        .function("stepTraced",          &ManifoldSim::stepTraced)
        .function("bodyCount",           &ManifoldSim::bodyCount)
        .function("bodies",              &ManifoldSim::bodies)
        .function("trace",               &ManifoldSim::trace)
        .function("params",              &ManifoldSim::params);
    // embind gives every class_ a .delete() automatically -- contract section 3's
    // sim.delete() needs no registration.
}
