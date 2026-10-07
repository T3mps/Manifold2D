// Cross-platform determinism fixture: bit-exact state hashes that CI compares
// ACROSS legs (Windows/MSVC, Linux/GCC+Clang, macOS/Apple Clang; Debug+Release).
//
// The in-process determinism gates (PhysicsDeterminismTest.cpp, the P2.6 harness,
// the MT-invariance suites) prove run-twice / serial-vs-MT identity inside ONE
// binary. They cannot see a divergence between two compilers or two libms. This
// file closes that gap: each scene is stepped and every body's raw float bits
// (x, y, angle, vx, vy, w) are folded into an FNV-1a hash per step. In-process
// it asserts run-twice identity (and serial == 4-worker MT for the trig-free
// scenes). When MANIFOLD2D_DETERMINISM_OUT names a file, it also appends one
// line per scene to it:
//
//   scene <name> <class> <hash:016x>
//   check <name> <step> <hash:016x>          (running hash at step checkpoints)
//   state <name> <body> <x> <y> <angle>      (final step, %a hex floats)
//   probe <runtime|folded> <hash:016x>       (libm fingerprint, see below)
//
// scripts/compare-determinism.py diffs those files across the CI legs.
//
// <class> encodes what the scene's math is allowed to depend on:
//   trigfree -- every body is fixedRotation, so the only transcendental the
//               pipeline evaluates is sin/cos(0) (exact on every libm). The math
//               is IEEE-754 + - * / sqrt + explicit fma only, all correctly
//               rounded under -ffp-contract=off / /fp:strict. MUST be bit-equal
//               on every OS, compiler and configuration.
//   trig     -- free rotation: std::sin/std::cos of non-trivial angles feed
//               back into the state. Those calls go to the platform libm (MSVC
//               UCRT, glibc, Apple libm), which C++ does not require to be
//               correctly rounded, so bits may legitimately differ BETWEEN
//               libms. MUST be bit-equal across compilers + configs sharing a
//               libm (all Linux legs; Debug vs Release on one OS).
//
// The two `probe` lines fingerprint the platform's float math itself, so a
// divergence can be attributed instead of guessed at: `runtime` hashes
// sin/cos/sqrt of angles the compiler cannot see (read through volatile), i.e.
// the libm the binary actually calls; `folded` hashes the same values passed as
// literals, which the optimizer may evaluate at compile time instead (LLVM folds
// sinf as double sin + round to float, which can differ from the runtime sinf by
// an ulp). runtime differing between two legs means different libms; folded
// differing while runtime matches means one leg constant-folded.
//
// The scenes are test-only consumers of the public API; nothing here changes
// library behaviour.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <Manifold2D/Physics/PhysicsTypes.hpp>
#include <Manifold2D/Physics/Shapes.hpp>
#include <Manifold2D/Physics/Body.hpp>
#include <Manifold2D/Physics/PhysicsWorld.hpp>
#include <Manifold2D/Physics/Joints/Joints.hpp>

#include "Support/TestWorkScheduler.hpp"

using namespace Manifold2D::Physics;

namespace
{
    constexpr Real kDt    = Real(1) / Real(60);
    constexpr int  kSteps = 300;

    constexpr std::uint64_t kFnvBasis = 0xcbf29ce484222325ull;

    // Steps at which the running hash is recorded, so the CI comparison can
    // report the FIRST step at which two legs part company (step 1 = the very
    // first Step's trig / contact math; a later step = accumulation).
    constexpr int kCheckpoints[] = { 1, 2, 5, 10, 30, 60, 120, 300 };

    struct SceneResult
    {
        std::uint64_t     hash = kFnvBasis;              // FNV-1a offset basis
        std::vector<Real> finalState;                    // x, y, angle per body
        std::vector<std::pair<int, std::uint64_t>> checkpoints;
    };

    void FoldBits(std::uint64_t& h, Real v)
    {
        std::uint32_t bits = 0;
        static_assert(sizeof(bits) == sizeof(v));
        std::memcpy(&bits, &v, sizeof(bits));
        for (int i = 0; i < 4; ++i)
        {
            h ^= (bits >> (8 * i)) & 0xffu;
            h *= 0x100000001b3ull; // FNV-1a prime
        }
    }

    void Record(PhysicsWorld& w, const std::vector<BodyHandle>& bodies, SceneResult& r, bool last)
    {
        for (BodyHandle b : bodies)
        {
            const Vec2 p = w.Position(b);
            const Vec2 v = w.Velocity(b);
            const Real a = w.GetAngle(b);
            FoldBits(r.hash, p.x);
            FoldBits(r.hash, p.y);
            FoldBits(r.hash, a);
            FoldBits(r.hash, v.x);
            FoldBits(r.hash, v.y);
            FoldBits(r.hash, w.AngularVelocity(b));
            if (last)
            {
                r.finalState.push_back(p.x);
                r.finalState.push_back(p.y);
                r.finalState.push_back(a);
            }
        }
    }

    void Checkpoint(SceneResult& r, int step)
    {
        for (int c : kCheckpoints)
        {
            if (c == step)
            {
                r.checkpoints.emplace_back(step, r.hash);
            }
        }
    }

    Shape BoxPolygon(Real hw, Real hh)
    {
        return MakePolygon({ Vec2(-hw, -hh), Vec2(hw, -hh), Vec2(hw, hh), Vec2(-hw, hh) });
    }

    // Trig-free: a pile of fixedRotation boxes + circles dropped into a static
    // bin. Exercises broadphase, AABB/circle narrowphase, islands, the graph-
    // colored (SIMD) contact solver, restitution, friction and sleeping.
    SceneResult RunFixedRotationPile(BroadphaseKind bp, Mosaic::IWorkScheduler* exec)
    {
        WorldDef wdef;
        wdef.gravityY   = Real(10);
        wdef.broadphase = bp;
        PhysicsWorld w(wdef);
        if (exec != nullptr)
        {
            w.SetExecutor(exec);
        }

        auto addStatic = [&](Vec2 pos, Real hw, Real hh) {
            BodyDef d;
            d.type     = BodyType::Static;
            d.position = pos;
            d.shape    = MakeAabb(hw, hh);
            w.AddBody(d);
        };
        addStatic(Vec2(Real(10), Real(20)), Real(12), Real(1));   // floor (top at y=19)
        addStatic(Vec2(Real(-2), Real(12)), Real(0.5f), Real(8)); // left wall
        addStatic(Vec2(Real(22), Real(12)), Real(0.5f), Real(8)); // right wall

        std::vector<BodyHandle> bodies;
        for (int col = 0; col < 4; ++col)
        {
            for (int row = 0; row < 6; ++row)
            {
                BodyDef d;
                d.type          = BodyType::Dynamic;
                d.position      = Vec2(Real(2 + col * 5) + Real(0.05f) * Real(row),
                                       Real(18.5f) - Real(row) * Real(1.05f));
                d.shape         = MakeAabb(Real(0.5f), Real(0.5f));
                d.fixedRotation = true;
                d.friction      = Real(0.6f);
                bodies.push_back(w.AddBody(d));
            }
        }
        for (int i = 0; i < 24; ++i)
        {
            BodyDef d;
            d.type          = BodyType::Dynamic;
            d.position      = Vec2(Real(1) + Real(i % 8) * Real(2.3f), Real(2) - Real(i / 8) * Real(1.7f));
            d.shape         = MakeCircle(Real(0.35f) + Real(0.05f) * Real(i % 3));
            d.fixedRotation = true;
            d.restitution   = Real(0.2f);
            d.friction      = Real(0.3f);
            bodies.push_back(w.AddBody(d));
        }

        SceneResult r;
        for (int step = 1; step <= kSteps; ++step)
        {
            if (step == 90)
            {
                w.ApplyImpulse(bodies[3], Vec2(Real(4), Real(-6)));
            }
            w.Step(kDt);
            Record(w, bodies, r, step == kSteps);
            Checkpoint(r, step);
        }
        return r;
    }

    // Trig: freely-rotating polygons tumbling onto a slab, a revolute pendulum
    // chain, a weld pair and a bullet fired at a thin wall (CCD). Every rotation
    // path (rotation-aware narrowphase, joint angle math, sin/cos transforms)
    // feeds the state.
    SceneResult RunRotatingMixed()
    {
        WorldDef wdef;
        wdef.gravityY   = Real(10);
        wdef.broadphase = BroadphaseKind::Tree;
        PhysicsWorld w(wdef);

        {
            BodyDef d;
            d.type     = BodyType::Static;
            d.position = Vec2(Real(20), Real(25));
            d.shape    = MakeAabb(Real(30), Real(1.2f));
            w.AddBody(d);
        }
        {
            BodyDef d;
            d.type     = BodyType::Static;
            d.position = Vec2(Real(35), Real(10));
            d.shape    = MakeAabb(Real(0.1f), Real(8));
            w.AddBody(d);
        }

        std::vector<BodyHandle> bodies;
        for (int i = 0; i < 8; ++i)
        {
            BodyDef d;
            d.type        = BodyType::Dynamic;
            d.position    = Vec2(Real(14) + Real(i % 4) * Real(1.6f), Real(20) - Real(i / 4) * Real(2.5f));
            d.shape       = BoxPolygon(Real(0.6f), Real(0.4f));
            d.friction    = Real(0.5f);
            d.restitution = Real(0.1f);
            const BodyHandle b = w.AddBody(d);
            w.SetAngle(b, Real(0.3f) * Real(i + 1));
            bodies.push_back(b);
        }

        BodyHandle anchor;
        {
            BodyDef d;
            d.type     = BodyType::Static;
            d.position = Vec2(Real(6), Real(4));
            d.shape    = MakeCircle(Real(0.2f));
            anchor     = w.AddBody(d);
        }
        BodyHandle prev = anchor;
        for (int i = 0; i < 4; ++i)
        {
            BodyDef d;
            d.type     = BodyType::Dynamic;
            d.position = Vec2(Real(6) + Real(1.2f) * Real(i + 1), Real(4));
            d.shape    = MakeCapsule(Real(0.4f), Real(0.15f));
            const BodyHandle link = w.AddBody(d);
            JointDef jd;
            jd.kind   = JointKind::Revolute;
            jd.a      = prev;
            jd.b      = link;
            jd.anchor = Vec2(Real(6) + Real(1.2f) * Real(i) + Real(0.6f), Real(4));
            w.AddJoint(jd);
            bodies.push_back(link);
            prev = link;
        }

        BodyHandle weldA;
        BodyHandle weldB;
        {
            BodyDef d;
            d.type     = BodyType::Dynamic;
            d.position = Vec2(Real(26), Real(15));
            d.shape    = BoxPolygon(Real(0.5f), Real(0.5f));
            weldA      = w.AddBody(d);
            d.position = Vec2(Real(27.2f), Real(15));
            d.shape    = MakeCircle(Real(0.5f));
            weldB      = w.AddBody(d);
            JointDef jd;
            jd.kind   = JointKind::Weld;
            jd.a      = weldA;
            jd.b      = weldB;
            jd.anchor = Vec2(Real(26.6f), Real(15));
            w.AddJoint(jd);
            bodies.push_back(weldA);
            bodies.push_back(weldB);
        }

        {
            BodyDef d;
            d.type     = BodyType::Dynamic;
            d.position = Vec2(Real(10), Real(10));
            d.shape    = MakeCircle(Real(0.2f));
            d.bullet   = true;
            const BodyHandle b = w.AddBody(d);
            w.SetVelocity(b, Vec2(Real(150), Real(0)));
            bodies.push_back(b);
        }

        SceneResult r;
        for (int step = 1; step <= kSteps; ++step)
        {
            if (step == 60)
            {
                w.ApplyImpulse(weldA, Vec2(Real(-3), Real(-2)));
            }
            w.Step(kDt);
            Record(w, bodies, r, step == kSteps);
            Checkpoint(r, step);
        }
        return r;
    }

    void Emit(const char* name, const char* cls, const SceneResult& r)
    {
        const char* path = std::getenv("MANIFOLD2D_DETERMINISM_OUT");
        if (path == nullptr || *path == '\0')
        {
            return;
        }
        std::FILE* f = std::fopen(path, "a");
        REQUIRE(f != nullptr);
        std::fprintf(f, "scene %s %s %016llx\n", name, cls, static_cast<unsigned long long>(r.hash));
        for (const auto& [step, h] : r.checkpoints)
        {
            std::fprintf(f, "check %s %d %016llx\n", name, step, static_cast<unsigned long long>(h));
        }
        for (std::size_t i = 0; i + 2 < r.finalState.size(); i += 3)
        {
            std::fprintf(f, "state %s %zu %a %a %a\n", name, i / 3,
                         static_cast<double>(r.finalState[i]),
                         static_cast<double>(r.finalState[i + 1]),
                         static_cast<double>(r.finalState[i + 2]));
        }
        std::fclose(f);
    }
}

namespace
{
    // One probe step: sin, cos and sqrt of `a`, folded into h.
#define MANIFOLD2D_PROBE_STEP(h, a)                 \
    do {                                            \
        FoldBits((h), std::sin(static_cast<Real>(a)));  \
        FoldBits((h), std::cos(static_cast<Real>(a)));  \
        FoldBits((h), std::sqrt(std::fabs(static_cast<Real>(a)))); \
    } while (false)

    // Angles read through volatile: the compiler cannot see the values, so these
    // are genuine calls into the platform libm.
    std::uint64_t ProbeRuntime()
    {
        static volatile float angles[] = {
            0.3f, 0.6f, 0.9f, 1.2f, 1.5f, 2.0f, 2.5f, 3.0f, -0.7f, 12.5f, 100.25f, 1000.5f,
        };
        std::uint64_t h = kFnvBasis;
        for (volatile float& a : angles)
        {
            const float v = a;
            MANIFOLD2D_PROBE_STEP(h, v);
        }
        return h;
    }

    // The same values as literals: the optimizer is free to evaluate these at
    // compile time rather than calling libm.
    std::uint64_t ProbeFolded()
    {
        std::uint64_t h = kFnvBasis;
        MANIFOLD2D_PROBE_STEP(h, 0.3f);
        MANIFOLD2D_PROBE_STEP(h, 0.6f);
        MANIFOLD2D_PROBE_STEP(h, 0.9f);
        MANIFOLD2D_PROBE_STEP(h, 1.2f);
        MANIFOLD2D_PROBE_STEP(h, 1.5f);
        MANIFOLD2D_PROBE_STEP(h, 2.0f);
        MANIFOLD2D_PROBE_STEP(h, 2.5f);
        MANIFOLD2D_PROBE_STEP(h, 3.0f);
        MANIFOLD2D_PROBE_STEP(h, -0.7f);
        MANIFOLD2D_PROBE_STEP(h, 12.5f);
        MANIFOLD2D_PROBE_STEP(h, 100.25f);
        MANIFOLD2D_PROBE_STEP(h, 1000.5f);
        return h;
    }

#undef MANIFOLD2D_PROBE_STEP

    void EmitProbe(const char* name, std::uint64_t h)
    {
        const char* path = std::getenv("MANIFOLD2D_DETERMINISM_OUT");
        if (path == nullptr || *path == '\0')
        {
            return;
        }
        std::FILE* f = std::fopen(path, "a");
        REQUIRE(f != nullptr);
        std::fprintf(f, "probe %s %016llx\n", name, static_cast<unsigned long long>(h));
        std::fclose(f);
    }
}

TEST_CASE("Cross-platform determinism: libm fingerprint", "[determinism][xplat]")
{
    // Diagnostic only: these hashes attribute a scene divergence to the libm or
    // to compile-time folding. A difference is reported by CI, never failed --
    // the library's own guarantees are the scene gates below.
    EmitProbe("runtime", ProbeRuntime());
    EmitProbe("folded", ProbeFolded());
    SUCCEED("libm fingerprint emitted");
}

TEST_CASE("Cross-platform determinism: trig-free pile (all broadphases, serial + MT)", "[determinism][xplat]")
{
    Manifold2D::Testing::TestWorkScheduler pool(4);

    const struct { BroadphaseKind kind; const char* name; } kinds[] = {
        { BroadphaseKind::Tree, "pile-tree" },
        { BroadphaseKind::Hash, "pile-hash" },
        { BroadphaseKind::Sap,  "pile-sap"  },
    };
    for (const auto& k : kinds)
    {
        const SceneResult a  = RunFixedRotationPile(k.kind, nullptr);
        const SceneResult b  = RunFixedRotationPile(k.kind, nullptr);
        const SceneResult mt = RunFixedRotationPile(k.kind, &pool);
        INFO(k.name);
        CHECK(a.hash == b.hash);
        CHECK(a.hash == mt.hash);
        CHECK(a.finalState == b.finalState);
        Emit(k.name, "trigfree", a);
    }
}

TEST_CASE("Cross-platform determinism: rotating mixed scene", "[determinism][xplat]")
{
    const SceneResult a = RunRotatingMixed();
    const SceneResult b = RunRotatingMixed();
    CHECK(a.hash == b.hash);
    CHECK(a.finalState == b.finalState);
    Emit("rotating-mixed", "trig", a);
}
