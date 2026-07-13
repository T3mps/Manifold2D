// [perf] wall-time tripwire: the MKS units conversion restored the full physics
// suite from ~1087 s (px era -- px content thrashing the 1 m residency tile /
// broadphase) to ~96 s. This asserts a representative dense many-body step stays
// fast, catching an ORDER-OF-MAGNITUDE regression (not 2x machine variance -- the
// ceiling carries ~10x headroom). Tagged [perf] so loaded machines exclude it
// (ArcaneTests.exe ~[perf]).
//
// Survey MKS P5 C Item-3: no in-repo wall-time-assert precedent exists (no
// [perf] tag, no Catch2 duration listener, no Jenkinsfile timeout()). This is a
// targeted, EXCLUDABLE in-suite tripwire (mechanism 1 of that survey) rather than
// a full-suite wall-clock assert (unmeasurable from inside one test case and
// flake-prone) or a CI-side timeout (mechanism 3) -- it reconstructs the
// workload CLASS that was pathological at px scale (a dense grid of bodies
// falling and piling inside a walled box, thrashing broadphase pair churn) and
// times just that, with enough headroom to never trip on ordinary machine
// variance.
#include <chrono>

#include <catch2/catch_test_macros.hpp>

#include <Manifold2D/Physics/PhysicsWorld.hpp>
#include <Manifold2D/Physics/Shapes.hpp>
#include <Manifold2D/Physics/Body.hpp>

using namespace Manifold2D::Physics;

TEST_CASE("PhysicsPerf: dense broadphase step stays fast at MKS", "[perf][physics]")
{
    // Meter-scale walled box (mirrors the Sandbox scene-0 / stress-scene arena
    // shape: a wide static floor + 2 tall static side walls) + a dense grid of
    // dynamic circles falling and piling inside it -- the workload class that
    // was pathologically slow at px scale, when px content lived inside a
    // single 1 m residency tile and thrashed the broadphase.
    WorldDef wd; // MKS defaults (gravityY = 10, sleepThreshold 0.05, ...)
    PhysicsWorld w(wd);

    const Real kFloorTopY  = Real(8);
    const Real kFloorHalfW = Real(8.5);
    const Real kFloorHalfH = Real(0.4);
    const Real kWallHalfW  = Real(0.5);
    const Real kWallHalfH  = Real(10);

    // Floor: static Aabb, top surface at kFloorTopY, spanning x in [-8.5, 8.5].
    {
        BodyDef bd;
        bd.type     = BodyType::Static;
        bd.position = Vec2(Real(0), kFloorTopY + kFloorHalfH);
        bd.shape    = MakeAabb(kFloorHalfW, kFloorHalfH);
        w.AddBody(bd);
    }
    // Left + right walls: static Aabb, inner faces flush with the floor's edges,
    // tall enough to contain the falling grid.
    {
        BodyDef bd;
        bd.type     = BodyType::Static;
        bd.position = Vec2(-(kFloorHalfW + kWallHalfW), kFloorTopY - kWallHalfH);
        bd.shape    = MakeAabb(kWallHalfW, kWallHalfH);
        w.AddBody(bd);
    }
    {
        BodyDef bd;
        bd.type     = BodyType::Static;
        bd.position = Vec2(kFloorHalfW + kWallHalfW, kFloorTopY - kWallHalfH);
        bd.shape    = MakeAabb(kWallHalfW, kWallHalfH);
        w.AddBody(bd);
    }

    // A dense grid of 300 dynamic circles (r = 0.1 m) inside the box -- enough
    // to exercise real broadphase pair churn as they fall and pile (the whole
    // point of this workload; a handful of bodies would make the broadphase
    // trivial and defeat the tripwire).
    const int  kCols   = 20;
    const int  kRows   = 15; // 300 bodies total
    const Real kRadius = Real(0.1);
    const Real kPitch  = Real(0.3); // > 2*radius: no initial overlap
    const Real kStartX = -(Real(kCols - 1) * kPitch) / Real(2);
    const Real kBottomY = kFloorTopY - Real(5); // lowest row starts 5 m above the floor
    const Real kStartY  = kBottomY - Real(kRows - 1) * kPitch;

    for (int row = 0; row < kRows; ++row)
    {
        for (int col = 0; col < kCols; ++col)
        {
            BodyDef bd;
            bd.type     = BodyType::Dynamic;
            bd.position = Vec2(kStartX + Real(col) * kPitch, kStartY + Real(row) * kPitch);
            bd.shape    = MakeCircle(kRadius);
            bd.density  = Real(1);
            w.AddBody(bd);
        }
    }

    // Step 300 times (5 s of sim) so the grid falls, collides, and piles/settles
    // on the floor inside the walled box.
    const auto t0 = std::chrono::steady_clock::now();
    for (int s = 0; s < 300; ++s)
    {
        w.Step(Real(1) / Real(60));
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    INFO("dense-step wall time (ms): " << ms);
    // MEASURED (Debug build, this machine, 4 runs): 1364.87 / 1373.95 / 1377.11 /
    // 1378.55 ms -- stable around ~1375 ms healthy. Ceiling set to ~10x that
    // (14000 ms) -- generous headroom so only an order-of-magnitude regression
    // (the px-era residency-tile pathology class, ~1087 s full-suite -> ~96 s)
    // trips this, never ordinary machine/load variance.
    CHECK(ms < 14000.0);
}
