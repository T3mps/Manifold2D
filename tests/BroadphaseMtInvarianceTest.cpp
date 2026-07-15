#include <catch2/catch_test_macros.hpp>
#include <Manifold2D/Physics/PhysicsWorld.hpp>
#include "Support/TestWorkScheduler.hpp"
#include <thread>

using namespace Manifold2D::Physics;

namespace
{
    // ~800 dynamic boxes over a floor, 50 steps while settling -> many moving
    // proxies/step (UpdatePairs work-list > grain -> real broadphase MT).
    std::vector<float> RunActivePile(Mosaic::IWorkScheduler* exec, int steps)
    {
        WorldDef wd; // gravityY inherits the MKS default (+10)
        PhysicsWorld w(wd); w.SetExecutor(exec);
        { BodyDef fd; fd.type=BodyType::Static; fd.position=Vec2(Real(0),Real(5));
          fd.shape=MakeAabb(Real(40),Real(0.5)); w.AddBody(fd); }
        std::vector<BodyHandle> bodies; bodies.reserve(800);
        for (int i=0;i<800;++i){ int c=i%20, r=i/20;
          BodyDef bd; bd.type=BodyType::Dynamic;
          bd.position=Vec2(static_cast<Real>(c-10)*Real(1.0), Real(-1)-static_cast<Real>(r)*Real(1.2));
          bd.shape=MakeAabb(Real(0.45),Real(0.45)); bd.density=Real(1); bd.friction=Real(0.3); bd.fixedRotation=true;
          bodies.push_back(w.AddBody(bd)); }
        for (int s=0;s<steps;++s) w.Step(Real(1)/Real(60));
        std::vector<float> out; out.reserve(bodies.size()*5u);
        for (auto h:bodies){ const Vec2 p=w.Position(h); const Vec2 v=w.Velocity(h);
          out.push_back((float)p.x); out.push_back((float)p.y); out.push_back((float)w.GetAngle(h));
          out.push_back((float)v.x); out.push_back((float)v.y);} return out;
    }
} // namespace

TEST_CASE("broadphase thread-count invariance: serial == enki(1) == enki(N)", "[physics][determinism][broadmt]")
{
    Mosaic::SerialWorkScheduler serial;
    // Drive Manifold2D through a std::thread pool via the IWorkScheduler seam (test-only).
    const std::uint32_t hw = std::thread::hardware_concurrency();
    Manifold2D::Testing::TestWorkScheduler oneSched(1);
    Manifold2D::Testing::TestWorkScheduler manySched(hw > 1u ? hw : 2u);
    const auto a=RunActivePile(&serial,50);
    const auto b=RunActivePile(&oneSched,50);
    const auto c=RunActivePile(&manySched,50);
    INFO("workers=" << manySched.WorkerCount());
    if (manySched.WorkerCount() <= 1u) WARN("single worker: broadphase MT path not exercised this run");
    REQUIRE(a.size()==b.size()); REQUIRE(a==b); REQUIRE(a==c);
}
