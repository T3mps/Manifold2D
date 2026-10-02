// PhysicsJointChainTest.cpp
// [physics][joints][chain]: a chain of light revolute-jointed links holding a
// heavy load (Box2D v3's revolute point constraint: soft, solved against the
// in-flight separation each sub-step, warm started).
//
// The Lua-port point constraint computed its Baumgarte bias ONCE per step from
// the start-of-step separation and applied it again in every sub-step; on a long
// chain under a heavy load that over-correction stretched the joints and set the
// links zig-zagging (alternate links tilted +-0.7 rad, pins 0.15 m apart).
//
// Engine convention: +Y is DOWN, gravity +Y.
//
// PRESENTATION-FREE, ASCII comments, C++23.
#include <algorithm>
#include <cmath>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <Manifold2D/Physics/PhysicsTypes.hpp>
#include <Manifold2D/Physics/Shapes.hpp>
#include <Manifold2D/Physics/Body.hpp>
#include <Manifold2D/Physics/PhysicsWorld.hpp>
#include <Manifold2D/Physics/Joints/Joints.hpp>

using namespace Manifold2D::Physics;

namespace
{
    constexpr Real kStep = Real(1) / Real(60);
    constexpr int kLinks = 6;
    constexpr Real kLink = Real(0.3);

    struct Chain
    {
        BodyHandle top;
        std::vector<BodyHandle> links;
        BodyHandle load;
    };

    // a chain hanging from `top` (static or kinematic) at the origin, a load box at its end
    Chain Hang(PhysicsWorld& w, BodyType topType, Real linkDensity, Real loadDensity)
    {
        Chain c;
        BodyDef t;
        t.type = topType;
        t.shape = MakeCircle(Real(0.05));
        t.maskBits = 0u;
        c.top = w.AddBody(t);
        BodyHandle prev = c.top;
        for (int i = 0; i < kLinks; ++i)
        {
            BodyDef d;
            d.type = BodyType::Dynamic;
            d.position = Vec2(Real(0), kLink * (Real(i) + Real(0.5)));
            d.shape = MakeCapsule(kLink / Real(2) - Real(0.02), Real(0.022));
            d.density = linkDensity;
            d.maskBits = 0u;
            const BodyHandle l = w.AddBody(d);
            w.SetAngle(l, Real(1.5707963)); // local +x down the chain
            JointDef jd;
            jd.kind = JointKind::Revolute;
            jd.a = prev;
            jd.b = l;
            jd.anchor = Vec2(Real(0), kLink * Real(i));
            REQUIRE(w.AddJoint(jd) != nullptr);
            c.links.push_back(l);
            prev = l;
        }
        BodyDef p;
        p.type = BodyType::Dynamic;
        p.position = Vec2(Real(0), kLink * kLinks + Real(0.05));
        p.shape = MakePolygon(std::vector<Vec2>{ Vec2(-0.75f, -0.05f), Vec2(0.75f, -0.05f), Vec2(0.75f, 0.05f), Vec2(-0.75f, 0.05f) });
        p.density = loadDensity;
        p.maskBits = 0u;
        c.load = w.AddBody(p);
        JointDef jd;
        jd.kind = JointKind::Revolute;
        jd.a = prev;
        jd.b = c.load;
        jd.anchor = Vec2(Real(0), kLink * kLinks);
        REQUIRE(w.AddJoint(jd) != nullptr);
        return c;
    }

    // the largest gap between consecutive links' joined ends, and the largest link tilt off vertical
    void Measure(const PhysicsWorld& w, const Chain& c, Real& gap, Real& tilt)
    {
        gap = Real(0);
        tilt = Real(0);
        const Real h = kLink / Real(2);
        for (std::size_t i = 0; i < c.links.size(); ++i)
        {
            const Real a = w.GetAngle(c.links[i]);
            tilt = std::max(tilt, std::abs(a - Real(1.5707963)));
            if (i == 0) { continue; }
            const Vec2 p = w.Position(c.links[i - 1]), q = w.Position(c.links[i]);
            const Real ap = w.GetAngle(c.links[i - 1]);
            const Vec2 endP(p.x + std::cos(ap) * h, p.y + std::sin(ap) * h);
            const Vec2 endQ(q.x - std::cos(a) * h, q.y - std::sin(a) * h);
            gap = std::max(gap, std::hypot(endP.x - endQ.x, endP.y - endQ.y));
        }
    }
} // namespace

// zig-zag: the largest tilt two neighbouring links share in opposite directions
static Real ZigZag(const PhysicsWorld& w, const Chain& c)
{
    Real z = Real(0);
    for (std::size_t k = 1; k < c.links.size(); ++k)
    {
        const Real a = w.GetAngle(c.links[k - 1]) - Real(1.5707963), b = w.GetAngle(c.links[k]) - Real(1.5707963);
        if (a * b < Real(0)) { z = std::max(z, std::min(std::abs(a), std::abs(b))); }
    }
    return z;
}

TEST_CASE("Chain: links holding a load six times their weight hang taut and straight",
          "[physics][joints][chain]")
{
    // Box2D's guidance: keep jointed mass ratios near 10:1 or below. Links ~0.08 kg,
    // load ~0.47 kg. (Box2D's joints are soft -- 60 Hz, damping ratio 2 -- so a hanging
    // chain sags a little under its load: here ~4 cm over 1.8 m.)
    PhysicsWorld w{ WorldDef{} };
    const Chain c = Hang(w, BodyType::Static, Real(6), Real(3));
    Real worstGap = Real(0);
    for (int i = 0; i < 180; ++i)
    {
        w.Step(kStep);
        Real gap = Real(0), tilt = Real(0);
        Measure(w, c, gap, tilt);
        worstGap = std::max(worstGap, gap);
    }
    Real gap = Real(0), tilt = Real(0);
    Measure(w, c, gap, tilt);
    const Real drop = w.Position(c.load).y - (kLink * kLinks + Real(0.05));
    INFO("gap " << gap << " worst " << worstGap << " tilt " << tilt << " drop " << drop);
    CHECK(worstGap < Real(0.015)); // (its worst is the first loading, ~1 cm)
    CHECK(tilt < Real(0.02));
    CHECK(drop < Real(0.05));
}

TEST_CASE("Chain: swung by a kinematic trolley that moves and stops, the links never zig-zag",
          "[physics][joints][chain]")
{
    // The Lua-port point constraint (one Baumgarte bias per step, applied in every
    // sub-step) zig-zagged here: neighbouring links 0.4 rad apart in opposite
    // directions, pins 2.4 cm open; at a 60:1 mass ratio 1.1 rad and 0.44 m.
    for (const Real loadDensity : { Real(3), Real(8) })
    {
        PhysicsWorld w{ WorldDef{} };
        const Chain c = Hang(w, BodyType::Kinematic, Real(6), loadDensity);
        Real worstGap = Real(0), worstZig = Real(0);
        for (int i = 0; i < 480; ++i)
        {
            // eased: 0 -> 0.4 m/s -> 0 over two seconds, then still
            const Real s = Real(i) / Real(60);
            const Real v = s < Real(2) ? Real(0.4) * std::sin(Real(3.14159265) * s / Real(2)) : Real(0);
            w.SetVelocity(c.top, Vec2(v, Real(0)));
            w.Step(kStep);
            Real gap = Real(0), tilt = Real(0);
            Measure(w, c, gap, tilt);
            worstGap = std::max(worstGap, gap);
            worstZig = std::max(worstZig, ZigZag(w, c));
        }
        INFO("load density " << loadDensity << " worst gap " << worstGap << " zig-zag " << worstZig);
        CHECK(worstZig < Real(0.02));
        CHECK(worstGap < Real(0.03));
        for (const BodyHandle l : c.links) { CHECK(std::isfinite(w.Position(l).x)); }
    }
}
