// PhysicsDistanceJointTest.cpp
// [physics][joints][distance]: the distance joint's Box2D v3 features
// (b2DistanceJoint: distance_joint.c b2SolveDistanceJoint, b2DistanceJointDef
// localAnchorA/B, length, enableSpring/hertz/dampingRatio, enableLimit/
// minLength/maxLength).
//
//   * anchors: in each body's frame; the constraint acts between the anchor
//     points, so it torques both bodies.
//   * rigid (the default): the length is a hard constraint (soft at the joint
//     constraint softness on the biased pass, warm started).
//   * spring: the length becomes a spring of (hertz, dampingRatio); hertz 0 is
//     no length constraint at all.
//   * limit (with the spring on): minLength <= length <= maxLength, two
//     one-sided speculative constraints.
//   * rope = spring at hertz 0 + limit [0, L]: holds only when taut.
//   * defaults (centre anchors, no spring, no limit) run the original Lua rod,
//     unchanged.
//
// Engine convention: +Y is DOWN, gravity +Y (10 m/s^2 by default).
//
// PRESENTATION-FREE, ASCII comments, C++23.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <Manifold2D/Physics/PhysicsTypes.hpp>
#include <Manifold2D/Physics/Shapes.hpp>
#include <Manifold2D/Physics/Body.hpp>
#include <Manifold2D/Physics/PhysicsWorld.hpp>
#include <Manifold2D/Physics/Joints/Joints.hpp>

using namespace Manifold2D::Physics;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace
{
    constexpr Real kStep = Real(1) / Real(60);
    constexpr double kG = 10.0;

    void Run(PhysicsWorld& w, int n)
    {
        for (int i = 0; i < n; ++i) { w.Step(kStep); }
    }

    WorldDef World(bool gravity)
    {
        WorldDef d;
        if (!gravity) { d.gravityX = Real(0); d.gravityY = Real(0); }
        return d;
    }

    // A static pin that touches nothing.
    BodyHandle Pin(PhysicsWorld& w, Vec2 p)
    {
        BodyDef s; s.type = BodyType::Static; s.shape = MakeCircle(Real(0.05)); s.position = p; s.maskBits = 0u;
        return w.AddBody(s);
    }

    // A box polygon (a dynamic AABB shape must be fixedRotation; these turn).
    Shape BoxShape(Real hx, Real hy)
    {
        return MakePolygon({ Vec2(-hx, -hy), Vec2(hx, -hy), Vec2(hx, hy), Vec2(-hx, hy) });
    }

    // A dynamic box (1 kg/m^2) that touches nothing and never sleeps: only the
    // joints act on it, and it is solved every step.
    BodyHandle Box(PhysicsWorld& w, Vec2 p, Real hx, Real hy, Real damping = Real(0))
    {
        BodyDef d; d.type = BodyType::Dynamic; d.shape = BoxShape(hx, hy); d.density = Real(1);
        d.position = p; d.maskBits = 0u; d.sleepThreshold = Real(0); d.linearDamping = damping;
        return w.AddBody(d);
    }
    BodyHandle Ball(PhysicsWorld& w, Vec2 p, Real r = Real(0.1))
    {
        BodyDef d; d.type = BodyType::Dynamic; d.shape = MakeCircle(r); d.density = Real(1);
        d.position = p; d.maskBits = 0u; d.sleepThreshold = Real(0);
        return w.AddBody(d);
    }

    Vec2 Rot(Real a, Vec2 v)
    {
        const Real c = std::cos(a), s = std::sin(a);
        return Vec2(c * v.x - s * v.y, s * v.x + c * v.y);
    }
    // The world point of a body-local anchor.
    Vec2 WorldPoint(const PhysicsWorld& w, BodyHandle h, Vec2 local)
    {
        const Vec2 p = w.Position(h);
        const Vec2 r = Rot(w.GetAngle(h), local);
        return Vec2(p.x + r.x, p.y + r.y);
    }
    double Dist(Vec2 a, Vec2 b) { return std::hypot(double(b.x) - double(a.x), double(b.y) - double(a.y)); }
    double Wrap(double a)
    {
        const double twoPi = 2.0 * 3.14159265358979323846;
        a = std::fmod(a, twoPi);
        if (a > 3.14159265358979323846) { a -= twoPi; }
        if (a < -3.14159265358979323846) { a += twoPi; }
        return a;
    }

    DistanceJoint* AddDistance(PhysicsWorld& w, JointDef j, BodyHandle a, BodyHandle b)
    {
        j.kind = JointKind::Distance; j.a = a; j.b = b;
        auto* dj = dynamic_cast<DistanceJoint*>(w.AddJoint(j));
        REQUIRE(dj != nullptr);
        return dj;
    }
    // A rope: no length constraint (spring at 0 Hz) inside the limit [0, maxLength].
    JointDef Rope(Real maxLength, Vec2 la = Vec2(Real(0), Real(0)), Vec2 lb = Vec2(Real(0), Real(0)))
    {
        JointDef j;
        j.localAnchorA = la; j.localAnchorB = lb;
        j.enableSpring = true; j.frequencyHz = Real(0); j.dampingRatio = Real(0);
        j.enableLimit = true; j.minLength = Real(0); j.maxLength = maxLength;
        j.length = maxLength;
        return j;
    }
} // namespace

TEST_CASE("Distance joint: a box hung by one off-centre anchor settles with its centre under the anchor, under the pivot",
          "[physics][joints][distance][anchor]")
{
    PhysicsWorld w{ World(true) };
    const BodyHandle pivot = Pin(w, Vec2(Real(0), Real(0)));
    // 1 x 0.5 m box; the anchor is its top-right corner (local (0.5, -0.25); +y is down).
    const BodyHandle box = Box(w, Vec2(Real(0), Real(2)), Real(0.5), Real(0.25), Real(0.5));
    JointDef j; j.localAnchorB = Vec2(Real(0.5), Real(-0.25)); // length <= 0: the current anchor distance
    DistanceJoint* dj = AddDistance(w, j, pivot, box);
    const double L = std::hypot(0.5, 1.75);
    CHECK_THAT(double(dj->GetLength()), WithinAbs(L, 1e-5));
    CHECK_THAT(double(dj->GetCurrentLength(w)), WithinAbs(L, 1e-5));

    Run(w, 1200);
    const Vec2 anchor = WorldPoint(w, box, dj->LocalAnchorB());
    const Vec2 c = w.Position(box);
    const double angle = Wrap(double(w.GetAngle(box)));
    INFO("anchor " << anchor.x << ", " << anchor.y << "  centre " << c.x << ", " << c.y << "  angle " << angle);
    // Hanging straight: the pivot, the anchor and the centre on one vertical line.
    CHECK_THAT(double(anchor.x), WithinAbs(0.0, 0.01));
    CHECK_THAT(double(c.x), WithinAbs(0.0, 0.01));
    CHECK(c.y > anchor.y);
    // The joint's torque turned the box until the corner-to-centre line hangs
    // vertical: R(angle) (-0.5, 0.25) points straight down at angle = atan(-2).
    CHECK_THAT(angle, WithinAbs(std::atan(-2.0), 0.02));
    CHECK_THAT(double(dj->GetCurrentLength(w)), WithinAbs(L, 0.005));

    // At rest the joint carries the box's weight (up, on the box) and no torque.
    const double mg = double(w.GetBodyMass(box)) * kG;
    const Vec2 f = w.JointReactionForce(dj);
    INFO("f " << f.x << ", " << f.y << " mg " << mg);
    CHECK_THAT(double(f.y), WithinRel(-mg, 0.02));
    CHECK_THAT(double(f.x), WithinAbs(0.0, 0.02 * mg));
    CHECK(w.JointReactionTorque(dj) == Real(0));
}

TEST_CASE("Distance joint: a two-rope bridle from one hook to two corners locks the box's attitude to the cable and kills a spin",
          "[physics][joints][distance][rope][bridle]")
{
    // The bridle: two ropes from the hook to the box's top corners. Taut, the
    // hook and the two corners form a rigid triangle, so the box can only turn
    // by swinging the whole sling about the hook. A box on a single rope at its
    // centre, spun the same way, is the control: nothing there can take its
    // spin out.
    const Vec2 left(Real(-0.5), Real(-0.25)), right(Real(0.5), Real(-0.25));
    const Real L = Real(std::hypot(0.5, 1.75));

    SECTION("undamped: the spin is gone within a second; what is left is the sling swinging, the box turning with it")
    {
        PhysicsWorld w{ World(true) };
        const BodyHandle hook = Pin(w, Vec2(Real(0), Real(0)));
        const BodyHandle box = Box(w, Vec2(Real(0), Real(2)), Real(0.5), Real(0.25));
        DistanceJoint* ropeL = AddDistance(w, Rope(L, Vec2(Real(0), Real(0)), left), hook, box);
        DistanceJoint* ropeR = AddDistance(w, Rope(L, Vec2(Real(0), Real(0)), right), hook, box);
        const BodyHandle hook2 = Pin(w, Vec2(Real(5), Real(0)));
        const BodyHandle control = Box(w, Vec2(Real(5), Real(2)), Real(0.5), Real(0.25));
        AddDistance(w, Rope(Real(2)), hook2, control);
        w.SetAngularVelocity(box, Real(3));
        w.SetAngularVelocity(control, Real(3));

        Run(w, 60);
        double maxSpin = 0.0, maxOffCable = 0.0, slackest = 9.0;
        for (int i = 0; i < 540; ++i)
        {
            w.Step(kStep);
            const Vec2 c = w.Position(box);
            // the box's attitude is the cable's: rotated by the swing angle of
            // the hook -> centre line from the vertical
            const double cable = -std::atan2(double(c.x), double(c.y));
            maxSpin = std::max(maxSpin, std::fabs(double(w.AngularVelocity(box))));
            maxOffCable = std::max(maxOffCable, std::fabs(Wrap(double(w.GetAngle(box))) - cable));
            slackest = std::min({ slackest, double(ropeL->GetCurrentLength(w)), double(ropeR->GetCurrentLength(w)) });
        }
        INFO("max spin " << maxSpin << " max off-cable " << maxOffCable << " slackest leg " << slackest
             << "  control spin " << w.AngularVelocity(control));
        CHECK(maxSpin < 0.1);                               // from 3 rad/s
        CHECK(maxOffCable < 0.005);                         // attitude locked to the cable
        CHECK_THAT(slackest, WithinAbs(double(L), 0.005));  // both legs stay taut
        CHECK(w.AngularVelocity(control) == Real(3));       // the control spins on untouched
    }

    SECTION("damped: it comes to rest level under the hook, each leg carrying half the weight")
    {
        PhysicsWorld w{ World(true) };
        const BodyHandle hook = Pin(w, Vec2(Real(0), Real(0)));
        const BodyHandle box = Box(w, Vec2(Real(0), Real(2)), Real(0.5), Real(0.25), Real(0.5));
        DistanceJoint* ropeL = AddDistance(w, Rope(L, Vec2(Real(0), Real(0)), left), hook, box);
        DistanceJoint* ropeR = AddDistance(w, Rope(L, Vec2(Real(0), Real(0)), right), hook, box);
        w.SetAngularVelocity(box, Real(3));
        Run(w, 900);
        const double angle = Wrap(double(w.GetAngle(box)));
        INFO("angle " << angle << " spin " << w.AngularVelocity(box) << " x " << w.Position(box).x);
        CHECK_THAT(angle, WithinAbs(0.0, 0.005));                     // level
        CHECK_THAT(double(w.AngularVelocity(box)), WithinAbs(0.0, 0.005));
        CHECK_THAT(double(w.Position(box).x), WithinAbs(0.0, 0.005)); // under the hook
        CHECK_THAT(double(ropeL->GetCurrentLength(w)), WithinAbs(double(L), 0.005));
        CHECK_THAT(double(ropeR->GetCurrentLength(w)), WithinAbs(double(L), 0.005));

        const double mg = double(w.GetBodyMass(box)) * kG;
        const Vec2 fl = w.JointReactionForce(ropeL), fr = w.JointReactionForce(ropeR);
        INFO("fl " << fl.x << ", " << fl.y << "  fr " << fr.x << ", " << fr.y << "  mg " << mg);
        CHECK_THAT(double(fl.y + fr.y), WithinRel(-mg, 0.02));
        CHECK_THAT(double(fl.x + fr.x), WithinAbs(0.0, 0.02 * mg));
        CHECK_THAT(double(fl.y), WithinRel(double(fr.y), 0.05));
        // each along its own leg: the legs lean in at 0.5 / 1.75
        CHECK_THAT(double(fl.x / fl.y), WithinAbs(-0.5 / 1.75, 0.01));
    }
}

TEST_CASE("Distance joint: a rope goes slack when the bodies approach and holds at its length when pulled",
          "[physics][joints][distance][rope]")
{
    SECTION("approaching: no push (a rod, the default, pushes back)")
    {
        PhysicsWorld w{ World(false) };
        const BodyHandle pinR = Pin(w, Vec2(Real(0), Real(0)));
        const BodyHandle ballR = Ball(w, Vec2(Real(1), Real(0)));
        DistanceJoint* rope = AddDistance(w, Rope(Real(1)), pinR, ballR);
        const BodyHandle pinD = Pin(w, Vec2(Real(0), Real(3)));
        const BodyHandle ballD = Ball(w, Vec2(Real(1), Real(3)));
        JointDef rod; // the default: the legacy rigid rod
        AddDistance(w, rod, pinD, ballD);
        w.SetVelocity(ballR, Vec2(Real(-2), Real(0)));
        w.SetVelocity(ballD, Vec2(Real(-2), Real(0)));
        Run(w, 15); // 0.25 s
        INFO("rope ball x " << w.Position(ballR).x << " vx " << w.Velocity(ballR).x
             << "  rod ball x " << w.Position(ballD).x << " vx " << w.Velocity(ballD).x);
        CHECK_THAT(double(w.Position(ballR).x), WithinAbs(0.5, 1e-4));  // flew on, unhindered
        CHECK_THAT(double(w.Velocity(ballR).x), WithinAbs(-2.0, 1e-5));
        CHECK_THAT(double(rope->GetCurrentLength(w)), WithinAbs(0.5, 1e-4));
        CHECK(Dist(Vec2(Real(0), Real(0)), w.JointReactionForce(rope)) < 1e-6); // carried nothing
        CHECK_THAT(double(w.Position(ballD).x), WithinAbs(1.0, 0.01));  // the rod held it off
        CHECK(std::fabs(double(w.Velocity(ballD).x)) < 0.05);
    }
    SECTION("pulled: a ball dropped on a slack rope is caught at the rope's length and hangs there carrying m g")
    {
        PhysicsWorld w{ World(true) };
        const BodyHandle pin = Pin(w, Vec2(Real(0), Real(0)));
        const BodyHandle ball = Ball(w, Vec2(Real(0), Real(0.5)));
        DistanceJoint* rope = AddDistance(w, Rope(Real(1)), pin, ball);
        double most = 0.0;
        for (int i = 0; i < 180; ++i)
        {
            w.Step(kStep);
            most = std::max(most, double(rope->GetCurrentLength(w)));
        }
        const double mg = double(w.GetBodyMass(ball)) * kG;
        const Vec2 f = w.JointReactionForce(rope);
        INFO("length " << rope->GetCurrentLength(w) << " most " << most << " vy " << w.Velocity(ball).y << " f " << f.y << " mg " << mg);
        CHECK_THAT(double(rope->GetCurrentLength(w)), WithinAbs(1.0, 0.003)); // mm
        CHECK(most < 1.01);                                                     // caught, not overstretched
        CHECK_THAT(double(w.Velocity(ball).y), WithinAbs(0.0, 1e-3));
        CHECK_THAT(double(f.y), WithinRel(-mg, 0.02));
        CHECK_THAT(double(f.x), WithinAbs(0.0, 0.01 * mg));
    }
}

TEST_CASE("Distance joint: the min/max length limits hold",
          "[physics][joints][distance][limit]")
{
    const auto limited = [](Real minLength, Real maxLength)
    {
        JointDef j;
        j.length = Real(1);
        j.enableSpring = true; j.frequencyHz = Real(0); // free inside the range
        j.enableLimit = true; j.minLength = minLength; j.maxLength = maxLength;
        return j;
    };
    SECTION("flung outward it stops at maxLength; flung inward it stops at minLength")
    {
        for (const Real v : { Real(3), Real(-3) })
        {
            PhysicsWorld w{ World(false) };
            const BodyHandle pin = Pin(w, Vec2(Real(0), Real(0)));
            const BodyHandle ball = Ball(w, Vec2(Real(1), Real(0)));
            DistanceJoint* dj = AddDistance(w, limited(Real(0.8), Real(1.2)), pin, ball);
            w.SetVelocity(ball, Vec2(v, Real(0)));
            double lo = 9.0, hi = 0.0;
            for (int i = 0; i < 120; ++i)
            {
                w.Step(kStep);
                const double len = double(dj->GetCurrentLength(w));
                lo = std::min(lo, len); hi = std::max(hi, len);
            }
            const double len = double(dj->GetCurrentLength(w));
            INFO("v " << v << " len " << len << " lo " << lo << " hi " << hi << " vx " << w.Velocity(ball).x);
            CHECK(hi < 1.2 + 0.005);
            CHECK(lo > 0.8 - 0.005);
            CHECK_THAT(len, WithinAbs(v > Real(0) ? 1.2 : 0.8, 0.005)); // stopped on the limit it hit
            CHECK_THAT(double(w.Velocity(ball).x), WithinAbs(0.0, 1e-3));
        }
    }
    SECTION("a spring stretched past maxLength by its load rests on the limit")
    {
        // 1 Hz spring, rest 1 m: alone it would hang at 1 + g / omega^2 = 1.253 m.
        PhysicsWorld w{ World(true) };
        const BodyHandle pin = Pin(w, Vec2(Real(0), Real(0)));
        const BodyHandle ball = Ball(w, Vec2(Real(0), Real(1)));
        JointDef j = limited(Real(0.5), Real(1.1));
        j.frequencyHz = Real(1); j.dampingRatio = Real(1);
        DistanceJoint* dj = AddDistance(w, j, pin, ball);
        Run(w, 300);
        CHECK_THAT(double(dj->GetCurrentLength(w)), WithinAbs(1.1, 0.003));
        const double mg = double(w.GetBodyMass(ball)) * kG;
        CHECK_THAT(double(w.JointReactionForce(dj).y), WithinRel(-mg, 0.02)); // spring + limit together
    }
    SECTION("with the spring off the joint is rigid and the limit is not solved (Box2D's rule)")
    {
        PhysicsWorld w{ World(true) };
        const BodyHandle pin = Pin(w, Vec2(Real(0), Real(0)));
        const BodyHandle ball = Ball(w, Vec2(Real(0), Real(1)));
        JointDef j = limited(Real(0.5), Real(2));
        j.enableSpring = false;
        DistanceJoint* dj = AddDistance(w, j, pin, ball);
        Run(w, 120);
        CHECK_THAT(double(dj->GetCurrentLength(w)), WithinAbs(1.0, 0.003)); // held at length, not let fall to 2
    }
    SECTION("SetLengthRange sorts and clamps to [linear slop, huge]; SetLength clamps")
    {
        PhysicsWorld w{ World(false) };
        const BodyHandle pin = Pin(w, Vec2(Real(0), Real(0)));
        const BodyHandle ball = Ball(w, Vec2(Real(1), Real(0)));
        DistanceJoint* dj = AddDistance(w, limited(Real(0), Real(1)), pin, ball);
        CHECK(dj->GetMinLength() == kLinearSlop); // a rope's 0 becomes the slop, as Box2D's
        CHECK(dj->GetMaxLength() == Real(1));
        dj->SetLengthRange(Real(3), Real(-1));
        CHECK(dj->GetMinLength() == kLinearSlop);
        CHECK(dj->GetMaxLength() == Real(3));
        dj->SetLengthRange(Real(2), Real(1e9));
        CHECK(dj->GetMinLength() == Real(2));
        CHECK(dj->GetMaxLength() == kDistanceJointHuge);
        dj->SetLength(Real(0));
        CHECK(dj->GetLength() == kLinearSlop);
    }
}

TEST_CASE("Distance joint: the spring oscillates at its hertz",
          "[physics][joints][distance][spring]")
{
    for (const Real hz : { Real(1), Real(2), Real(4) })
    {
        PhysicsWorld w{ World(false) };
        const BodyHandle pin = Pin(w, Vec2(Real(0), Real(0)));
        const BodyHandle ball = Ball(w, Vec2(Real(1.2), Real(0)));
        JointDef j; j.length = Real(1); j.enableSpring = true; j.frequencyHz = hz; j.dampingRatio = Real(0);
        DistanceJoint* dj = AddDistance(w, j, pin, ball);
        // Time the upward zero crossings of (length - rest) over 4 s.
        std::vector<double> crossings;
        double prev = double(dj->GetCurrentLength(w)) - 1.0;
        for (int i = 1; i <= 240; ++i)
        {
            w.Step(kStep);
            const double x = double(dj->GetCurrentLength(w)) - 1.0;
            if (prev < 0.0 && x >= 0.0) { crossings.push_back((double(i - 1) + prev / (prev - x)) * double(kStep)); }
            prev = x;
        }
        REQUIRE(crossings.size() >= 3);
        const double period = (crossings.back() - crossings.front()) / double(crossings.size() - 1);
        INFO("hz " << hz << " crossings " << crossings.size() << " measured " << 1.0 / period);
        CHECK_THAT(1.0 / period, WithinRel(double(hz), 0.03));
    }
}

TEST_CASE("Distance joint: defaults are the original rigid centre-to-centre rod",
          "[physics][joints][distance][defaults]")
{
    SECTION("the JointDef defaults")
    {
        const JointDef d;
        CHECK(d.localAnchorA.x == Real(0)); CHECK(d.localAnchorA.y == Real(0));
        CHECK(d.localAnchorB.x == Real(0)); CHECK(d.localAnchorB.y == Real(0));
        CHECK_FALSE(d.enableSpring);
        CHECK_FALSE(d.enableLimit);
        CHECK(d.minLength == Real(0));
        CHECK(d.maxLength == kDistanceJointHuge);

        PhysicsWorld w{ World(true) };
        const BodyHandle pin = Pin(w, Vec2(Real(0), Real(0)));
        const BodyHandle ball = Ball(w, Vec2(Real(0.6), Real(0.8)));
        DistanceJoint* dj = AddDistance(w, JointDef{}, pin, ball);
        CHECK_FALSE(dj->IsSpringEnabled());
        CHECK_FALSE(dj->IsLimitEnabled());
        CHECK_THAT(double(dj->GetLength()), WithinAbs(1.0, 1e-6)); // the current centre distance
        CHECK_THAT(double(dj->GetCurrentLength(w)), WithinAbs(1.0, 1e-6));
    }
    SECTION("a joint switched to a rope and back runs bit-identically to one never touched")
    {
        // Two copies of a scene of rods; in one, every joint has its spring and
        // limit enabled for 60 steps and then switched off again. Once back on
        // the default path, it must be the original rod: the copies only
        // differ by what happened during those 60 steps, so re-sync them and
        // then compare bit for bit.
        struct Scene
        {
            PhysicsWorld w{ World(true) };
            BodyHandle pin, a, b;
            DistanceJoint* j1 = nullptr;
            DistanceJoint* j2 = nullptr;
            Scene()
            {
                pin = Pin(w, Vec2(Real(0), Real(0)));
                a = Box(w, Vec2(Real(0.8), Real(0.3)), Real(0.2), Real(0.1));
                b = Box(w, Vec2(Real(1.6), Real(0.9)), Real(0.15), Real(0.15));
                j1 = AddDistance(w, JointDef{}, pin, a);
                JointDef d; d.length = Real(0.9);
                j2 = AddDistance(w, d, a, b);
            }
        };
        Scene x, y;
        y.j1->EnableSpring(true); y.j1->EnableLimit(true); y.j1->SetLengthRange(Real(0), Real(2));
        y.j2->EnableSpring(true); y.j2->SetSpring(Real(3), Real(0.2));
        Run(y.w, 60);
        y.j1->EnableSpring(false); y.j1->EnableLimit(false);
        y.j2->EnableSpring(false);
        // re-sync y's bodies to x's state
        for (const auto& [from, to] : { std::pair{ x.a, y.a }, std::pair{ x.b, y.b } })
        {
            y.w.SetPosition(to, x.w.Position(from));
            y.w.SetAngle(to, x.w.GetAngle(from));
            y.w.SetVelocity(to, x.w.Velocity(from));
            y.w.SetAngularVelocity(to, x.w.AngularVelocity(from));
        }
        for (int i = 0; i < 300; ++i)
        {
            x.w.Step(kStep);
            y.w.Step(kStep);
            for (const auto& [p, q] : { std::pair{ x.a, y.a }, std::pair{ x.b, y.b } })
            {
                const Vec2 pp = x.w.Position(p), qq = y.w.Position(q);
                REQUIRE(std::memcmp(&pp, &qq, sizeof(Vec2)) == 0);
                const Vec2 pv = x.w.Velocity(p), qv = y.w.Velocity(q);
                REQUIRE(std::memcmp(&pv, &qv, sizeof(Vec2)) == 0);
            }
        }
    }
    SECTION("explicit centre anchors are the default")
    {
        const auto run = [](bool explicitAnchors)
        {
            PhysicsWorld w{ World(true) };
            const BodyHandle pin = Pin(w, Vec2(Real(0), Real(0)));
            const BodyHandle box = Box(w, Vec2(Real(0.7), Real(0.4)), Real(0.3), Real(0.1));
            JointDef d;
            if (explicitAnchors) { d.localAnchorA = Vec2(Real(0), Real(0)); d.localAnchorB = Vec2(Real(0), Real(0)); d.minLength = Real(0.2); }
            AddDistance(w, d, pin, box);
            w.SetAngularVelocity(box, Real(2));
            Run(w, 240);
            return std::pair{ w.Position(box), w.AngularVelocity(box) };
        };
        const auto [p0, w0] = run(false);
        const auto [p1, w1] = run(true);
        CHECK(std::memcmp(&p0, &p1, sizeof(Vec2)) == 0);
        CHECK(w0 == w1);
        // ...and a centre joint cannot turn the box (no torque): the spin it
        // was given is untouched.
        CHECK(w0 == Real(2));
    }
}

TEST_CASE("Distance joint: a scene of anchored ropes and springs is deterministic",
          "[physics][joints][distance][determinism]")
{
    const auto run = []()
    {
        PhysicsWorld w{ World(true) };
        BodyDef g; g.type = BodyType::Static; g.shape = MakeAabb(Real(5), Real(0.2)); g.position = Vec2(Real(0), Real(4));
        w.AddBody(g);
        const BodyHandle hook = Pin(w, Vec2(Real(0), Real(0)));
        BodyDef d; d.type = BodyType::Dynamic; d.shape = BoxShape(Real(0.4), Real(0.2)); d.position = Vec2(Real(0.5), Real(1.5));
        const BodyHandle box = w.AddBody(d);
        d.shape = MakeCircle(Real(0.2)); d.position = Vec2(Real(-0.5), Real(2.5));
        const BodyHandle ball = w.AddBody(d);
        AddDistance(w, Rope(Real(1.7), Vec2(Real(0), Real(0)), Vec2(Real(-0.4), Real(-0.2))), hook, box);
        AddDistance(w, Rope(Real(1.7), Vec2(Real(0), Real(0)), Vec2(Real(0.4), Real(-0.2))), hook, box);
        JointDef s; s.localAnchorA = Vec2(Real(0.4), Real(0.2)); s.enableSpring = true; s.frequencyHz = Real(3); s.dampingRatio = Real(0.1);
        s.enableLimit = true; s.minLength = Real(0.3); s.maxLength = Real(1.5);
        AddDistance(w, s, box, ball);
        w.SetAngularVelocity(box, Real(4));
        std::vector<Real> out;
        for (int i = 0; i < 300; ++i)
        {
            w.Step(kStep);
            for (const BodyHandle h : { box, ball })
            {
                out.push_back(w.Position(h).x); out.push_back(w.Position(h).y); out.push_back(w.GetAngle(h));
            }
        }
        return out;
    };
    const std::vector<Real> a = run(), b = run();
    REQUIRE(a.size() == b.size());
    CHECK(std::memcmp(a.data(), b.data(), a.size() * sizeof(Real)) == 0);
}
