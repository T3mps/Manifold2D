// PhysicsContactEventsTest.cpp
// [physics][events]: contact begin/end arrays (spec 2026-10-08 s6.1),
// including destroy-time Ends (RemoveBody, DropFixture, filter, fat box).
#include <bit>
#include <cstdint>
#include <thread>

#include <catch2/catch_test_macros.hpp>
#include "PhysicsEventTestHelpers.hpp"
#include "Support/TestWorkScheduler.hpp"

using namespace EventTest;

TEST_CASE("Event flags default off and are captured when a contact is created", "[physics][events]")
{
    PhysicsWorld w{ WorldDef{} };
    FixtureDef fd;                       // defaults
    CHECK_FALSE(fd.contactEvents);
    CHECK_FALSE(fd.sensorEvents);
    CHECK_FALSE(fd.hitEvents);
    CHECK(WorldDef{}.hitEventThreshold == Real(1));
    CHECK_FALSE(WorldDef{}.contactEventsRequireBoth);

    const BodyHandle g = AddGround(w, /*contactEvents*/ false);
    const BodyHandle b = AddBox(w, Real(0), Real(-0.49), /*contactEvents*/ true, /*hit*/ true);
    w.Step(kStep);
    const std::uint8_t flags = w.DebugContactEventFlags(w.GetBodyFixture(b, 0), w.GetBodyFixture(g, 0));
    REQUIRE(flags != 0xFF);              // the pair exists
    CHECK(flags == (kEvContact | kEvHit)); // either-fixture rule (Box2D contact.c:253, :535)
}

TEST_CASE("A flag change leaves an existing contact's bits alone and applies to the next contact",
          "[physics][events]")
{
    // Zero-g so the authored overlaps stay put (spec s6.1 / amendment A3:
    // contact and hit bits are fixed when the pool contact is created).
    WorldDef wd;
    wd.gravityX = Real(0);
    wd.gravityY = Real(0);
    PhysicsWorld w{ wd };

    const BodyHandle g = AddGround(w, /*contactEvents*/ false);
    const BodyHandle b = AddBox(w, Real(0), Real(-0.49), /*contactEvents*/ true, /*hit*/ true);
    w.Step(kStep);

    const FixtureHandle bf = w.GetBodyFixture(b, 0);
    const FixtureHandle gf = w.GetBodyFixture(g, 0);
    REQUIRE(w.DebugContactEventFlags(bf, gf) == (kEvContact | kEvHit));

    // Ground never opted in, so a per-step recompute would drop kEvContact.
    w.SetFixtureEvents(bf, /*contact*/ false, /*sensor*/ false, /*hit*/ true);
    w.Step(kStep);
    CHECK(w.DebugContactEventFlags(bf, gf) == (kEvContact | kEvHit));

    // New box overlaps b only, and opts out of both. The fresh pair must take
    // b's updated bits (hit, no contact), not the bits captured for b-ground.
    const BodyHandle b2 = AddBox(w, Real(0), Real(-1.4),
                                 /*contactEvents*/ false, /*hit*/ false, /*sensor*/ false);
    w.Step(kStep);
    const std::uint8_t fresh = w.DebugContactEventFlags(bf, w.GetBodyFixture(b2, 0));
    REQUIRE(fresh != 0xFF);
    CHECK(fresh == kEvHit);
    CHECK(w.DebugContactEventFlags(bf, gf) == (kEvContact | kEvHit));
}

TEST_CASE("contactEventsRequireBoth needs both fixtures; hit stays either", "[physics][events]")
{
    WorldDef wd; wd.contactEventsRequireBoth = true;
    PhysicsWorld w{ wd };
    const BodyHandle g = AddGround(w, /*contactEvents*/ false);
    const BodyHandle b = AddBox(w, Real(0), Real(-0.49), /*contactEvents*/ true, /*hit*/ true);
    w.Step(kStep);
    CHECK(w.DebugContactEventFlags(w.GetBodyFixture(b, 0), w.GetBodyFixture(g, 0)) == kEvHit);
}

TEST_CASE("A sensor pair captures no contact or hit flags", "[physics][events]")
{
    PhysicsWorld w{ WorldDef{} };
    AddGround(w);
    BodyDef d; d.type = BodyType::Dynamic; d.fixedRotation = true; d.position = Vec2(Real(0), Real(-2));
    d.shape = MakeAabb(Real(0.5), Real(0.5)); d.isSensor = true; d.contactEvents = true; d.hitEvents = true;
    const BodyHandle s = w.AddBody(d);
    const BodyHandle o = AddBox(w, Real(0), Real(-2), true, true);   // overlapping mover pair
    w.Step(kStep);
    const std::uint8_t f = w.DebugContactEventFlags(w.GetBodyFixture(s, 0), w.GetBodyFixture(o, 0));
    CHECK((f == 0xFF || f == 0u));       // either no pool contact or a non-solver one
}

TEST_CASE("A box landing on static ground begins once and ends when lifted", "[physics][events]")
{
    PhysicsWorld w{ WorldDef{} };
    const BodyHandle g = AddGround(w);
    const BodyHandle b = AddBox(w, Real(0), Real(-2));
    EventLog log;
    log.StepAndCollect(w, 120);
    REQUIRE(log.begin.size() == 1);                 // dynamic-vs-static now reports (the old events skipped it)
    CHECK(log.begin[0].bodyA == b);                 // canonical: A is the dynamic side (Contact.hpp:65)
    CHECK(log.begin[0].bodyB == g);
    CHECK(log.end.empty());
    w.SetPosition(b, Vec2(Real(0), Real(-5)));      // teleport away
    log.StepAndCollect(w, 2);
    CHECK(log.end.size() == 1);
}

TEST_CASE("Two dynamic boxes report begin; kinematic-vs-static reports nothing", "[physics][events]")
{
    PhysicsWorld w{ WorldDef{} };
    AddGround(w);
    AddBox(w, Real(0), Real(-0.5));
    AddBox(w, Real(0), Real(-1.6));
    BodyDef k; k.type = BodyType::Kinematic; k.position = Vec2(Real(5), Real(-0.4));
    k.shape = MakeAabb(Real(0.5), Real(0.5)); k.contactEvents = true;
    w.AddBody(k);                                   // overlaps the ground: no solver contact in Box2D terms
    EventLog log;
    log.StepAndCollect(w, 90);
    CHECK(log.begin.size() == 2);                   // box-ground, box-box; never kinematic-ground (A9)
}

TEST_CASE("Opting out on both fixtures silences the pair", "[physics][events]")
{
    PhysicsWorld w{ WorldDef{} };
    AddGround(w, false);
    AddBox(w, Real(0), Real(-2), false);
    EventLog log;
    log.StepAndCollect(w, 120);
    CHECK(log.begin.empty());
}

TEST_CASE("Begin arrays are sorted by fixture pair", "[physics][events]")
{
    PhysicsWorld w{ WorldDef{} };
    AddGround(w);
    for (int i = 0; i < 6; ++i) AddBox(w, Real(-5 + 2 * i), Real(-0.49));   // all land in step 1
    w.Step(kStep);
    const ContactEvents c = w.GetContactEvents();
    REQUIRE(c.begin.size() == 6);
    for (std::size_t i = 1; i < c.begin.size(); ++i)
        CHECK((FixtureLess(c.begin[i - 1].a, c.begin[i].a) ||
               (c.begin[i - 1].a == c.begin[i].a && FixtureLess(c.begin[i - 1].b, c.begin[i].b))));
}

TEST_CASE("the world gate drops events without a burst on re-enable", "[physics][events]")
{
    PhysicsWorld w{ WorldDef{} };
    AddGround(w);
    const BodyHandle b = AddBox(w, Real(0), Real(-2));
    w.SetEventsEnabled(false);
    EventLog log;
    log.StepAndCollect(w, 120);                     // lands while gated
    CHECK(log.begin.empty());
    w.SetEventsEnabled(true);
    log.StepAndCollect(w, 10);                      // still touching
    CHECK(log.begin.empty());                       // no burst
    CHECK(log.end.empty());                         // and no orphan End
    w.SetPosition(b, Vec2(Real(0), Real(-5)));      // separate after re-enable
    log.StepAndCollect(w, 2);
    CHECK(log.end.empty());                         // no End: its Begin was never delivered (R10)
}

TEST_CASE("disabling the gate mid-contact still delivers the End", "[physics][events]")
{
    PhysicsWorld w{ WorldDef{} };
    AddGround(w);
    const BodyHandle b = AddBox(w, Real(0), Real(-2));
    EventLog log;
    log.StepAndCollect(w, 120);                     // lands while the gate is open
    REQUIRE(log.begin.size() == 1);
    CHECK(log.end.empty());
    w.SetEventsEnabled(false);
    w.SetPosition(b, Vec2(Real(0), Real(-5)));      // separate while gated
    log.StepAndCollect(w, 2);
    CHECK(log.begin.size() == 1);                   // no further Begin
    CHECK(log.end.size() == 1);                     // the reported Begin still closes (R10)
}

TEST_CASE("Removing a touching body ends the contact on the next step, once", "[physics][events]")
{
    PhysicsWorld w{ WorldDef{} };
    AddGround(w);                                   // contactEvents on (R10)
    const BodyHandle b = AddBox(w, Real(0), Real(-2));
    EventLog settle; settle.StepAndCollect(w, 120);
    REQUIRE(settle.begin.size() == 1);
    // No Step between the remove and the two Steps below. The End is buffered,
    // delivered on the first of those Steps, and not repeated on the second.
    w.RemoveBody(b);
    EventLog log;
    log.StepAndCollect(w, 1);
    REQUIRE(log.end.size() == 1);
    CHECK(log.end[0].bodyA == b);                   // old generation (Contact::genA), not the bumped one
    log.StepAndCollect(w, 1);
    CHECK(log.end.size() == 1);                     // delivered once, on the first step only
}

TEST_CASE("DropFixture and a filter change end a touching contact", "[physics][events]")
{
    PhysicsWorld w{ WorldDef{} };
    AddGround(w);
    const BodyHandle b1 = AddBox(w, Real(-3), Real(-2));
    const BodyHandle b2 = AddBox(w, Real(3), Real(-2));
    EventLog settle; settle.StepAndCollect(w, 120);
    REQUIRE(settle.begin.size() == 2);
    // Keep b1 a body: capture fixture 0 before AddFixture (drop is swap-remove;
    // add appends, but the handle is taken first so it cannot be the new one),
    // add a second fixture, then drop the original.
    const FixtureHandle original = w.GetBodyFixture(b1, 0);
    const FixtureHandle b2Fixture = w.GetBodyFixture(b2, 0);
    FixtureDef extra; extra.shape = MakeCircle(Real(0.1)); extra.localPos = Vec2(Real(0), Real(-3));
    w.AddFixture(b1, extra);
    w.DropFixture(original);
    w.SetBodyFilter(b2, 2u, 0u);                    // collides with nothing now
    EventLog log; log.StepAndCollect(w, 1);
    REQUIRE(log.end.size() == 2);
    // Exactly one End per contact. A pair of Ends for one body and none for
    // the other would still have size 2.
    int endsB1 = 0;
    int endsB2 = 0;
    for (const ContactEndEvent& e : log.end)
    {
        const bool hit1 = (e.bodyA == b1 || e.bodyB == b1) &&
                          (e.a == original || e.b == original);
        const bool hit2 = (e.bodyA == b2 || e.bodyB == b2) &&
                          (e.a == b2Fixture || e.b == b2Fixture);
        if (hit1 && !hit2) ++endsB1;
        if (hit2 && !hit1) ++endsB2;
    }
    CHECK(endsB1 == 1);                             // dropped fixture 0, not the added circle
    CHECK(endsB2 == 1);                             // b2's fixture, ended by the filter
}

TEST_CASE("A contact that separates by fat box while touching still ends", "[physics][events]")
{
    PhysicsWorld w{ WorldDef{} };
    AddGround(w);
    const BodyHandle b = AddBox(w, Real(0), Real(-2));
    EventLog settle; settle.StepAndCollect(w, 120);
    REQUIRE(settle.begin.size() == 1);              // Begin was reported (R10)
    w.SetPosition(b, Vec2(Real(0), Real(-50)));     // far beyond the fat AABB in one move
    EventLog log; log.StepAndCollect(w, 1);
    CHECK(log.end.size() == 1);
}

namespace
{
    // 256 dynamics: SoftStep::BuildStages sizes body blocks at a minimum of 32
    // awake bodies (SoftStep.cpp, kBodyMinBlock) and a target of 4*WorkerCount().
    // ceil(256/32) = 8, so any executor with WorkerCount() > 1 gets multiple
    // body blocks to steal. 20 boxes never crossed that threshold.
    constexpr int kDynamicBoxes = 256;
    constexpr int kColumns      = 64;   // 4 rows; the resting stack stays short
    constexpr int kSteps        = 160;
    constexpr int kSeparateAt   = 120;  // shift off the ground so contact Ends fire
    constexpr int kBodyMinBlock = 32;   // SoftStep.cpp BuildStages

    struct EventTrace
    {
        std::vector<std::uint32_t> words;
        std::uint32_t contactBegin = 0;
        std::uint32_t contactEnd   = 0;
        std::uint32_t contactHit   = 0;
        std::uint32_t sensorBegin  = 0;
        std::uint32_t sensorEnd    = 0;
    };

    void PushRealBits(std::vector<std::uint32_t>& words, Real v)
    {
        static_assert(sizeof(Real) == sizeof(std::uint32_t));
        words.push_back(std::bit_cast<std::uint32_t>(v));
    }

    void PushFixture(std::vector<std::uint32_t>& words, FixtureHandle h)
    {
        words.push_back(h.index);
        words.push_back(h.generation);
    }

    void PushBody(std::vector<std::uint32_t>& words, BodyHandle h)
    {
        words.push_back(h.index);
        words.push_back(h.generation);
    }

    // Declaration order from Events.hpp. The tag keeps the five arrays apart
    // inside one word stream.
    template <typename Event>
    void PushPair(std::vector<std::uint32_t>& words, std::uint32_t tag, const Event& e)
    {
        words.push_back(tag);
        PushFixture(words, e.a);
        PushFixture(words, e.b);
        PushBody(words, e.bodyA);
        PushBody(words, e.bodyB);
    }

    template <typename Event>
    void PushSensor(std::vector<std::uint32_t>& words, std::uint32_t tag, const Event& e)
    {
        words.push_back(tag);
        PushFixture(words, e.sensor);
        PushFixture(words, e.visitor);
        PushBody(words, e.sensorBody);
        PushBody(words, e.visitorBody);
    }

    void AppendStep(EventTrace& trace, const PhysicsWorld& w)
    {
        const ContactEvents c = w.GetContactEvents();
        for (const ContactBeginEvent& e : c.begin)
        {
            PushPair(trace.words, 1u, e);
            ++trace.contactBegin;
        }
        for (const ContactEndEvent& e : c.end)
        {
            PushPair(trace.words, 2u, e);
            ++trace.contactEnd;
        }
        for (const ContactHitEvent& e : c.hit)
        {
            PushPair(trace.words, 3u, e);
            PushRealBits(trace.words, e.point.x);
            PushRealBits(trace.words, e.point.y);
            PushRealBits(trace.words, e.normal.x);
            PushRealBits(trace.words, e.normal.y);
            PushRealBits(trace.words, e.approachSpeed);
            ++trace.contactHit;
        }
        const SensorEvents s = w.GetSensorEvents();
        for (const SensorBeginEvent& e : s.begin)
        {
            PushSensor(trace.words, 4u, e);
            ++trace.sensorBegin;
        }
        for (const SensorEndEvent& e : s.end)
        {
            PushSensor(trace.words, 5u, e);
            ++trace.sensorEnd;
        }
    }

    void RequirePopulated(const EventTrace& trace)
    {
        REQUIRE(trace.contactBegin > 0u);
        REQUIRE(trace.contactEnd > 0u);
        REQUIRE(trace.contactHit > 0u);
        REQUIRE(trace.sensorBegin > 0u);
        REQUIRE(trace.sensorEnd > 0u);
    }

    // 64x4 grid of dynamic boxes dropped onto a wide ground, through three
    // static sensor bands that sit below the spawn and above the resting
    // stack. Restitution is 0, so a settled pair never separates: at
    // kSeparateAt every box is shifted off the ground and the next steps
    // deliver the contact Ends. nullptr keeps the world's serial executor.
    EventTrace RunEventTrace(Mosaic::IWorkScheduler* exec)
    {
        PhysicsWorld w{ WorldDef{} };
        w.SetExecutor(exec);

        BodyDef ground;
        ground.type          = BodyType::Static;
        ground.position      = Vec2(Real(0), Real(0.5)); // top face at y = 0
        ground.shape         = MakeAabb(Real(40), Real(0.5));
        ground.contactEvents = true;
        w.AddBody(ground);

        // Bands the boxes fall through. Half-height 0.15. +Y is down, so these
        // y values are above the ground and clear of both the spawn (bottoms
        // at y=-7.5) and a 4-high rest stack (top face near y=-4).
        for (const Real sy : { Real(-6.8f), Real(-6.0f), Real(-5.0f) })
        {
            BodyDef band;
            band.type         = BodyType::Static;
            band.position     = Vec2(Real(0), sy);
            band.shape        = MakeAabb(Real(40), Real(0.15f));
            band.isSensor     = true;
            band.sensorEvents = true;
            w.AddBody(band);
        }

        std::vector<BodyHandle> boxes;
        boxes.reserve(static_cast<std::size_t>(kDynamicBoxes));
        for (int i = 0; i < kDynamicBoxes; ++i)
        {
            const int c = i % kColumns;
            const int r = i / kColumns;
            const Real x = (static_cast<Real>(c) - Real(31.5f)) * Real(1.05f);
            const Real y = Real(-8) - static_cast<Real>(r) * Real(1.15f);
            boxes.push_back(AddBox(w, x, y, /*contactEvents*/ true, /*hitEvents*/ true));
        }

        EventTrace trace;
        for (int s = 0; s < kSteps; ++s)
        {
            if (s == kSeparateAt)
            {
                for (const BodyHandle h : boxes)
                {
                    const Vec2 p = w.Position(h);
                    w.SetPosition(h, Vec2(p.x + Real(100), p.y));
                }
            }
            w.Step(kStep);
            AppendStep(trace, w);
        }
        return trace;
    }
}

TEST_CASE("Event arrays are byte-identical across runs", "[physics][events][determinism]")
{
    const EventTrace a = RunEventTrace(nullptr);
    const EventTrace b = RunEventTrace(nullptr);
    RequirePopulated(a);
    CHECK(a.words == b.words);
}

TEST_CASE("Event arrays are byte-identical serial vs MT", "[physics][events][determinism][solvermt]")
{
    Mosaic::SerialWorkScheduler serial;
    const std::uint32_t hw = std::thread::hardware_concurrency();
    Manifold2D::Testing::TestWorkScheduler many(hw > 1u ? hw : 2u);

    // Same sizing as SoftStep::BuildStages. There is no runtime counter for
    // blocks stolen; WorkerCount() is the hook SolverMtInvarianceTest uses.
    const int maxBodyBlocks = (kDynamicBoxes + kBodyMinBlock - 1) / kBodyMinBlock;
    const int targetBlocks  = 4 * static_cast<int>(many.WorkerCount());
    const int bodyBlocks    = targetBlocks < maxBodyBlocks ? targetBlocks : maxBodyBlocks;
    INFO("workers=" << many.WorkerCount() << " dynamics=" << kDynamicBoxes
         << " bodyMinBlock=" << kBodyMinBlock << " bodyBlocks=" << bodyBlocks);
    REQUIRE(many.WorkerCount() > 1u);
    REQUIRE(bodyBlocks > 1);

    const EventTrace serialTrace = RunEventTrace(&serial);
    const EventTrace mtTrace     = RunEventTrace(&many);
    RequirePopulated(serialTrace);
    RequirePopulated(mtTrace);
    CHECK(serialTrace.words == mtTrace.words);
}
