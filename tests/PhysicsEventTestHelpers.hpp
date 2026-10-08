#pragma once
// Shared helpers for the [physics][events] suites: an EventLog that copies each
// step's arrays (they are only valid until the next Step) and tiny scene builders.
// +Y DOWN, gravity 10 (WorldDef defaults).
#include <vector>
#include <Manifold2D/Physics/PhysicsWorld.hpp>
#include <Manifold2D/Physics/Events.hpp>

namespace EventTest
{
    using namespace Manifold2D::Physics;
    inline constexpr Real kStep = Real(1) / Real(60);

    struct EventLog
    {
        std::vector<ContactBeginEvent> begin;
        std::vector<ContactEndEvent>   end;
        std::vector<ContactHitEvent>   hit;
        std::vector<SensorBeginEvent>  sensorBegin;
        std::vector<SensorEndEvent>    sensorEnd;

        void Collect(const PhysicsWorld& w)
        {
            const ContactEvents c = w.GetContactEvents();
            begin.insert(begin.end(), c.begin.begin(), c.begin.end());
            end.insert(end.end(), c.end.begin(), c.end.end());
            hit.insert(hit.end(), c.hit.begin(), c.hit.end());
            const SensorEvents s = w.GetSensorEvents();
            sensorBegin.insert(sensorBegin.end(), s.begin.begin(), s.begin.end());
            sensorEnd.insert(sensorEnd.end(), s.end.begin(), s.end.end());
        }
        void StepAndCollect(PhysicsWorld& w, int steps)
        {
            for (int i = 0; i < steps; ++i) { w.Step(kStep); Collect(w); }
        }
    };

    // Static ground slab whose top face is y = 0, 20 m wide.
    inline BodyHandle AddGround(PhysicsWorld& w, bool contactEvents = true, bool sensorEvents = true)
    {
        BodyDef d;
        d.type = BodyType::Static;
        d.position = Vec2(Real(0), Real(0.5));
        d.shape = MakeAabb(Real(10), Real(0.5));
        d.contactEvents = contactEvents;
        d.sensorEvents = sensorEvents;
        return w.AddBody(d);
    }

    // Dynamic 1 m box whose centre starts at (x, y).
    inline BodyHandle AddBox(PhysicsWorld& w, Real x, Real y, bool contactEvents = true,
                             bool hitEvents = false, bool sensorEvents = true)
    {
        BodyDef d;
        d.type = BodyType::Dynamic;
        d.fixedRotation = true; // dynamic AABBs must be fixedRotation (PhysicsWorld.cpp AddBody)
        d.position = Vec2(x, y);
        d.shape = MakeAabb(Real(0.5), Real(0.5));
        d.contactEvents = contactEvents;
        d.hitEvents = hitEvents;
        d.sensorEvents = sensorEvents;
        return w.AddBody(d);
    }
}
