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
// Two surfaces share the class. The HERO surface (addStaticBox / addBox /
// addCircle / addPolygon / the tumbler hexagon) is what the step-dissection hero
// was built on and is unchanged. The SCENE surface (setMaterial .. jointCount,
// then wake .. setJointMotor) is general: any body type and shape, compound
// fixtures, removal, teleport, impulses, kinematic motion, all seven joint
// kinds, a radial gravity well with per-body gravity scale, a per-step contact
// export, and live motor control -- enough to stage a living scene whose
// denizens arrive, interact and leave. Everything is
// addressed by body SLOT (the handle index) and joint ID (an index into this
// binding's own table); a bad slot or id is refused (-1 / false / no-op),
// never an abort, because the wasm build has no exception catching.
//
// MEMORY GROWTH detaches views: JS must copy (slice()) what bodies()/trace()
// return before the next call into the module.
//
// Units are MKS, +Y is DOWN (the engine's default gravity is (0, 10)).

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <Manifold2D/Physics/PhysicsTypes.hpp>
#include <Manifold2D/Physics/Shapes.hpp>
#include <Manifold2D/Physics/Fixture.hpp>
#include <Manifold2D/Physics/PhysicsWorld.hpp>
#include <Manifold2D/Physics/StepTrace.hpp>
#include <Manifold2D/Physics/Joints/Joint.hpp>
#include <Manifold2D/Physics/Joints/Joints.hpp>     // MouseJoint::SetTarget
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

    // Scene-surface shape kinds (addBody / addFixture).
    constexpr int kShapeCircle  = 0;   // p0 = radius
    constexpr int kShapeBox     = 1;   // p0, p1 = half-extents
    constexpr int kShapeCapsule = 2;   // p0 = segment half-length (local x), p1 = radius
    constexpr int kShapeNgon    = 3;   // p0 = circumradius, p1 = sides (3..12)
    constexpr int kMaxNgonSides = 12;
    constexpr std::size_t kMaxSceneVerts = 16;
    // removeBody wakes bodies within this distance of the removed body's AABB.
    constexpr Real kWakeMargin = Real(0.05);

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

    // ---- the scene surface ------------------------------------------------

    // Material and flags for every body and fixture added AFTER the call
    // (the scene surface only; the hero surface keeps its fixed materials).
    void setMaterial(float density, float friction, float restitution, float linearDamping)
    {
        m_next.density       = density > 0.0f ? static_cast<Real>(density) : Real(1);
        m_next.friction      = static_cast<Real>(std::max(0.0f, friction));
        m_next.restitution   = static_cast<Real>(std::max(0.0f, restitution));
        m_next.linearDamping = static_cast<Real>(std::max(0.0f, linearDamping));
    }
    // A contact needs (A.category & B.mask) and (B.category & A.mask) non-zero.
    void setFilter(std::uint32_t category, std::uint32_t mask)
    {
        m_next.category = category;
        m_next.mask     = mask;
    }
    // The collision group for the next bodies and fixtures (b2Filter.groupIndex):
    // a shared negative group never collides (a ragdoll's parts), a shared
    // positive one always does, 0 leaves it to the category/mask rule.
    void setFilterGroup(int group) { m_next.group = group; }
    void setFixedRotation(bool on) { m_next.fixedRotation = on; }

    // More options for the next bodies: sleepThreshold (m/s; < 0 inherits the
    // world's 0.05, 0 = never sleeps -- drones, anything a controller steers),
    // bullet (CCD against statics) and gravityScale (world gravity + the well).
    void setBodyOptions(float sleepThreshold, bool bullet, float gravityScale)
    {
        m_next.sleepThreshold = static_cast<Real>(sleepThreshold);
        m_next.bullet         = bullet;
        m_next.gravityScale   = std::isfinite(gravityScale) ? static_cast<Real>(gravityScale) : Real(1);
    }

    // type: 0 static, 1 kinematic, 2 dynamic. kind: see kShape*. Returns the
    // slot, or -1 for a bad type, kind or size.
    int addBody(int type, int kind, float x, float y, float angle, float p0, float p1)
    {
        if (type < 0 || type > 2) { return -1; }
        Shape shape;
        if (!makeShape(kind, p0, p1, shape)) { return -1; }
        return addBodyWithShape(type, x, y, angle, shape);
    }

    // A convex polygon from a flat [x0, y0, x1, y1, ...] list (3..16 vertices,
    // either winding). Too few, collinear or concave -> -1.
    int addBodyPoly(int type, float x, float y, float angle, emscripten::val verts)
    {
        if (type < 0 || type > 2) { return -1; }
        Shape shape;
        if (!makeConvex(verts, shape)) { return -1; }
        return addBodyWithShape(type, x, y, angle, shape);
    }

    // An extra fixture on a live body, in the body's frame (compound bodies:
    // a truck's cab on its bed). Returns the fixture index or -1.
    int addFixture(int slot, int kind, float lx, float ly, float langle, float p0, float p1)
    {
        BodyHandle h;
        Shape shape;
        if (!handle(slot, h) || !makeShape(kind, p0, p1, shape)) { return -1; }
        return addFixtureWithShape(h, lx, ly, langle, shape);
    }
    int addFixturePoly(int slot, float lx, float ly, float langle, emscripten::val verts)
    {
        BodyHandle h;
        Shape shape;
        if (!handle(slot, h) || !makeConvex(verts, shape)) { return -1; }
        return addFixtureWithShape(h, lx, ly, langle, shape);
    }

    // Destroy a body: its contacts, its joints (their ids then refuse every
    // call) and its kinematic motion go with it; the slot may be reused.
    // Everything near it or jointed to it is woken afterwards: the engine's
    // RemoveBody does not wake neighbours, so a sleeping crate would otherwise
    // hang in the air when the floor under it is removed. "Near" is the body's
    // AABB grown by kWakeMargin -- a geometric test, because the contact-event
    // table does not reliably hold a sleeping body's pair with a static one.
    bool removeBody(int slot)
    {
        BodyHandle h;
        if (!handle(slot, h)) { return false; }
        std::vector<BodyHandle> partners;
        Aabb box = m_world->SlotAabb(h.index);
        box.min = Vec2(box.min.x - kWakeMargin, box.min.y - kWakeMargin);
        box.max = Vec2(box.max.x + kWakeMargin, box.max.y + kWakeMargin);
        m_world->QueryAABB(box, partners);
        for (Joint*& j : m_joints)
        {
            // RemoveBody frees these joints itself; forget them first.
            if (j != nullptr && (j->HandleA() == h || j->HandleB() == h))
            {
                partners.push_back(j->HandleA() == h ? j->HandleB() : j->HandleA());
                j = nullptr;
            }
        }
        m_kin.erase(std::remove_if(m_kin.begin(), m_kin.end(),
                                   [&](const Kin& k) { return k.h == h; }),
                    m_kin.end());
        m_world->RemoveBody(h);
        wakeAll(partners);
        return true;
    }
    bool isAlive(int slot) const
    {
        BodyHandle h;
        return handle(slot, h);
    }

    // Teleport (no swept motion between the old and new pose). A dynamic body
    // is woken: the engine's SetPosition does not, so a sleeping body would
    // otherwise hang at its new pose.
    void setTransform(int slot, float x, float y, float angle)
    {
        BodyHandle h;
        if (!handle(slot, h)) { return; }
        m_world->SetPosition(h, Vec2(static_cast<Real>(x), static_cast<Real>(y)));
        m_world->SetAngle(h, static_cast<Real>(angle));
        for (Kin& k : m_kin) { if (k.h == h) { k.angle = static_cast<Real>(angle); } }
        if (m_world->GetType(h) == BodyType::Dynamic) { m_world->Wake(h); }
    }
    void setVelocity(int slot, float vx, float vy, float w)
    {
        BodyHandle h;
        if (!handle(slot, h)) { return; }
        m_world->SetVelocity(h, Vec2(static_cast<Real>(vx), static_cast<Real>(vy)));
        m_world->SetAngularVelocity(h, static_cast<Real>(w));
    }
    // A linear impulse at world point (px, py). Dynamic bodies only.
    void applyImpulse(int slot, float ix, float iy, float px, float py)
    {
        BodyHandle h;
        if (!handle(slot, h)) { return; }
        m_world->ApplyImpulse(h, Vec2(static_cast<Real>(ix), static_cast<Real>(iy)),
                              Vec2(static_cast<Real>(px), static_cast<Real>(py)));
    }
    // Drive a kinematic body. The engine integrates a kinematic body's
    // position but not its angle (see setHexagonSpin), so the binding advances
    // the angle before every step, the same way it does for the hexagon.
    void setKinematic(int slot, float vx, float vy, float w)
    {
        BodyHandle h;
        if (!handle(slot, h) || m_world->GetType(h) != BodyType::Kinematic) { return; }
        m_world->SetVelocity(h, Vec2(static_cast<Real>(vx), static_cast<Real>(vy)));
        m_world->SetAngularVelocity(h, static_cast<Real>(w));
        for (Kin& k : m_kin)
        {
            if (k.h == h) { k.omega = static_cast<Real>(w); return; }
        }
        m_kin.push_back(Kin{ h, static_cast<Real>(w), m_world->GetAngle(h) });
    }

    // ---- joints. Each returns a joint id, or -1 for a bad slot. ----------
    // Distance: hold the CENTRES at `length` (<= 0: their current distance).
    int addDistanceJoint(int a, int b, float length)
    {
        JointDef d;
        d.kind   = JointKind::Distance;
        d.length = static_cast<Real>(length);
        return addJoint(a, b, d);
    }
    // Distance between ANCHOR points (b2DistanceJointDef localAnchorA/B):
    // world point (ax, ay) on A and (bx, by) on B, converted to each body's
    // local frame now, so the joint torques both bodies (a sling on a corner,
    // a two-leg bridle). `length` <= 0: the current distance between the
    // anchors. Rigid until setDistanceSpring / setDistanceLimits say otherwise.
    int addDistanceJointAt(int a, int b, float ax, float ay, float bx, float by, float length)
    {
        JointDef d;
        d.kind = JointKind::Distance;
        if (!handle(a, d.a) || !handle(b, d.b) || a == b) { return -1; }
        d.localAnchorA = toLocal(d.a, ax, ay);
        d.localAnchorB = toLocal(d.b, bx, by);
        d.length = static_cast<Real>(length);
        return storeJoint(m_world->AddJoint(d));
    }
    // Revolute: pin A and B at world point (ax, ay).
    int addRevoluteJoint(int a, int b, float ax, float ay)
    {
        JointDef d;
        d.kind   = JointKind::Revolute;
        d.anchor = Vec2(static_cast<Real>(ax), static_cast<Real>(ay));
        return addJoint(a, b, d);
    }
    // Weld: revolute at (ax, ay) plus the relative angle locked as it is now.
    int addWeldJoint(int a, int b, float ax, float ay)
    {
        JointDef d;
        d.kind   = JointKind::Weld;
        d.anchor = Vec2(static_cast<Real>(ax), static_cast<Real>(ay));
        return addJoint(a, b, d);
    }
    // Prismatic: B slides along the world axis only, no relative rotation.
    int addPrismaticJoint(int a, int b, float axisX, float axisY)
    {
        JointDef d;
        d.kind = JointKind::Prismatic;
        d.axis = Vec2(static_cast<Real>(axisX), static_cast<Real>(axisY));
        return addJoint(a, b, d);
    }
    // Wheel: B rides A's axis (A's frame, now) on a spring of hz / zeta at world
    // anchor (ax, ay); maxTorque > 0 drives B's spin toward motorSpeed (rad/s,
    // positive = clockwise on screen with +y down, so a car rolls to +x).
    int addWheelJoint(int a, int b, float ax, float ay, float axisX, float axisY,
                      float hz, float zeta, float motorSpeed, float maxTorque)
    {
        JointDef d;
        d.kind           = JointKind::Wheel;
        d.anchor         = Vec2(static_cast<Real>(ax), static_cast<Real>(ay));
        d.axis           = Vec2(static_cast<Real>(axisX), static_cast<Real>(axisY));
        d.frequencyHz    = static_cast<Real>(hz);
        d.dampingRatio   = static_cast<Real>(zeta);
        d.enableMotor    = maxTorque > 0.0f;
        d.motorSpeed     = static_cast<Real>(motorSpeed);
        d.maxMotorTorque = static_cast<Real>(std::max(0.0f, maxTorque));
        return addJoint(a, b, d);
    }
    // Motor: drive B's angular velocity relative to A toward `speed`.
    int addMotorJoint(int a, int b, float speed, float maxTorque)
    {
        JointDef d;
        d.kind           = JointKind::Motor;
        d.motorSpeed     = static_cast<Real>(speed);
        d.maxMotorTorque = static_cast<Real>(std::max(0.0f, maxTorque));
        return addJoint(a, b, d);
    }
    // Mouse: a soft spring dragging body B toward (tx, ty); move it with
    // setMouseTarget. Body A is none.
    int addMouseJoint(int b, float tx, float ty, float maxForce)
    {
        BodyHandle hb;
        if (!handle(b, hb)) { return -1; }
        JointDef d;
        d.kind     = JointKind::Mouse;
        d.a        = BodyHandle{ kInvalidSlot, 0u };
        d.b        = hb;
        d.target   = Vec2(static_cast<Real>(tx), static_cast<Real>(ty));
        d.maxForce = static_cast<Real>(std::max(0.0f, maxForce));
        return storeJoint(m_world->AddJoint(d));
    }
    // Moving the target wakes the dragged body: the engine skips a joint whose
    // body sleeps, so a body that dozed at its old target would never follow.
    bool setMouseTarget(int id, float tx, float ty)
    {
        Joint* j = joint(id);
        auto* mj = j != nullptr ? dynamic_cast<MouseJoint*>(j) : nullptr;
        if (mj == nullptr) { return false; }
        mj->SetTarget(Vec2(static_cast<Real>(tx), static_cast<Real>(ty)));
        if (m_world->IsValid(mj->HandleB())) { m_world->Wake(mj->HandleB()); }
        return true;
    }

    // Live motor control for a wheel, motor or prismatic joint. Wheel/motor:
    // speed in rad/s, maxTorque in N*m. Prismatic: speed in m/s along the
    // axis, maxTorque is the force limit in N (a press, piston or gate that
    // stalls against what it cannot move). > 0 enables the motor at that
    // limit, 0 lets it coast. false for any other joint kind or a dead id.
    bool setJointMotor(int id, float speed, float maxTorque)
    {
        Joint* j = joint(id);
        if (j == nullptr) { return false; }
        const Real sp = static_cast<Real>(speed);
        const Real tq = static_cast<Real>(std::max(0.0f, maxTorque));
        if (auto* wj = dynamic_cast<WheelJoint*>(j))
        {
            wj->EnableMotor(tq > Real(0));
            wj->SetMotorSpeed(sp);
            wj->SetMaxMotorTorque(tq);
        }
        else if (auto* mj = dynamic_cast<MotorJoint*>(j))
        {
            mj->SetMotorSpeed(sp);
            mj->SetMaxMotorTorque(tq);
        }
        else if (auto* pj = dynamic_cast<PrismaticJoint*>(j))
        {
            pj->EnableMotor(tq > Real(0));
            pj->SetMotorSpeed(sp);
            pj->SetMaxMotorForce(tq);
        }
        else if (auto* rj = revolute(j))
        {
            rj->EnableMotor(tq > Real(0)); // rad/s, N m (b2RevoluteJoint_SetMaxMotorTorque)
            rj->SetMotorSpeed(sp);
            rj->SetMaxMotorTorque(tq);
        }
        else
        {
            return false;
        }
        if (m_world->IsValid(j->HandleA())) { m_world->Wake(j->HandleA()); }
        if (m_world->IsValid(j->HandleB())) { m_world->Wake(j->HandleB()); }
        return true;
    }

    // ---- revolute joint (b2RevoluteJoint_*) --------------------------------
    // The joint angle is angleB - angleA - reference (Box2D: the angle between the
    // joint frames; the reference starts at 0, so pass the relative angle at
    // creation to measure the limits and the spring from that pose).
    bool setRevoluteReference(int id, float angle)
    {
        RevoluteJoint* rj = revolute(joint(id));
        if (rj == nullptr) { return false; }
        rj->SetReferenceAngle(static_cast<Real>(angle));
        wakeJoint(rj);
        return true;
    }
    // lower <= joint angle <= upper (radians), or off.
    bool setRevoluteLimits(int id, bool enable, float lower, float upper)
    {
        RevoluteJoint* rj = revolute(joint(id));
        if (rj == nullptr) { return false; }
        rj->EnableLimit(enable);
        rj->SetLimits(static_cast<Real>(lower), static_cast<Real>(upper));
        wakeJoint(rj);
        return true;
    }
    // A rotational spring-damper driving the joint angle to `target` (radians),
    // at `hertz` with `dampingRatio` (1 = critical), or off.
    bool setRevoluteSpring(int id, bool enable, float hertz, float dampingRatio, float target)
    {
        RevoluteJoint* rj = revolute(joint(id));
        if (rj == nullptr) { return false; }
        rj->EnableSpring(enable);
        rj->SetSpring(static_cast<Real>(hertz), static_cast<Real>(dampingRatio));
        rj->SetTargetAngle(static_cast<Real>(target));
        wakeJoint(rj);
        return true;
    }
    // The revolute joint's current angle (radians); NaN for any other joint.
    float jointAngle(int id) const
    {
        const RevoluteJoint* rj = revolute(joint(id));
        return rj != nullptr ? static_cast<float>(rj->JointAngle(*m_world)) : std::numeric_limits<float>::quiet_NaN();
    }

    // ---- prismatic joint (b2PrismaticJoint_*) -------------------------------
    // The translation is B's slide relative to A along the axis, in metres,
    // zero at creation (b2PrismaticJoint_GetTranslation).
    // lower <= translation <= upper (m), or off (b2PrismaticJoint_EnableLimit /
    // _SetLimits; the bounds are sorted). A motor driven into the limit stops
    // there. false for any other joint kind or a dead id.
    bool setPrismaticLimits(int id, bool enable, float lower, float upper)
    {
        Joint* j = joint(id);
        auto* pj = j != nullptr ? dynamic_cast<PrismaticJoint*>(j) : nullptr;
        if (pj == nullptr) { return false; }
        pj->EnableLimit(enable);
        pj->SetLimits(static_cast<Real>(lower), static_cast<Real>(upper));
        wakeJoint(pj);
        return true;
    }
    // The prismatic joint's current translation (m); NaN for any other joint
    // or a dead id.
    float jointTranslation(int id) const
    {
        const Joint* j = joint(id);
        const auto* pj = j != nullptr ? dynamic_cast<const PrismaticJoint*>(j) : nullptr;
        return pj != nullptr ? static_cast<float>(pj->GetTranslation(*m_world)) : std::numeric_limits<float>::quiet_NaN();
    }

    // ---- distance joint (b2DistanceJoint_*) ---------------------------------
    // Box2D's rules: the limit is solved only while the spring is ON (with the
    // spring off the joint is a rigid rod at its length); a spring of 0 Hz is
    // no length constraint at all. So a ROPE -- slack when the bodies close in,
    // holding at maxLength when pulled -- is
    //   setDistanceSpring(id, true, 0, 0); setDistanceLimits(id, true, 0, maxLength);
    // Lengths are clamped to [0.005, 100000] m (a rope's 0 becomes 5 mm) and
    // sorted. false for any other joint kind or a dead id; wakes the bodies.
    bool setDistanceLimits(int id, bool enable, float minLength, float maxLength)
    {
        DistanceJoint* dj = distanceJoint(joint(id));
        if (dj == nullptr) { return false; }
        dj->EnableLimit(enable);
        dj->SetLengthRange(static_cast<Real>(minLength), static_cast<Real>(maxLength));
        wakeJoint(dj);
        return true;
    }
    // The length as a spring-damper about the joint's length at `hertz` with
    // `dampingRatio` (1 = critical), or off (rigid). hertz 0: no length
    // constraint (a rope, with setDistanceLimits).
    bool setDistanceSpring(int id, bool enable, float hertz, float dampingRatio)
    {
        DistanceJoint* dj = distanceJoint(joint(id));
        if (dj == nullptr) { return false; }
        dj->EnableSpring(enable);
        dj->SetSpring(static_cast<Real>(hertz), static_cast<Real>(dampingRatio));
        wakeJoint(dj);
        return true;
    }
    // The distance joint's current anchor-to-anchor length (m); NaN for any
    // other joint or a dead id.
    float jointLength(int id) const
    {
        const DistanceJoint* dj = distanceJoint(joint(id));
        return dj != nullptr ? static_cast<float>(dj->GetCurrentLength(*m_world)) : std::numeric_limits<float>::quiet_NaN();
    }

    // What joint `id` carried over the last step (b2Joint_GetConstraintForce /
    // b2Joint_GetConstraintTorque): [fx, fy, torque] -- the force in N, world
    // frame, acting on the joint's body B (the second body passed to add*Joint;
    // A feels the opposite), and the joint's pure angular torque in N m (an
    // angle lock, motor, spring or limit; not the force's moment). Averaged over
    // the step: the impulse delivered across its sub-steps / dt. A breakable
    // joint compares hypot(fx, fy) or |torque| to its threshold after each
    // step() and calls removeJoint. A sleeping joint keeps its last reading.
    // NaN x 3 for a dead id. A typed_memory_view: copy before the next call.
    emscripten::val jointReaction(int id)
    {
        const Joint* j = joint(id);
        if (j == nullptr)
        {
            m_reactBuf.fill(std::numeric_limits<float>::quiet_NaN());
        }
        else
        {
            const Vec2 f = m_world->JointReactionForce(j);
            m_reactBuf = { static_cast<float>(f.x), static_cast<float>(f.y),
                           static_cast<float>(m_world->JointReactionTorque(j)) };
        }
        return emscripten::val(emscripten::typed_memory_view(m_reactBuf.size(), m_reactBuf.data()));
    }

    // ---- queries -----------------------------------------------------------
    // The nearest fixture the ray (ox, oy) -> (ox + tx, oy + ty) hits
    // (b2World_CastRayClosest): a fixture is seen when its category is in `mask`
    // and its mask holds `category` (b2QueryFilter); excludeSlot (-1 = none)
    // skips one body and a negative `group` every fixture of that group (a
    // ragdoll's own parts);
    // skips one body; sensors and fixtures containing the origin are skipped.
    // Returns [hit, body, fixture, px, py, nx, ny, fraction] -- hit 0 for a miss;
    // body -1 for a tile span; the normal faces back along the ray. A
    // typed_memory_view: copy before the next call.
    emscripten::val castRayClosest(float ox, float oy, float tx, float ty,
                                   std::uint32_t category, std::uint32_t mask, int excludeSlot, int group)
    {
        QueryFilter f;
        f.categoryBits = category;
        f.maskBits = mask;
        f.groupIndex = group;
        BodyHandle ex;
        if (excludeSlot >= 0 && handle(excludeSlot, ex)) { f.exclude = ex; }
        const std::optional<RayResult> r = m_world->CastRayClosest(
            Vec2(static_cast<Real>(ox), static_cast<Real>(oy)),
            Vec2(static_cast<Real>(tx), static_cast<Real>(ty)), f);
        m_rayBuf.fill(0.0f);
        if (r)
        {
            m_rayBuf = {
                1.0f,
                r->body != kInvalidBody ? static_cast<float>(r->body.index) : -1.0f,
                r->fixture != kInvalidFixture ? static_cast<float>(r->fixture.index) : -1.0f,
                static_cast<float>(r->point.x), static_cast<float>(r->point.y),
                static_cast<float>(r->normal.x), static_cast<float>(r->normal.y),
                static_cast<float>(r->fraction),
            };
        }
        return emscripten::val(emscripten::typed_memory_view(m_rayBuf.size(), m_rayBuf.data()));
    }

    // The tight world AABB of body `slot` at its current pose
    // (b2Body_ComputeAABB): [minX, minY, maxX, maxY], the union of every live
    // fixture's rotated bounds (round shapes grown by their radius) -- NOT the
    // fat, margin-grown broadphase box. A drone with a rope-slung load unions
    // its own box with the load's. NaN x 4 for a dead or invalid slot. A
    // typed_memory_view: copy before the next call.
    emscripten::val bodyAABB(int slot)
    {
        BodyHandle h;
        if (!handle(slot, h))
        {
            m_aabbBuf.fill(std::numeric_limits<float>::quiet_NaN());
        }
        else
        {
            const Aabb box = m_world->SlotAabb(h.index);
            m_aabbBuf = { static_cast<float>(box.min.x), static_cast<float>(box.min.y),
                          static_cast<float>(box.max.x), static_cast<float>(box.max.y) };
        }
        return emscripten::val(emscripten::typed_memory_view(m_aabbBuf.size(), m_aabbBuf.data()));
    }

    // ---- motion helpers ----------------------------------------------------
    void wake(int slot)
    {
        BodyHandle h;
        if (handle(slot, h)) { m_world->Wake(h); }
    }
    bool isAwake(int slot) const
    {
        BodyHandle h;
        return handle(slot, h) && m_world->IsAwake(h);
    }
    // kg; 0 for static, kinematic and bad slots.
    float mass(int slot) const
    {
        BodyHandle h;
        return handle(slot, h) ? static_cast<float>(m_world->GetBodyMass(h)) : 0.0f;
    }
    // A linear impulse at the centre of mass (no torque) -- what a thruster or
    // a controller's capped correction applies.
    void applyLinearImpulse(int slot, float ix, float iy)
    {
        BodyHandle h;
        if (!handle(slot, h) || !std::isfinite(ix) || !std::isfinite(iy)) { return; }
        m_world->ApplyImpulse(h, Vec2(static_cast<Real>(ix), static_cast<Real>(iy)));
    }

    // ---- the gravity well ------------------------------------------------
    // falloff 0 = inverse-square (real orbits), 1 = fade: surface gravity to
    // fadeStart, smoothly to zero at fadeEnd, zero beyond. Evaluated by the
    // engine per body per sub-step (PhysicsWorld::SetGravityWell). An invalid
    // well is ignored. Use with new ManifoldSim(0, substeps) for a pure well.
    void setGravityWell(float cx, float cy, float surfaceRadius, float surfaceGravity,
                        int falloff, float fadeStart, float fadeEnd)
    {
        GravityWell gw;
        gw.enabled        = true;
        gw.center         = Vec2(static_cast<Real>(cx), static_cast<Real>(cy));
        gw.surfaceRadius  = static_cast<Real>(surfaceRadius);
        gw.surfaceGravity = static_cast<Real>(surfaceGravity);
        gw.falloff        = falloff == 1 ? GravityFalloff::Fade : GravityFalloff::InverseSquare;
        gw.fadeStart      = static_cast<Real>(fadeStart);
        gw.fadeEnd        = static_cast<Real>(fadeEnd);
        m_world->SetGravityWell(gw);
    }
    void clearGravityWell()
    {
        GravityWell off;
        off.enabled = false;
        m_world->SetGravityWell(off);
    }
    // Change a live body's collision filter (every fixture), e.g. a walker
    // stepping into another depth lane. Its contacts are rebuilt against the
    // new filter (PhysicsWorld::SetBodyFilter, b2Shape_SetFilter's recipe).
    void setBodyFilter(int slot, std::uint32_t category, std::uint32_t mask)
    {
        BodyHandle h;
        if (handle(slot, h))
        {
            m_world->SetBodyFilter(h, category, mask);
        }
    }

    // Make a live body a bullet or not (b2Body_SetBullet): a dynamic bullet's step
    // is swept against statics, kinematic and non-bullet dynamic bodies and clamped
    // to the time of impact. For things that move fast for a while -- a thrown body.
    void setBullet(int slot, bool bullet)
    {
        BodyHandle h;
        if (handle(slot, h))
        {
            m_world->SetBullet(h, bullet);
        }
    }
    // Continuous collision for fast non-bullet bodies vs statics (b2World_EnableContinuous; on by default).
    void enableContinuous(bool on) { m_world->EnableContinuous(on); }
    bool isBullet(int slot) const
    {
        BodyHandle h;
        return handle(slot, h) && m_world->IsBullet(h);
    }

    void setGravityScale(int slot, float scale)
    {
        BodyHandle h;
        if (handle(slot, h) && std::isfinite(scale)) { m_world->SetGravityScale(h, static_cast<Real>(scale)); }
    }

    // ---- contacts --------------------------------------------------------
    // The last step's solved contacts, 8 floats each:
    //   [bodyA, bodyB, nx, ny, px, py, normalImpulse, approachSpeed]
    // bodyA is always an awake dynamic body; bodyB is -1 for a tile span. The
    // normal points B -> A. (px, py) is A's origin plus the first point's
    // anchor (exact for single-fixture bodies, approximate for compound ones).
    // normalImpulse sums the manifold points; approachSpeed is the fastest
    // closing speed captured at Prepare (> 0 while approaching) -- a landing or
    // a hit reads it. A typed_memory_view: copy before the next call.
    emscripten::val contacts()
    {
        m_contactsBuf.clear();
        m_world->ForEachContactConstraint([&](const ContactConstraint& cc) {
            if (cc.pointCount <= 0) { return; }
            Real impulse = Real(0);
            Real approach = Real(0);
            for (int k = 0; k < cc.pointCount; ++k)
            {
                impulse += cc.points[k].normalImpulse;
                approach = std::max(approach, -cc.points[k].relativeVelocity);
            }
            const Vec2 pa = m_world->PosSlot(cc.bodyA);
            m_contactsBuf.push_back(static_cast<float>(cc.bodyA));
            m_contactsBuf.push_back(cc.bodyBIsBody ? static_cast<float>(cc.bodyB) : -1.0f);
            m_contactsBuf.push_back(static_cast<float>(cc.normal.x));
            m_contactsBuf.push_back(static_cast<float>(cc.normal.y));
            m_contactsBuf.push_back(static_cast<float>(pa.x + cc.points[0].anchorA.x));
            m_contactsBuf.push_back(static_cast<float>(pa.y + cc.points[0].anchorA.y));
            m_contactsBuf.push_back(static_cast<float>(impulse));
            m_contactsBuf.push_back(static_cast<float>(approach));
        });
        return emscripten::val(emscripten::typed_memory_view(m_contactsBuf.size(), m_contactsBuf.data()));
    }
    // Removing a joint wakes both its bodies (a released load must fall).
    bool removeJoint(int id)
    {
        Joint* j = joint(id);
        if (j == nullptr) { return false; }
        const std::vector<BodyHandle> ends{ j->HandleA(), j->HandleB() };
        m_world->RemoveJoint(j);
        wakeAll(ends);
        m_joints[static_cast<std::size_t>(id)] = nullptr;
        return true;
    }
    std::uint32_t jointCount() const
    {
        return static_cast<std::uint32_t>(m_world->JointCount());
    }

    void step(float dt)
    {
        const Real h = static_cast<Real>(dt);
        advanceHexagon(h);
        advanceKinematics(h);
        m_world->Step(h);
    }

    // Step PLUS the 22-stop capture. The packed trace stays readable until the
    // next stepTraced().
    void stepTraced(float dt)
    {
        const Real h = static_cast<Real>(dt);
        advanceHexagon(h);
        advanceKinematics(h);
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
    // What setMaterial / setFilter / setFixedRotation leave for the next add.
    struct NextBody
    {
        Real          density       = kDynDensity;
        Real          friction      = kDynFriction;
        Real          restitution   = kDynRestitution;
        Real          linearDamping = Real(0);
        std::uint32_t category      = 1u;
        std::int32_t  group         = 0;
        std::uint32_t mask          = 0xFFFFFFFFu;
        bool          fixedRotation = false;
        Real          sleepThreshold = Real(-1);  // < 0 inherits the world default
        bool          bullet         = false;
        Real          gravityScale   = Real(1);
    };
    // A kinematic body whose angle the binding advances (setKinematic).
    struct Kin
    {
        BodyHandle h;
        Real       omega = Real(0);
        Real       angle = Real(0);
    };

    // slot -> live handle; false for a negative, out-of-range or dead slot.
    bool handle(int slot, BodyHandle& out) const
    {
        if (slot < 0 || static_cast<std::uint32_t>(slot) >= m_world->Count()) { return false; }
        const std::uint32_t i = static_cast<std::uint32_t>(slot);
        if (!m_world->Alive(i)) { return false; }
        out = m_world->HandleOf(i);
        return true;
    }

    // Validated shapes: a bad kind, a non-positive size or an out-of-range
    // side count is refused here, before the engine (which would throw).
    static bool makeShape(int kind, float p0, float p1, Shape& out)
    {
        const Real a = static_cast<Real>(p0);
        const Real b = static_cast<Real>(p1);
        switch (kind)
        {
        case kShapeCircle:
            if (!(p0 > 0.0f)) { return false; }
            out = MakeCircle(a);
            return true;
        case kShapeBox:
            if (!(p0 > 0.0f) || !(p1 > 0.0f)) { return false; }
            out = MakeAabb(a, b);
            return true;
        case kShapeCapsule:
            if (!(p0 > 0.0f) || !(p1 > 0.0f)) { return false; }
            out = MakeCapsule(a, b);
            return true;
        case kShapeNgon:
        {
            const int sides = static_cast<int>(p1);
            if (!(p0 > 0.0f) || sides < 3 || sides > kMaxNgonSides) { return false; }
            std::vector<Vec2> verts;
            verts.reserve(static_cast<std::size_t>(sides));
            for (int k = 0; k < sides; ++k)
            {
                const Real t = Real(2) * kPi * static_cast<Real>(k) / static_cast<Real>(sides);
                verts.push_back(Vec2(a * std::cos(t), a * std::sin(t)));
            }
            out = MakePolygon(verts);
            return true;
        }
        default:
            return false;
        }
    }

    // A flat JS number array -> a convex polygon. Every turn must have the same
    // sign and a non-trivial magnitude: that refuses collinear, concave and
    // self-intersecting lists along with the too-short ones.
    static bool makeConvex(const emscripten::val& flat, Shape& out)
    {
        const std::vector<float> f = emscripten::vecFromJSArray<float>(flat);
        if (f.size() % 2u != 0u) { return false; }
        const std::size_t n = f.size() / 2u;
        if (n < 3u || n > kMaxSceneVerts) { return false; }
        std::vector<Vec2> verts;
        verts.reserve(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            verts.push_back(Vec2(static_cast<Real>(f[2u * i]), static_cast<Real>(f[2u * i + 1u])));
        }
        int sign = 0;
        for (std::size_t i = 0; i < n; ++i)
        {
            const Vec2& p0 = verts[i];
            const Vec2& p1 = verts[(i + 1u) % n];
            const Vec2& p2 = verts[(i + 2u) % n];
            const Real cross = (p1.x - p0.x) * (p2.y - p1.y) - (p1.y - p0.y) * (p2.x - p1.x);
            if (std::abs(cross) < Real(1e-9)) { return false; }
            const int s = cross > Real(0) ? 1 : -1;
            if (sign == 0) { sign = s; }
            else if (s != sign) { return false; }
        }
        out = MakePolygon(verts);
        return true;
    }

    int addBodyWithShape(int type, float x, float y, float angle, const Shape& shape)
    {
        BodyDef bd;
        bd.type           = static_cast<BodyType>(type);
        bd.position       = Vec2(static_cast<Real>(x), static_cast<Real>(y));
        bd.shape          = shape;
        bd.density        = m_next.density;
        bd.friction       = m_next.friction;
        bd.restitution    = m_next.restitution;
        bd.linearDamping  = m_next.linearDamping;
        bd.fixedRotation  = m_next.fixedRotation;
        bd.categoryBits   = m_next.category;
        bd.groupIndex     = m_next.group;
        bd.maskBits       = m_next.mask;
        bd.sleepThreshold = m_next.sleepThreshold;
        bd.bullet         = m_next.bullet;
        bd.gravityScale   = m_next.gravityScale;
        const BodyHandle h = m_world->AddBody(bd);
        if (angle != 0.0f) { m_world->SetAngle(h, static_cast<Real>(angle)); }
        return static_cast<int>(h.index);
    }

    int addFixtureWithShape(BodyHandle h, float lx, float ly, float langle, const Shape& shape)
    {
        FixtureDef fd;
        fd.shape        = shape;
        fd.localPos     = Vec2(static_cast<Real>(lx), static_cast<Real>(ly));
        fd.localAngle   = static_cast<Real>(langle);
        fd.density      = m_next.density;
        fd.friction     = m_next.friction;
        fd.restitution  = m_next.restitution;
        fd.categoryBits = m_next.category;
        fd.groupIndex   = m_next.group;
        fd.maskBits     = m_next.mask;
        return static_cast<int>(m_world->AddFixture(h, fd).index);
    }

    int addJoint(int a, int b, JointDef d)
    {
        if (!handle(a, d.a) || !handle(b, d.b) || a == b) { return -1; }
        return storeJoint(m_world->AddJoint(d));
    }
    int storeJoint(Joint* j)
    {
        if (j == nullptr) { return -1; }
        // Ids are never reused: a stale id held by JS can never reach a newer joint.
        m_joints.push_back(j);
        return static_cast<int>(m_joints.size() - 1u);
    }
    Joint* joint(int id) const
    {
        if (id < 0 || static_cast<std::size_t>(id) >= m_joints.size()) { return nullptr; }
        return m_joints[static_cast<std::size_t>(id)];
    }
    static DistanceJoint* distanceJoint(Joint* j)
    {
        return j != nullptr ? dynamic_cast<DistanceJoint*>(j) : nullptr;
    }
    // A world point in body h's local frame (relative to its origin).
    Vec2 toLocal(BodyHandle h, float x, float y) const
    {
        const Vec2 p = m_world->Position(h);
        const Real a = m_world->GetAngle(h);
        const Real dx = static_cast<Real>(x) - p.x, dy = static_cast<Real>(y) - p.y;
        const Real c = std::cos(a), sn = std::sin(a);
        return Vec2(c * dx + sn * dy, -sn * dx + c * dy);
    }
    // a plain revolute joint (a weld derives from it but is not one)
    static RevoluteJoint* revolute(Joint* j)
    {
        if (j == nullptr || dynamic_cast<WeldJoint*>(j) != nullptr) { return nullptr; }
        return dynamic_cast<RevoluteJoint*>(j);
    }
    void wakeJoint(const Joint* j)
    {
        if (m_world->IsValid(j->HandleA())) { m_world->Wake(j->HandleA()); }
        if (m_world->IsValid(j->HandleB())) { m_world->Wake(j->HandleB()); }
    }

    void wakeAll(const std::vector<BodyHandle>& hs)
    {
        for (const BodyHandle& p : hs)
        {
            if (m_world->IsValid(p) && m_world->GetType(p) == BodyType::Dynamic) { m_world->Wake(p); }
        }
    }

    void advanceKinematics(Real dt)
    {
        for (Kin& k : m_kin)
        {
            if (k.omega == Real(0)) { continue; }
            k.angle = std::fmod(k.angle + k.omega * dt, Real(2) * kPi);
            m_world->SetAngle(k.h, k.angle);
        }
    }

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
    NextBody                      m_next;     // the scene surface's pending material
    std::vector<Joint*>           m_joints;   // joint id -> borrowed Joint* (nullptr once gone)
    std::vector<Kin>              m_kin;      // kinematic bodies with a binding-advanced angle
    std::vector<float>            m_contactsBuf; // contacts()'s backing store
    std::array<float, 8>          m_rayBuf{};    // castRayClosest()'s backing store
    std::array<float, 4>          m_aabbBuf{};   // bodyAABB()'s backing store
    std::array<float, 3>          m_reactBuf{};  // jointReaction()'s backing store
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
        .function("params",              &ManifoldSim::params)
        // the scene surface
        .function("setMaterial",         &ManifoldSim::setMaterial)
        .function("setFilter",           &ManifoldSim::setFilter)
        .function("setFixedRotation",    &ManifoldSim::setFixedRotation)
        .function("addBody",             &ManifoldSim::addBody)
        .function("addBodyPoly",         &ManifoldSim::addBodyPoly)
        .function("addFixture",          &ManifoldSim::addFixture)
        .function("addFixturePoly",      &ManifoldSim::addFixturePoly)
        .function("removeBody",          &ManifoldSim::removeBody)
        .function("isAlive",             &ManifoldSim::isAlive)
        .function("setTransform",        &ManifoldSim::setTransform)
        .function("setVelocity",         &ManifoldSim::setVelocity)
        .function("applyImpulse",        &ManifoldSim::applyImpulse)
        .function("setKinematic",        &ManifoldSim::setKinematic)
        .function("addDistanceJoint",    &ManifoldSim::addDistanceJoint)
        .function("addDistanceJointAt",  &ManifoldSim::addDistanceJointAt)
        .function("addRevoluteJoint",    &ManifoldSim::addRevoluteJoint)
        .function("addWeldJoint",        &ManifoldSim::addWeldJoint)
        .function("addPrismaticJoint",   &ManifoldSim::addPrismaticJoint)
        .function("addWheelJoint",       &ManifoldSim::addWheelJoint)
        .function("addMotorJoint",       &ManifoldSim::addMotorJoint)
        .function("addMouseJoint",       &ManifoldSim::addMouseJoint)
        .function("setMouseTarget",      &ManifoldSim::setMouseTarget)
        .function("removeJoint",         &ManifoldSim::removeJoint)
        .function("jointCount",          &ManifoldSim::jointCount)
        .function("setJointMotor",       &ManifoldSim::setJointMotor)
        .function("setRevoluteReference", &ManifoldSim::setRevoluteReference)
        .function("setRevoluteLimits",   &ManifoldSim::setRevoluteLimits)
        .function("setRevoluteSpring",   &ManifoldSim::setRevoluteSpring)
        .function("jointAngle",          &ManifoldSim::jointAngle)
        .function("setPrismaticLimits",  &ManifoldSim::setPrismaticLimits)
        .function("jointTranslation",    &ManifoldSim::jointTranslation)
        .function("setDistanceLimits",   &ManifoldSim::setDistanceLimits)
        .function("setDistanceSpring",   &ManifoldSim::setDistanceSpring)
        .function("jointLength",         &ManifoldSim::jointLength)
        .function("jointReaction",       &ManifoldSim::jointReaction)
        .function("castRayClosest",      &ManifoldSim::castRayClosest)
        .function("bodyAABB",            &ManifoldSim::bodyAABB)
        .function("setFilterGroup",      &ManifoldSim::setFilterGroup)
        .function("setBodyOptions",      &ManifoldSim::setBodyOptions)
        .function("wake",                &ManifoldSim::wake)
        .function("isAwake",             &ManifoldSim::isAwake)
        .function("mass",                &ManifoldSim::mass)
        .function("applyLinearImpulse",  &ManifoldSim::applyLinearImpulse)
        .function("setGravityWell",      &ManifoldSim::setGravityWell)
        .function("clearGravityWell",    &ManifoldSim::clearGravityWell)
        .function("setGravityScale",     &ManifoldSim::setGravityScale)
        .function("setBodyFilter",       &ManifoldSim::setBodyFilter)
        .function("setBullet",           &ManifoldSim::setBullet)
        .function("isBullet",            &ManifoldSim::isBullet)
        .function("enableContinuous",    &ManifoldSim::enableContinuous)
        .function("contacts",            &ManifoldSim::contacts);
    // embind gives every class_ a .delete() automatically -- contract section 3's
    // sim.delete() needs no registration.
}
