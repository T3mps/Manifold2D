#pragma once

// Collider shapes for the Manifold2D 2D physics engine (v2, Phase A, Task 1).
//
// UNIFIED CORE+RADIUS MODEL (v2):
//   Every Shape carries a populated convex polygon core (verts + normals +
//   count) plus a scalar radius, for ALL four kinds:
//     Circle(r)         -> 1 core vert at (0,0),       radius = r
//     Capsule(hl, r)    -> 2 core verts (-hl,0),(+hl,0), radius = r
//     Aabb(hw, hh)      -> 4 corner verts (CCW),        radius = 0
//     Polygon(verts)    -> N verts (CCW, baked),         radius = 0
//   All collision ultimately reduces to "distance between two convex polygon
//   cores, inflated by rA + rB" (Box2D v3 model). The ShapeKind tag is
//   retained as a fast-path HINT for circle/capsule segment-distance paths;
//   it is NOT the sole dispatch anymore -- callers may use the unified core.
//
// ROTATION-AWARE (v2):
//   ComputeAABB(xf) now rotates the core verts by xf.rotation, bounds them,
//   then expands by radius. The old "fixedRotation world, xf.rotation ignored"
//   note no longer applies; all shapes compute geometrically-correct
//   world-space AABBs under arbitrary orientation.
//
// ADDITIVE (v2 -> compatible with M6 narrowphase):
//   The existing kind/halfLen/radius/halfW/halfH/legacy-verts/normals/
//   polyCentroid fields are unchanged so the old narrowphase keeps compiling
//   and the full [physics] suite stays green. The new unified core data
//   lives IN the same verts/normals vectors (now populated for all four kinds).
//
// PRESENTATION-FREE + C++20-clean: Geometry::Vec2 + std + sibling Physics headers only.
// No SDL3/NVRHI/Batcher2D/ImGui, no C++23-only features (this module is also
// Arcane-only, /MD. Shapes are immutable
// value geometry -- multiple bodies can share one. Precomputed data (centroid,
// polygon normals) is baked at construction; copyable + cheap to share.

#include <cstdint>
#include <vector>

#include <Manifold2D/Physics/PhysicsTypes.hpp>

namespace Manifold2D
{
    namespace Physics
    {
        // ----------------------------------------------------------------
        // Transform: position + rotation (both load-bearing in v2).
        // ----------------------------------------------------------------
        //
        // v2: rotation is now USED by ComputeAABB (rotates the core verts).
        // Pass the body's actual angle; zero is fine for axis-aligned shapes
        // or when rotation is locked (fixedRotation=true bodies stay at their
        // constant angle, which flows through correctly).
        struct Transform
        {
            Vec2 position{ Real(0), Real(0) };
            // Rotation in radians. Load-bearing in v2: ComputeAABB rotates
            // the core verts by this angle before bounding.
            Real rotation = Real(0);
        };

        // ----------------------------------------------------------------
        // AABB: world-space axis-aligned bounds (min/max corners).
        // ----------------------------------------------------------------
        //
        // Matches the Lua aabbOf return tuple (x0,y0,x1,y1) -> (min, max).
        struct Aabb
        {
            Vec2 min{ Real(0), Real(0) };
            Vec2 max{ Real(0), Real(0) };
        };

        // ----------------------------------------------------------------
        // MassData: result of ComputeMass(density).
        // ----------------------------------------------------------------
        //
        // Box2D-style mass properties. `inertia` is the rotational inertia
        // about the shape's centroid (NOT about the local origin); the solver
        // re-bases it onto the body center when bodies are built in Phase 2.
        struct MassData
        {
            Real mass = Real(0);                  // density * area
            Vec2 centroid{ Real(0), Real(0) };    // local-space area centroid
            Real inertia = Real(0);               // about the centroid
        };

        // ----------------------------------------------------------------
        // ShapeKind: tagged-union discriminator (Lua used a `kind` string).
        // ----------------------------------------------------------------
        enum class ShapeKind : std::uint8_t
        {
            Circle  = 0,
            Capsule = 1,
            Aabb    = 2,
            Polygon = 3,
        };

        // ----------------------------------------------------------------
        // Shape: immutable collider geometry, copyable + shareable.
        // ----------------------------------------------------------------
        //
        // A single struct carries all four kinds (data-oriented; matches the
        // Lua `kind`-tagged table). Use the named factory functions
        // (MakeCircle/MakeCapsule/MakeAabb/MakePolygon) to construct -- they
        // bake precomputed polygon data (CCW verts, outward normals, centroid)
        // so the SAT port in P1.2 reads it directly.
        struct Shape
        {
            ShapeKind kind = ShapeKind::Circle;

            // Circle / Capsule radius.
            Real radius = Real(0);
            // Capsule half-length: segment is (-halfLen,0)-(+halfLen,0).
            Real halfLen = Real(0);
            // AABB half-extents.
            Real halfW = Real(0);
            Real halfH = Real(0);

            // Unified convex polygon core. Populated for ALL four kinds (v2):
            //   Circle  -> 1 vert at (0,0), no normals (radius encodes shape)
            //   Capsule -> 2 verts at (-halfLen,0) and (+halfLen,0), no normals
            //   Aabb    -> 4 corner verts CCW, 4 outward edge normals
            //   Polygon -> N verts CCW, N outward edge normals
            // verts.size() == the core vertex count; for circle/capsule the
            // radius field carries the round-inflation. The old narrowphase
            // reads these fields directly (additive -- still correct).
            std::vector<Vec2> verts;
            std::vector<Vec2> normals;
            // Area centroid of the core (local space). Baked at construction.
            // For circle/capsule: (0,0) by symmetry. Replaces polyCentroid
            // (the alias is kept for existing narrowphase callers).
            Vec2 polyCentroid{ Real(0), Real(0) };

            // World AABB of this shape under transform xf. v2: rotation-aware.
            // Rotates the unified core verts by xf.rotation, bounds them, then
            // expands by radius. Works correctly for all four shape kinds.
            Aabb ComputeAABB(const Transform& xf) const;

            // Standard Box2D-style mass properties for the given density.
            // NEW computation (no Lua oracle): validated against hand-derived
            // analytic values in the test. See Shapes.cpp for the per-kind
            // derivations.
            MassData ComputeMass(Real density) const;
        };

        // ----------------------------------------------------------------
        // Factory functions (port of shapes.circle/capsule/aabb/polygon).
        // ----------------------------------------------------------------

        // Circle of radius r. PORT: shapes.circle(r).
        [[nodiscard]] Shape MakeCircle(Real r);

        // Horizontal capsule: segment (-halfLen,0)-(+halfLen,0) inflated by r.
        // PORT: shapes.capsule(halfLen, r).
        [[nodiscard]] Shape MakeCapsule(Real halfLen, Real r);

        // Axis-aligned box by half-extents. PORT: shapes.aabb(hw, hh).
        [[nodiscard]] Shape MakeAabb(Real hw, Real hh);

        // Convex polygon from a flat {x0,y0,x1,y1,...} vertex list in local
        // space. PORT: shapes.polygon(verts) -- requires >= 3 (x,y) verts and
        // <= kMaxPolyVerts; throws std::invalid_argument otherwise (the Lua
        // used assert). The input may be CW or CCW: the polygon is normalized
        // to CCW winding and outward unit edge normals + the area centroid are
        // baked. Geometry.polyPoly was winding-agnostic, but the C++ Polygon
        // stores CCW + precomputed normals for the SAT port (P1.2).
        [[nodiscard]] Shape MakePolygon(const std::vector<Vec2>& verts);

    } // namespace Physics
} // namespace Manifold2D
