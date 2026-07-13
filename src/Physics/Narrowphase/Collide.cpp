// Collide.cpp -- unified rotation-aware narrowphase entry (Physics v2, Task 3).
//
// See Collide.hpp for the full contract and algorithm description.
//
// Reference: Box2D v3 manifold.c (b2CollidePolygons, b2ClipSegments) +
//            the existing M6 Sat.cpp / Gjk.cpp for the local style.
//
// PRESENTATION-FREE + C++20-clean: Geometry::Vec2 + std + sibling Physics only.

#include <Manifold2D/Physics/Narrowphase/Collide.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

#include <Manifold2D/Physics/Narrowphase/Epa.hpp>
#include <Manifold2D/Physics/Narrowphase/Gjk.hpp>
#include <Manifold2D/Physics/Narrowphase/Mpr.hpp>
#include <Manifold2D/Physics/Narrowphase/NarrowphaseTrace.hpp>
#include <Manifold2D/Physics/PhysicsTypes.hpp>
#include <Manifold2D/Physics/Shapes.hpp>

namespace Manifold2D
{
    namespace Physics
    {
        namespace
        {
            // ----------------------------------------------------------------
            // RotateInto: rotate + translate a shape's unified core verts
            // (Shape::verts, local space) into world space under xf.
            // xf.rotation is the angle in radians (cos/sin precomputed below).
            // Writes into `out` (caller provides buffer of at least verts.size()).
            // Returns the vertex count.
            // ----------------------------------------------------------------
            int RotateInto(Vec2* out, const Shape& s, const Transform& xf)
            {
                const float c = std::cos(xf.rotation);
                const float si = std::sin(xf.rotation);
                const float px = xf.position.x;
                const float py = xf.position.y;
                const int n = static_cast<int>(s.verts.size());
                for (int i = 0; i < n; ++i)
                {
                    const float lx = s.verts[i].x;
                    const float ly = s.verts[i].y;
                    out[i] = Vec2(lx * c - ly * si + px,
                                  lx * si + ly * c  + py);
                }
                return n;
            }

            // ----------------------------------------------------------------
            // Stable feature-id packing.
            //
            // Bits: [31]    = refIsA  (1 = reference face is on shape A)
            //       [30:16] = refEdgeIdx (15 bits)
            //       [15: 0] = incFeature (16 bits; vertex index or GJK feature)
            //
            // This is invariant to the depth of each frame's overlap: the same
            // physical feature pair always produces the same 32-bit id, enabling
            // the solver's warm-start cache to find the right impulse.
            // ----------------------------------------------------------------
            constexpr uint32_t MakeFeatureId(bool refIsA,
                                             uint32_t refEdge,
                                             uint32_t incFeature) noexcept
            {
                return (refIsA  ? 0x8000'0000u : 0u)
                     | ((refEdge   & 0x7FFFu) << 16)
                     |  (incFeature & 0xFFFFu);
            }

            // ----------------------------------------------------------------
            // Classify a SHALLOW round-path contact for the debug-viz tag.
            //
            // The round cell handles circle (1 core vert) and capsule (2 core
            // verts) pairs (against each other or a polygon). DISPLAY-ONLY:
            //   * either core a 2-vert segment -> Capsule
            //   * both cores single points     -> CircleCircle
            //   * otherwise (one circle core vs a >=3-vert polygon) -> CircleVsPolygon
            // The deep branches override this with Epa / Mpr at their own return.
            // ----------------------------------------------------------------
            constexpr NarrowphaseKind RoundKind(int na, int nb) noexcept
            {
                if (na == 2 || nb == 2) return NarrowphaseKind::Capsule;
                if (na == 1 && nb == 1) return NarrowphaseKind::CircleCircle;
                return NarrowphaseKind::CircleVsPolygon;
            }

            // ----------------------------------------------------------------
            // SAT helper: find the minimum-separation axis for `va` (na verts)
            // against `vb` (nb verts), testing the CCW edge normals of `vsRef`.
            //
            // Returns (edgeIndex, separation) where separation > 0 means
            // separated (no overlap along this axis), separation < 0 means
            // the ref polygon's edge normal axis has that much penetration.
            //
            // For each CCW edge normal n of vsRef, the minimum support of vb
            // along -n gives the separation on that axis.  We pick the axis
            // with the LEAST penetration (maximum separation, i.e. closest to
            // touching).  Tie-break: lowest edge index wins (deterministic).
            //
            // When the function returns with separation > 0 the shapes are
            // already separated -- caller should not proceed to clipping.
            // ----------------------------------------------------------------
            struct SatAxis
            {
                int  edgeIdx   = 0;
                Real separation = Real(-1e30);
                // The outward edge normal on vsRef (world space, unit).
                Vec2 normal{ Real(0), Real(0) };
            };

            // Compute the CCW outward edge normal of edge i->(i+1) in `va`.
            // For CCW winding the outward normal is (dy, -dx) where d = v[i+1]-v[i].
            // (In y-down screen space CCW means shoelace positive with y increasing
            // downward; the outward normal for edge (a->b) is (b.y-a.y, a.x-b.x).)
            inline Vec2 EdgeNormalCCW(const Vec2* va, int na, int i)
            {
                const Vec2& v0 = va[i];
                const Vec2& v1 = va[(i + 1) % na];
                const float dx = v1.x - v0.x;
                const float dy = v1.y - v0.y;
                // Outward normal (CCW poly, y-down): (dy, -dx), then normalise.
                const float len = std::sqrt(dx * dx + dy * dy);
                if (len < Real(1e-9f)) return Vec2(Real(1), Real(0));
                return Vec2(dy / len, -dx / len);
            }

            // Signed support of `verts` along axis `n` (min projection).
            inline Real MinSupport(const Vec2* verts, int n, const Vec2& axis)
            {
                Real minProj = std::numeric_limits<Real>::infinity();
                for (int i = 0; i < n; ++i)
                {
                    const Real p = verts[i].x * axis.x + verts[i].y * axis.y;
                    if (p < minProj) minProj = p;
                }
                return minProj;
            }

            // MaxSupport of `verts` along axis `n` (max projection).
            inline Real MaxSupport(const Vec2* verts, int n, const Vec2& axis)
            {
                Real maxProj = -std::numeric_limits<Real>::infinity();
                for (int i = 0; i < n; ++i)
                {
                    const Real p = verts[i].x * axis.x + verts[i].y * axis.y;
                    if (p > maxProj) maxProj = p;
                }
                return maxProj;
            }

            // Find the SAT axis (most-penetrating = least-negative separation)
            // by testing all CCW edge normals of `vsRef` against `vsInc`.
            // Returns the edge index and separation (negative = penetration).
            // If separation >= 0 the shapes are separated along this polygon.
            //
            // `trace` (debug-viz, OPT-IN): when trace is NON-null, each
            // candidate axis is appended to trace->satAxes with its
            // reference/incident projection intervals (minRef/maxRef = this
            // call's REFERENCE shape `vsRef`, minInc/maxInc = the INCIDENT shape
            // `vsInc`). Recording reads already-computed support projections only
            // -- the selection math (`best`) is unchanged whether trace is null
            // or not, so the returned axis is byte-identical on the Step path.
            SatAxis FindReferenceAxis(const Vec2* vsRef, int nRef,
                                      const Vec2* vsInc, int nInc,
                                      NarrowphaseTrace* trace = nullptr)
            {
                SatAxis best{};
                best.separation = -std::numeric_limits<Real>::infinity();
                best.edgeIdx    = 0;

                for (int i = 0; i < nRef; ++i)
                {
                    const Vec2 n = EdgeNormalCCW(vsRef, nRef, i);

                    // Support of vsRef along n (max projection from vsRef).
                    const Real maxRef = MaxSupport(vsRef, nRef, n);
                    // Support of vsInc along -n (min projection into vsRef).
                    const Real minInc = MinSupport(vsInc, nInc, n);
                    // Separation = minInc - maxRef.
                    // > 0 : separated.   < 0 : penetrating.
                    const Real sep = minInc - maxRef;

                    // Debug-viz recording (OPT-IN; nullptr on the Step path).
                    // Append this candidate axis with both shapes' projection
                    // intervals along n. minRef/maxRef bound the reference
                    // shape; minInc bounds the incident shape -- compute its
                    // maxInc too purely for display (no effect on selection).
                    if (trace)
                    {
                        const Real minRef = MinSupport(vsRef, nRef, n);
                        const Real maxInc = MaxSupport(vsInc, nInc, n);
                        // NOTE: the trace's ::Manifold2D::Physics::SatAxis differs
                        // from this TU's anonymous-namespace SatAxis (which the
                        // unqualified name resolves to); qualify it explicitly.
                        ::Manifold2D::Physics::SatAxis ax;
                        ax.dir = n;
                        ax.minRef = minRef;
                        ax.maxRef = maxRef;
                        ax.minInc = minInc;
                        ax.maxInc = maxInc;
                        ax.chosen = false; // marked after the search resolves
                        trace->satAxes.push_back(ax);
                    }

                    // Keep the axis with the LEAST penetration (highest sep).
                    // Tie-break: lower edge index wins (deterministic, per spec).
                    if (sep > best.separation)
                    {
                        best.separation = sep;
                        best.edgeIdx    = i;
                        best.normal     = n;
                    }
                }
                return best;
            }

            // ----------------------------------------------------------------
            // ClipSegment: clip segment (p0, p1) against the half-plane
            // defined by the plane normal `n` and offset `offset`:
            //   n . x <= offset  (keep side)
            // Writes into `outP`, `outT` (parametric t) up to 2 results.
            // Returns the number of points kept (0, 1, or 2).
            // This is the Box2D v3 b2ClipSegments half-plane logic.
            // ----------------------------------------------------------------
            int ClipSegment(const Vec2& p0, const Vec2& p1,
                            const Vec2& n, Real offset,
                            Vec2 outP[2], float outT[2])
            {
                const Real d0 = p0.x * n.x + p0.y * n.y - offset;
                const Real d1 = p1.x * n.x + p1.y * n.y - offset;
                int count = 0;

                if (d0 <= Real(0))
                {
                    outP[count] = p0;
                    outT[count] = Real(0);
                    ++count;
                }
                if (d1 <= Real(0))
                {
                    outP[count] = p1;
                    outT[count] = Real(1);
                    ++count;
                }
                // If the segment crosses the plane, emit the crossing point.
                // Guard: only emit if count < 2 to prevent a third write when an
                // endpoint lies exactly ON the plane (d==0): both the keep-branch
                // and this branch would fire, overrunning the 2-element output
                // array.  Box2D v3 b2ClipSegments carries the same guard.
                if (count < 2 && (d0 < Real(0)) != (d1 < Real(0)))
                {
                    const Real t = d0 / (d0 - d1);
                    outP[count] = Vec2(p0.x + t * (p1.x - p0.x),
                                       p0.y + t * (p1.y - p0.y));
                    outT[count] = t;
                    ++count;
                }
                return count;
            }

            // ----------------------------------------------------------------
            // Segment(capsule)-vs-polygon TWO-POINT face clip.
            //
            // The round path's GJK-witness branch emits a SINGLE contact point.
            // That is correct for a circle (1-vert core) but WRONG for a capsule
            // (2-vert SEGMENT core) lying roughly PARALLEL to a polygon face: a
            // capsule resting flat on a floor is an edge-edge contact that needs
            // TWO points (one near each segment end) to be rotationally stable.
            // With one point the contact location jumps end-to-end as the body
            // tips, the lone point torques the body the other way, and it ROCKS
            // in a never-damped limit cycle -- it never settles and never sleeps
            // (the bug this function fixes).
            //
            // Box2D v3 treats a capsule as a 2-vertex rounded polygon and clips
            // it against the reference face (b2CollidePolygons / b2CollideSegment-
            // AndPolygon). We do the same for the one common, important case:
            // exactly one core is a 2-vert segment, the other is a >=3-vert
            // polygon, AND the segment is roughly parallel to the contacting
            // face (the edge-edge case). Otherwise we return pointCount==0 and
            // the caller keeps its single-point witness path (circle-vs-poly,
            // capsule-vs-capsule, capsule-vs-circle, or an end-cap contact where
            // a single point is correct).
            //
            // `n` is the contact normal (B->A, unit). The reference face is on
            // the POLYGON; the SEGMENT is the incident feature we clip. Each
            // surviving clipped point's separation is measured along `n` and the
            // surface point is offset inward by the radii. The returned manifold
            // matches the surface convention of the witness path (separation > 0
            // = penetration; midpoint of the two surface witnesses per point).
            //
            // Returns a manifold with pointCount 2 on success, else pointCount 0.
            // ----------------------------------------------------------------
            Manifold CollideSegmentPolyTwoPoint(const Vec2* va, int na, Real rA,
                                                const Vec2* vb, int nb, Real rB,
                                                const Vec2& n, uint32_t baseId)
            {
                Manifold m{};

                // Exactly one core must be a 2-vert segment, the other a polygon.
                const bool aSeg = (na == 2 && nb >= 3);
                const bool bSeg = (nb == 2 && na >= 3);
                if (!aSeg && !bSeg)
                {
                    return m; // not the capsule-vs-polygon case
                }

                // Identify segment (seg/rSeg) and polygon (poly/rPoly). The
                // returned normal stays B->A regardless of which is the segment.
                const Vec2* seg  = aSeg ? va : vb;
                const Real  rSeg = aSeg ? rA : rB;
                const Vec2* poly = aSeg ? vb : va;
                const int   nPoly = aSeg ? nb : na;
                const Real  rPoly = aSeg ? rB : rA;

                // Axis from the POLYGON toward the SEGMENT (so the reference face
                // is the polygon edge facing the segment). With n = B->A:
                //   - A is the segment (aSeg): segment is on the +n side -> axis = n.
                //   - B is the segment (bSeg): segment is on the -n side -> axis = -n.
                const Vec2 axis = aSeg ? n : Vec2(-n.x, -n.y);

                // Reference face: the polygon edge whose outward normal is most
                // aligned with `axis`. Require near-parallel (dot close to 1) so
                // we only take the edge-edge case; an end-cap / corner contact
                // (segment endpoint vs polygon vertex) is left to the 1-point
                // path where a single witness is correct.
                int   refEdge = -1;
                Real  bestDot = Real(-1e30);
                Vec2  refNormal{ Real(0), Real(0) };
                for (int i = 0; i < nPoly; ++i)
                {
                    const Vec2 en = EdgeNormalCCW(poly, nPoly, i);
                    const Real d = en.x * axis.x + en.y * axis.y;
                    if (d > bestDot)
                    {
                        bestDot   = d;
                        refEdge   = i;
                        refNormal = en;
                    }
                }
                // Parallel gate part 1: the reference FACE normal must be roughly
                // parallel to the contact axis (the face actually faces the
                // segment). cos 0.94 ~ 20 deg of tip tolerance.
                if (refEdge < 0 || bestDot < Real(0.94))
                {
                    return m;
                }

                const Vec2 refV0 = poly[refEdge];
                const Vec2 refV1 = poly[(refEdge + 1) % nPoly];

                // Parallel gate part 2: the SEGMENT itself must lie roughly ALONG
                // the reference face (segment direction nearly perpendicular to
                // the face normal). A capsule TILTED into a face/corner (e.g. a
                // 30 deg jam) is NOT an edge-edge rest -- its two endpoints sit at
                // very different depths and it pivots on the deeper one, which the
                // single EPA/witness point models correctly. Only the near-parallel
                // resting capsule (the rocking-bug case) needs two points.
                // |dot(segDir, refNormal)| <= sin(20 deg) ~ 0.342 keeps the same
                // ~20 deg tolerance as part 1.
                Vec2 segDir(seg[1].x - seg[0].x, seg[1].y - seg[0].y);
                const Real segLen = std::sqrt(segDir.x * segDir.x +
                                              segDir.y * segDir.y);
                if (segLen < Real(1e-9f))
                {
                    return m; // degenerate segment (acts like a circle) -> 1 point
                }
                segDir = Vec2(segDir.x / segLen, segDir.y / segLen);
                const Real segDotN = std::abs(segDir.x * refNormal.x +
                                              segDir.y * refNormal.y);
                if (segDotN > Real(0.342))
                {
                    return m; // segment tilted off the face -> 1-point path
                }

                // Clip the segment against the reference face's two side planes
                // (the face tangent bounds). This keeps the part of the segment
                // that overlaps the face span -> up to 2 points.
                Vec2 tangent = Vec2(refV1.x - refV0.x, refV1.y - refV0.y);
                const Real tlen = std::sqrt(tangent.x * tangent.x +
                                            tangent.y * tangent.y);
                if (tlen < Real(1e-9f))
                {
                    return m; // degenerate face
                }
                tangent = Vec2(tangent.x / tlen, tangent.y / tlen);

                // Side plane 1: keep dot(p, -tangent) <= dot(refV0, -tangent).
                const Vec2 negT(-tangent.x, -tangent.y);
                const Real off1 = refV0.x * negT.x + refV0.y * negT.y;
                Vec2  clip1[2];
                float t1[2]{};
                const int nc1 = ClipSegment(seg[0], seg[1], negT, off1, clip1, t1);
                if (nc1 < 2)
                {
                    return m; // segment does not span the face -> 1-point path
                }
                // Side plane 2: keep dot(p, tangent) <= dot(refV1, tangent).
                const Real off2 = refV1.x * tangent.x + refV1.y * tangent.y;
                Vec2  clip2[2];
                float t2[2]{};
                const int nc2 = ClipSegment(clip1[0], clip1[1], tangent, off2,
                                            clip2, t2);
                if (nc2 < 2)
                {
                    return m; // less than two survivors -> 1-point path
                }

                // Reference face plane offset along its outward normal.
                const Real refOff = refV0.x * refNormal.x + refV0.y * refNormal.y;
                const Real totalR = rSeg + rPoly;

                // Build the two contact points. For each clipped SEGMENT point:
                //   coreGap = dot(clipPoint, refNormal) - refOff   (>=0 outside)
                //   separation = totalR - coreGap                  (>0 = penetrate)
                // The contact point sits on the mid-surface: push the clipped
                // segment point toward the face by (rSeg - 0.5*(coreGap + ... )).
                // We keep it simple and correct: surfaceSeg = clip - axis*rSeg
                // (segment surface toward the poly), surfacePoly = projection of
                // clip onto the face plane + refNormal*rPoly (poly surface toward
                // the segment); contact = midpoint.
                m.normal = n;
                m.pointCount = 0;
                m.kind = NarrowphaseKind::Capsule; // segment(capsule)-vs-polygon
                for (int i = 0; i < 2; ++i)
                {
                    const Vec2 cpt = clip2[i];
                    const Real coreGap =
                        (cpt.x * refNormal.x + cpt.y * refNormal.y) - refOff;
                    const Real separation = totalR - coreGap;

                    // Surface witnesses (axis = polygon->segment outward dir):
                    //   segment surface = clipPoint - axis*rSeg
                    //   poly    surface = clipPoint - axis*coreGap + axis*rPoly
                    // (project clip onto the face along -axis by coreGap, then
                    // push out by rPoly). midpoint is the manifold point.
                    const Vec2 segSurf(cpt.x - axis.x * rSeg,
                                       cpt.y - axis.y * rSeg);
                    const Vec2 polySurf(cpt.x - axis.x * (coreGap - rPoly),
                                        cpt.y - axis.y * (coreGap - rPoly));
                    const Vec2 contact((segSurf.x + polySurf.x) * Real(0.5f),
                                       (segSurf.y + polySurf.y) * Real(0.5f));

                    // Stable per-point id: mix the base round id with the segment
                    // endpoint index (0/1) so the two points get DISTINCT, stable
                    // warm-start keys that do not alias the 1-point round id.
                    const uint32_t pid = baseId ^ (0x9E3779B9u * (static_cast<uint32_t>(i) + 1u));

                    ManifoldPoint mp{};
                    mp.point      = contact;
                    mp.separation = separation;
                    mp.normal     = n;
                    mp.id         = pid;
                    m.points[m.pointCount++] = mp;
                }
                return m;
            }

            // ----------------------------------------------------------------
            // Round-shape fast path (circle / capsule).
            //
            // When at least one core is 1 or 2 verts (circle or capsule), the
            // GJK witness pair IS the contact (no face-clip needed).  After
            // computing via GjkDistanceCore:
            //   - If coreDist >= rA+rB  -> separated or speculative.
            //   - If coreDist == 0      -> deep overlap: EXACT nearest-face axis
            //     via EPA, with MPR (independent seed) as the convergence
            //     fallback, and an empty manifold if both fail (degenerate input;
            //     no contact this frame beats a wrong normal). This replaced the
            //     old centroid-to-centroid approximation.
            //   - Otherwise             -> coreDist < rA+rB -> penetration via
            //     the witness pair.
            // Returns a 1-point manifold (for circle-vs-* the cores are both
            // single points or short segments; the closest feature pair is the
            // contact).
            //
            // Feature id for the speculative/round path: pack featureA (from A's
            // core) and featureB (from B's core) from GjkDistanceCore directly
            // into the stable id scheme (refIsA=true, refEdge=featureA & 0x7FFF,
            // incFeature=featureB & 0xFFFF). This gives a stable key as long as
            // the same core features are in contact, independent of depth.
            // ----------------------------------------------------------------
            Manifold CollideRound(const Vec2* va, int na, Real rA,
                                  const Vec2* vb, int nb, Real rB,
                                  Real speculativeMargin,
                                  NarrowphaseTrace* trace = nullptr)
            {
                Manifold m{};
                const GjkCoreResult g = GjkDistanceCore(va, na, vb, nb);

                // Build a stable id from the GJK feature pair.
                // We use refIsA=true, encode featureA in bits [30:16] and
                // featureB in bits [15:0].
                const uint32_t id = MakeFeatureId(
                    true,
                    g.featureA & 0x7FFFu,
                    g.featureB & 0xFFFFu);

                const Real totalR = rA + rB;

                // Separated beyond speculative margin: no contact.
                // Use strict ">" so coreDist == totalR (surface touch) still
                // generates a contact.
                if (g.distance > totalR + speculativeMargin)
                {
                    return m;
                }

                Vec2 normal{ Real(-1), Real(0) }; // default fallback
                Vec2 ptA = g.pointA;
                Vec2 ptB = g.pointB;

                if (g.distance > Real(1e-6f))
                {
                    // SHALLOW (separated / speculative): the GJK witness pair IS
                    // the contact. Normal from B's witness toward A's witness
                    // (B->A convention). Surface witnesses + separation are
                    // computed by the shared code below.
                    const Real inv = Real(1) / g.distance;
                    normal = Vec2((g.pointA.x - g.pointB.x) * inv,
                                  (g.pointA.y - g.pointB.y) * inv);

                    // Capsule(segment)-vs-polygon edge-edge: emit TWO points so a
                    // capsule resting parallel to a face is rotationally stable
                    // (a single witness point lets it rock forever). Falls back to
                    // the 1-point witness path below when not the parallel case.
                    const Manifold two = CollideSegmentPolyTwoPoint(
                        va, na, rA, vb, nb, rB, normal, id);
                    if (two.pointCount == 2)
                    {
                        return two;
                    }
                }
                else
                {
                    // Debug-viz: record the TERMINAL GJK origin-enclosing
                    // simplex (EPA's seed) as one GJK snapshot in Minkowski
                    // space. This is the same GjkOriginSimplex EPA itself seeds
                    // from; recording it here is an OPT-IN read (the extra call
                    // runs ONLY when trace is non-null, i.e. never on the Step
                    // path) so the computed manifold is unchanged. Per-iteration
                    // GJK simplices are not recorded -- the GJK loops live in a
                    // separate TU and instrumenting them risks the byte-identical
                    // Step path; the terminal simplex is the inspector-relevant
                    // one for the deep round path.
                    if (trace)
                    {
                        const GjkSimplex seed = GjkOriginSimplex(va, na, vb, nb);
                        SimplexSnapshot snap;
                        snap.count = seed.count;
                        for (int i = 0; i < seed.count && i < 3; ++i)
                        {
                            snap.verts[i] = seed.v[i].md;
                        }
                        snap.searchDir      = Vec2(Real(0), Real(0));
                        snap.containsOrigin = seed.enclosesOrigin;
                        trace->gjkSnapshots.push_back(snap);
                    }

                    // DEEP (cores overlap, GJK distance ~0): the GJK witness pair
                    // is meaningless here, so we call EPA on the SAME world cores
                    // (radii NOT applied) for the EXACT nearest-face penetration.
                    // This replaces the old centroid-to-centroid approximation,
                    // which pointed along the line of centres rather than toward
                    // the nearest face (wrong for a round core buried in a poly).
                    const EpaResult epa = Epa(va, na, vb, nb, trace);

                    if (epa.ok)
                    {
                        // Capsule(segment)-vs-polygon edge-edge, deep: emit TWO
                        // points (same rationale as the shallow branch) using the
                        // EPA normal as the contact axis. Falls back to the
                        // single EPA point below when not the parallel case.
                        const Manifold two = CollideSegmentPolyTwoPoint(
                            va, na, rA, vb, nb, rB, epa.normal, id);
                        if (two.pointCount == 2)
                        {
                            return two;
                        }

                        // epa.normal is the unit B->A axis; epa.depth is the CORE
                        // penetration (radii NOT applied). Surface witnesses use
                        // the SAME offset convention as the shallow branch below:
                        //   surfaceA = witnessA - normal*rA  (A's surface toward B)
                        //   surfaceB = witnessB + normal*rB  (B's surface toward A)
                        const Vec2 n = epa.normal;
                        const Vec2 surfaceA = Vec2(epa.witnessA.x - n.x * rA,
                                                   epa.witnessA.y - n.y * rA);
                        const Vec2 surfaceB = Vec2(epa.witnessB.x + n.x * rB,
                                                   epa.witnessB.y + n.y * rB);
                        const Vec2 cp = Vec2((surfaceA.x + surfaceB.x) * Real(0.5f),
                                             (surfaceA.y + surfaceB.y) * Real(0.5f));

                        // Surface penetration = core depth + both radii
                        // (positive = penetrating, matching the manifold
                        // convention and the shallow branch's totalR - coreDist
                        // with coreDist == 0).
                        const Real separation = epa.depth + rA + rB;

                        m.normal = n;
                        m.points[0] = ManifoldPoint{ cp, separation, n, id };
                        m.pointCount = 1;
                        m.kind = NarrowphaseKind::Epa; // deep round via EPA
                        return m;
                    }

                    // T4: EPA failed to converge (rare, degenerate input).
                    // Fall back to MPR -- a fast single-point deep-overlap
                    // solver -- and build the SAME 1-point manifold the EPA path
                    // builds (surface offsets by rA/rB, separation = depth + rA +
                    // rB, same id scheme). The old centroid approximation is
                    // DELETED: it pointed along the line of centres rather than
                    // toward the nearest face, which is wrong for a round core
                    // buried in a poly (the very case EPA/MPR exist to fix).
                    const MprResult mpr = Mpr(va, na, vb, nb, trace);
                    if (mpr.ok)
                    {
                        const Vec2 n = mpr.normal;
                        const Vec2 surfaceA = Vec2(mpr.point.x - n.x * rA,
                                                   mpr.point.y - n.y * rA);
                        const Vec2 surfaceB = Vec2(mpr.point.x + n.x * rB,
                                                   mpr.point.y + n.y * rB);
                        const Vec2 cp = Vec2((surfaceA.x + surfaceB.x) * Real(0.5f),
                                             (surfaceA.y + surfaceB.y) * Real(0.5f));
                        const Real separation = mpr.depth + rA + rB;

                        m.normal = n;
                        m.points[0] = ManifoldPoint{ cp, separation, n, id };
                        m.pointCount = 1;
                        m.kind = NarrowphaseKind::Mpr; // deep round via MPR fallback
                        return m;
                    }

                    // BOTH EPA and MPR failed (extremely rare). A non-converging
                    // deep overlap is a degenerate input; emitting NO contact for
                    // one frame is safer than guessing a wrong normal (the old
                    // centroid stopgap). Return the empty manifold (pointCount=0).
                    return m;
                }

                // Surface witnesses: offset the core witnesses outward by their radii.
                // surfaceA = ptA - normal*rA  (A's surface toward B)
                // surfaceB = ptB + normal*rB  (B's surface toward A)
                const Vec2 surfaceA = Vec2(ptA.x - normal.x * rA,
                                           ptA.y - normal.y * rA);
                const Vec2 surfaceB = Vec2(ptB.x + normal.x * rB,
                                           ptB.y + normal.y * rB);

                // Contact point: midpoint of the two surface witnesses.
                const Vec2 cp = Vec2((surfaceA.x + surfaceB.x) * Real(0.5f),
                                     (surfaceA.y + surfaceB.y) * Real(0.5f));

                const Real separation = totalR - g.distance; // positive = penetration

                // DISPLAY-ONLY tag for the shallow round path (circle/capsule
                // witness contact). The capsule TWO-point paths returned earlier
                // already tagged Capsule; this 1-point tail covers circle-circle,
                // circle-vs-poly, and the capsule end-cap / tilted cases.
                m.kind = RoundKind(na, nb);

                if (g.distance > totalR)
                {
                    // Speculative gap: coreDist > totalR but within margin.
                    // separation = totalR - coreDist (negative = gap).
                    m.normal = normal;
                    m.points[0] = ManifoldPoint{ cp, separation, normal, id };
                    m.pointCount = 1;
                }
                else
                {
                    // Real penetration (coreDist <= totalR).
                    m.normal = normal;
                    m.points[0] = ManifoldPoint{ cp, separation, normal, id };
                    m.pointCount = 1;
                }

                return m;
            }

            // ----------------------------------------------------------------
            // Poly-cell helper (both cores >= 3 verts: polygon / AABB).
            //
            // GJK tests separation / speculative; on deep overlap SAT finds the
            // minimum-penetration reference face and clips the incident edge ->
            // up to 2 contact points. This cell is NOT routed to EPA -- the SAT
            // reference-face MTV is already exact for two convex polygons, and
            // EPA's single-point output would discard the second clip point the
            // 2-point poly solver needs. (Round cells use EPA for deep overlap
            // because the witness pair is a single point anyway.)
            //
            // Operates on the world cores `va`/`vb` already rotated by the
            // dispatcher (radii via rA/rB). Behaviorally identical to the old
            // inline poly path -- only the function boundary is new.
            // ----------------------------------------------------------------
            Manifold CollidePoly(const Vec2* va, int na, Real rA,
                                 const Vec2* vb, int nb, Real rB,
                                 Real speculativeMargin,
                                 NarrowphaseTrace* trace = nullptr)
            {
            Manifold m{};

            const GjkCoreResult g = GjkDistanceCore(va, na, vb, nb);
            const Real totalR = rA + rB;

            if (g.distance > totalR + speculativeMargin)
            {
                // Fully separated beyond the speculative margin -- no contact.
                return m;
            }

            if (g.distance > totalR)
            {
                // Speculative gap: surface distance in (0, margin].
                // Emit one speculative contact point.
                const Real dist = g.distance;
                Vec2 normal{ Real(-1), Real(0) };
                if (dist > Real(1e-6f))
                {
                    const Real inv = Real(1) / dist;
                    normal = Vec2((g.pointA.x - g.pointB.x) * inv,
                                  (g.pointA.y - g.pointB.y) * inv);
                }
                const Vec2 cp = Vec2((g.pointA.x + g.pointB.x) * Real(0.5f),
                                     (g.pointA.y + g.pointB.y) * Real(0.5f));

                const uint32_t id = MakeFeatureId(
                    true,
                    g.featureA & 0x7FFFu,
                    g.featureB & 0xFFFFu);

                m.normal = normal;
                m.points[0] = ManifoldPoint{ cp, totalR - dist, normal, id };
                m.pointCount = 1;
                m.kind = NarrowphaseKind::SatPolygon; // poly-poly speculative gap
                return m;
            }

            // ----------------------------------------------------------------
            // Cores overlap (GJK returned distance 0). Use SAT to find the
            // reference face (minimum-penetration axis), then clip the incident
            // edge to produce up to 2 contact points.
            //
            // We test all edge normals of both polygons and take the axis that
            // produces the LEAST penetration (Box2D v3 style).
            // ----------------------------------------------------------------

            // Debug-viz: the satAxes vector accumulates A's axes first then
            // B's axes (the two FindReferenceAxis calls below append in that
            // order). Remember the count before each call so we can mark the
            // chosen axis (by reference-shape edge index) once the SAT search
            // resolves. Reads/writes only the trace -- no effect on selection.
            const std::size_t satBaseA = trace ? trace->satAxes.size() : 0;
            // Test A's edge normals as reference axes against B.
            const SatAxis axisA = FindReferenceAxis(va, na, vb, nb, trace);
            const std::size_t satBaseB = trace ? trace->satAxes.size() : 0;
            // Test B's edge normals as reference axes against A.
            const SatAxis axisB = FindReferenceAxis(vb, nb, va, na, trace);

            // Pick the less-penetrating (higher separation) axis.
            // Tie-break: if equal, prefer A as reference (refIsA=true).
            // Both separations are negative (we're in the overlap case).
            bool refIsA;
            SatAxis refAxis;
            if (axisA.separation >= axisB.separation)
            {
                refIsA  = true;
                refAxis = axisA;
            }
            else
            {
                refIsA  = false;
                refAxis = axisB;
            }

            // Debug-viz: mark the chosen candidate axis (the min-penetration
            // reference axis the SAT search selected). The A-block axes occupy
            // satAxes[satBaseA .. satBaseB), the B-block axes [satBaseB .. end);
            // index within each block is the reference-shape edge index.
            if (trace)
            {
                const std::size_t pick = refIsA
                    ? satBaseA + static_cast<std::size_t>(refAxis.edgeIdx)
                    : satBaseB + static_cast<std::size_t>(refAxis.edgeIdx);
                if (pick < trace->satAxes.size())
                {
                    trace->satAxes[pick].chosen = true;
                }
            }

            // Reference face: vertices of the reference polygon.
            const Vec2* vsRef = refIsA ? va : vb;
            const int   nRef  = refIsA ? na : nb;
            const Vec2* vsInc = refIsA ? vb : va;
            const int   nInc  = refIsA ? nb : na;

            // The reference edge index.
            const int refEdge = refAxis.edgeIdx;

            // The reference face outward normal (world space, unit).
            // This points AWAY from the reference shape (outward).
            // We need it to point B->A: if refIsA the outward normal of A
            // already points away from A (outward), and since A is the body
            // being pushed out is NOT the goal -- actually:
            //   normal convention: B -> A (push A out of B).
            // If refIsA (reference face on A): the outward normal of A's edge
            // points AWAY from A -- that is, toward B. So the B->A direction
            // is the NEGATION of A's outward normal.
            // If refIsB (reference face on B): the outward normal of B's edge
            // points AWAY from B -- that is, toward A. So the B->A direction
            // IS B's outward normal.
            Vec2 contactNormal;
            if (refIsA)
            {
                // Reference is A; its outward normal points toward B (away from A).
                // B->A = opposite direction.
                contactNormal = Vec2(-refAxis.normal.x, -refAxis.normal.y);
            }
            else
            {
                // Reference is B; its outward normal points toward A (B->A).
                contactNormal = refAxis.normal;
            }

            // Reference face verts: edge refEdge -> refEdge+1.
            const Vec2& refV0 = vsRef[refEdge];
            const Vec2& refV1 = vsRef[(refEdge + 1) % nRef];

            // Reference face tangent (direction along the face).
            const Vec2 refTangent = Vec2(refV1.x - refV0.x, refV1.y - refV0.y);

            // Find the incident edge on the incident polygon: the edge whose
            // outward normal is most anti-parallel to the REFERENCE face's
            // outward normal (refAxis.normal). This selects the face of the
            // incident shape that is most facing the reference face, i.e.,
            // the face being "pushed into" the reference shape.
            // (Most anti-parallel = dot product with refAxis.normal most negative.)
            int incEdge = 0;
            {
                Real minDot = std::numeric_limits<Real>::infinity();
                for (int i = 0; i < nInc; ++i)
                {
                    const Vec2 n = EdgeNormalCCW(vsInc, nInc, i);
                    // Dot product with the reference face outward normal:
                    // the incident face that faces the reference face has the
                    // most negative dot product with refAxis.normal.
                    const Real d = n.x * refAxis.normal.x + n.y * refAxis.normal.y;
                    if (d < minDot)
                    {
                        minDot  = d;
                        incEdge = i;
                    }
                }
            }

            // Incident edge verts (the two endpoints to clip).
            const Vec2& incV0 = vsInc[incEdge];
            const Vec2& incV1 = vsInc[(incEdge + 1) % nInc];

            // ----------------------------------------------------------------
            // Clip the incident segment against the two side planes of the
            // reference face (Box2D v3 b2ClipSegments).
            //
            // Side plane 1: the "left" side of the reference edge (along -tangent).
            //   plane normal = -refTangent (normalised), offset = -tangent . refV0
            // Side plane 2: the "right" side (along +tangent).
            //   plane normal = +refTangent (normalised), offset = +tangent . refV1
            // ----------------------------------------------------------------
            const Real tangentLen = std::sqrt(refTangent.x * refTangent.x +
                                               refTangent.y * refTangent.y);
            Vec2 tangentN{ Real(1), Real(0) };
            if (tangentLen > Real(1e-9f))
            {
                tangentN = Vec2(refTangent.x / tangentLen,
                                refTangent.y / tangentLen);
            }

            // Side clip 1: keep points where dot(p, -tangentN) <= -dot(refV0, tangentN)
            //   <=> dot(p, tangentN) >= dot(refV0, tangentN)
            // Rewrite as: dot(p, -tangentN) <= dot(refV0, -tangentN)
            const Vec2 negTangentN = Vec2(-tangentN.x, -tangentN.y);
            const Real offset1 = refV0.x * negTangentN.x + refV0.y * negTangentN.y;

            Vec2 clipBuf1[2];
            float clipT1[2]{};
            int nc1 = ClipSegment(incV0, incV1, negTangentN, offset1,
                                   clipBuf1, clipT1);

            if (nc1 == 0) return m; // degenerate: nothing survives clip 1

            // Side clip 2: keep points where dot(p, tangentN) <= dot(refV1, tangentN)
            const Real offset2 = refV1.x * tangentN.x + refV1.y * tangentN.y;

            Vec2 clipBuf2[2];
            float clipT2[2]{};
            int nc2;
            if (nc1 >= 2)
            {
                // Normal 2-point case: Sutherland-Hodgman clip of the surviving segment.
                nc2 = ClipSegment(clipBuf1[0], clipBuf1[1],
                                   tangentN, offset2, clipBuf2, clipT2);
            }
            else
            {
                // nc1 == 1: only a single point survived clip 1.
                // Do NOT pass (p, p) to ClipSegment — the zero-length segment
                // would fire the crossing logic and could emit the same point twice.
                // Instead, just apply a single half-plane keep/drop on the one point.
                const Real d = clipBuf1[0].x * tangentN.x
                             + clipBuf1[0].y * tangentN.y - offset2;
                if (d <= Real(0))
                {
                    clipBuf2[0] = clipBuf1[0];
                    clipT2[0]   = clipT1[0];
                    nc2 = 1;
                }
                else
                {
                    nc2 = 0;
                }
            }

            if (nc2 == 0) return m;

            // ----------------------------------------------------------------
            // Keep only the clipped points that are on the "inside" of the
            // reference face (behind its plane, i.e. penetrating the reference
            // shape). Compute the signed depth per point.
            //
            // The reference face plane: normal = refAxis.normal (outward from
            // reference shape), offset = dot(refV0, refAxis.normal).
            // A point p is inside (penetrating) when:
            //   dot(p, refAxis.normal) < dot(refV0, refAxis.normal)
            // i.e. depth = dot(refV0, refAxis.normal) - dot(p, refAxis.normal) > 0
            // ----------------------------------------------------------------
            const Real refFaceOffset = refV0.x * refAxis.normal.x +
                                       refV0.y * refAxis.normal.y;

            m.normal = contactNormal;
            m.pointCount = 0;
            m.kind = NarrowphaseKind::SatPolygon; // poly-poly SAT ref-face clip

            for (int i = 0; i < nc2 && m.pointCount < 2; ++i)
            {
                const Vec2& cp = clipBuf2[i];
                const Real  proj = cp.x * refAxis.normal.x + cp.y * refAxis.normal.y;
                const Real  depth = refFaceOffset - proj; // positive = penetrating

                // Only keep penetrating points (depth > 0).
                // A point exactly on the face (depth==0) is a degenerate touch;
                // we discard it for stability (the speculative path handles gaps).
                if (depth <= Real(0)) continue;

                // Stable feature id: pack (refIsA, refEdge, incEdge + sub-index).
                // For the two clipped points we use incEdge itself and incEdge+1
                // as the "incident vertex" sub-index so the two ids differ:
                //   point 0: incFeature = incEdge
                //   point 1: incFeature = incEdge+1  (wraps mod nInc)
                // incFeature identifies the INCIDENT EDGE VERTEX (start or end of
                // the incident edge), NOT the clipped-point location.  This is
                // intentional (Box2D v3 semantics): the same feature pair produces
                // the same id regardless of how deeply the shapes overlap, so the
                // solver's warm-start cache finds the right impulse every frame.
                const uint32_t incFeature = static_cast<uint32_t>(
                    (incEdge + i) % nInc);
                const uint32_t id = MakeFeatureId(
                    refIsA,
                    static_cast<uint32_t>(refEdge),
                    incFeature);

                ManifoldPoint mp{};
                mp.point      = cp;
                mp.separation = depth;
                mp.normal     = contactNormal;
                mp.id         = id;
                m.points[m.pointCount++] = mp;
            }

            return m;
            } // CollidePoly

        } // namespace

        // ====================================================================
        // Collide -- the unified v2 narrowphase entry (explicit dispatcher).
        // ====================================================================
        //
        // The top of Collide is a small static type-pair dispatcher over named
        // cells, classified by the two cores' vertex counts:
        //
        //   ROUND cell  (na <= 2 || nb <= 2 -- circle = 1 vert, capsule = 2):
        //     CollideRound. Its GJK-witness branch handles separated / shallow /
        //     speculative; its DEEP branch (cores overlapping) now calls EPA for
        //     the EXACT nearest-face penetration (replacing the old centroid
        //     approximation). Always a 1-point manifold.
        //
        //   POLY cell   (na >= 3 && nb >= 3 -- polygon / AABB):
        //     CollidePoly. GJK for separated / speculative; SAT reference-face
        //     clip for deep overlap (up to 2 points). This cell is NOT routed to
        //     EPA -- the convex-poly SAT MTV is already exact and the 2-point
        //     clip is what the poly solver consumes. UNCHANGED by Task 3.
        //
        // The dispatcher rotates each shape's unified core verts into world space
        // (RotateInto) and forwards the world cores + radii to the chosen cell.
        // ====================================================================
        Manifold Collide(const Shape& a, const Transform& xfA,
                         const Shape& b, const Transform& xfB,
                         Real speculativeMargin,
                         NarrowphaseTrace* trace)
        {
            // Debug-viz (OPT-IN; nullptr on every Step-path call -> no-op):
            // record the two world shapes + transforms up front so the inspector
            // can draw the colliding shapes even when the early AABB reject
            // returns an empty manifold below. The per-path snapshots + the
            // final manifold are filled at the natural points / on return.
            if (trace)
            {
                trace->shapeA = a;
                trace->shapeB = b;
                trace->xfA    = xfA;
                trace->xfB    = xfB;
            }

            // Stack scratch: kMaxPolyVerts = 128, but the common cases (circle,
            // capsule, aabb) are 1-4 verts. 128 * sizeof(Vec2) = 1 KB -- fine.
            Vec2 va[kMaxPolyVerts];
            Vec2 vb[kMaxPolyVerts];

            const int na = RotateInto(va, a, xfA);
            const int nb = RotateInto(vb, b, xfB);

            const Real rA = a.radius;
            const Real rB = b.radius;

            // ---- early world-AABB reject (cheap broadphase-grade cull) --------
            //
            // The unified narrowphase is called once per FIXTURE PAIR by the
            // contact-gen + event-overlap loops, which iterate every fixture of
            // body A against every fixture of body B with only a BODY-level
            // broadphase pre-filter.  A multi-fixture body (e.g. a 37-wire whisk
            // agitator, or a compound) therefore drives the full GJK/SAT/EPA on
            // many fixture pairs that are nowhere near each other -- profiling a
            // 1000-body churn measured ~89% of Collide() calls on PROVABLY
            // DISJOINT pairs (45k calls/step to find ~2k real contacts).
            //
            // A rounded shape's EXACT world AABB is its core-vert AABB inflated
            // by its radius (circle: point +/- r; capsule: segment +/- r;
            // polygon: r == 0).  If A's and B's AABBs, with the gap further
            // inflated by speculativeMargin, do not overlap then the shapes are
            // strictly more than the speculative margin apart, so the dispatch
            // below would return an EMPTY manifold regardless.  Rejecting here is
            // therefore conservative + exact: identical manifolds, minus the
            // wasted support-mapping iterations.  Index-order / determinism are
            // unaffected (the decision is a pure function of the two transforms).
            if (na >= 1 && nb >= 1)
            {
                Real aMinX = va[0].x, aMaxX = va[0].x, aMinY = va[0].y, aMaxY = va[0].y;
                for (int i = 1; i < na; ++i) {
                    aMinX = std::min(aMinX, va[i].x); aMaxX = std::max(aMaxX, va[i].x);
                    aMinY = std::min(aMinY, va[i].y); aMaxY = std::max(aMaxY, va[i].y);
                }
                Real bMinX = vb[0].x, bMaxX = vb[0].x, bMinY = vb[0].y, bMaxY = vb[0].y;
                for (int i = 1; i < nb; ++i) {
                    bMinX = std::min(bMinX, vb[i].x); bMaxX = std::max(bMaxX, vb[i].x);
                    bMinY = std::min(bMinY, vb[i].y); bMaxY = std::max(bMaxY, vb[i].y);
                }
                const Real m = rA + rB + speculativeMargin;
                if ((aMaxX + m < bMinX) || (bMaxX + m < aMinX) ||
                    (aMaxY + m < bMinY) || (bMaxY + m < aMinY))
                {
                    // Provably separated -> no contact points. The trace keeps
                    // its default (Separated kind + empty manifold), already set
                    // by Clear() on the DebugCollide path.
                    return Manifold{};
                }
            }

            // Type-pair dispatch by core vertex count.
            const bool roundCell = (na <= 2 || nb <= 2);
            const Manifold result = roundCell
                // ROUND: circle/capsule core -> GJK witness (shallow) or EPA (deep).
                ? CollideRound(va, na, rA, vb, nb, rB, speculativeMargin, trace)
                // POLY: both cores >= 3 verts -> GJK speculative or SAT clip.
                : CollidePoly(va, na, rA, vb, nb, rB, speculativeMargin, trace);

            // Debug-viz: copy the final kind + manifold into the trace. The
            // per-path snapshot vectors were already filled at the natural
            // points inside the cells above. NO-op on the Step path (trace null).
            if (trace)
            {
                trace->kind     = result.kind;
                trace->manifold = result;
            }
            return result;
        }

    } // namespace Physics
} // namespace Manifold2D
