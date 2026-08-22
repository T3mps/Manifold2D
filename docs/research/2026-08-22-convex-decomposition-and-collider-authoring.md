# Convex decomposition and collider authoring

**Date:** 2026-08-22
**Status:** research — input to a spec, not itself a plan
**Scope:** a `ConvexDecompose<Policy>` entry in `Manifold2D::Geometry`, and the
Arcane-side collider editor that consumes it
**Asked for:** *"a nice fully integrated collider editor where users can place
points, and if they make a [concave] shape, we use decomp into other colliders
for the fixtures. This would probably be best placed in the manifold2d library
itself and then consumed by arcane."*

---

## One correction to the framing

Decomposition is for **concave** polygons. A convex polygon is already a single
valid fixture and needs no work. The editor's job is to *detect* concavity and
split only then — which also means the fast path for the common case (a box, a
triangle, a hand-drawn convex blob) is a convexity test and an early return.

---

## How other engines do this

**Unity — `PolygonCollider2D`.** The Inspector carries an *Edit Collider*
toggle that switches the scene view into a vertex-editing mode. Drag a vertex to
move it; hover the outline and click the dot to insert one; hold Ctrl/Cmd to turn
edges red and click to delete. Multiple disjoint paths are supported on one
collider. The decomposition is internal and not surfaced.

**Godot — `CollisionPolygon2D`.** A **Build Mode** enum picks *Solids* (the
polygon is decomposed into as many convex pieces as possible, so every
narrowphase query stays convex-vs-convex) or *Segments* (a hollow edge chain).
Worth knowing as a cautionary note: Godot's own tooling here was uneven for
years — `ConvexPolygonShape2D` and `ConcavePolygonShape2D` could not be
mouse-edited the way `CollisionPolygon2D` could
([godot#21394](https://github.com/godotengine/godot/issues/21394)), and only got
basic point editing in
[godot#100183](https://github.com/godotengine/godot/pull/100183). The lesson is
that "the shape resource" and "the authoring node" drifting apart is the failure
mode to design against.

**Box2D.** Ships **no** decomposition. The ecosystem standardised on Mark
Bayazit's algorithm — that is what `box2d-editor` and most bindings carry.

**Takeaway for us.** The interaction model is settled and worth copying almost
verbatim (drag to move, click-on-edge to insert, modifier-click to delete). What
is *not* settled anywhere is exposing the decomposition itself as a choice — and
because of the policy convention below, we get that nearly for free.

---

## What Manifold2D has today — measured

| | Status |
|---|---|
| `Geometry/ConvexHull.hpp` | **Present** — six policies |
| `Geometry/detail/Predicates.hpp` | **Present** — exact orientation kernel |
| `Geometry/Vec2.hpp` | Present |
| **Triangulation** | **ABSENT** — the `Monotone` match is `MonotoneChain`, the *hull* algorithm |
| **Convex decomposition** | **ABSENT** — no `Bayazit` / `Hertel` / `Decompose` anywhere |

`Geometry/` is exactly `ConvexHull.hpp`, `Vec2.hpp`, and `detail/`. The exact
predicates already there are the thing that keeps a decomposition from being
fragile on near-degenerate input, so the foundation is genuinely in place.

Binding constraints from the library's own invariants: **zero external
dependencies** (stdlib only), **header-only** for `Geometry/`, and
**determinism** under `/fp:strict` with no `/fp:fast`. All three are compatible
with pure-geometry decomposition; none of them are compatible with pulling in a
third-party decomposer.

---

## The convention, stated precisely

From `Geometry/ConvexHull.hpp`:

- Six algorithms — `MonotoneChain`, `GrahamScan`, `JarvisMarch`, `QuickHull`,
  `Chan`, `KirkpatrickSeidel` — each a **stateless policy tag struct** whose
  static `Build<T>(std::span<const Pt<T>>)` forwards to a `detail::XxxBuild<T>`
  in its own header.
- One public entry, `ConvexHull<Policy, T = float>(points)`.
- **The wrapper owns the shared contract**: dedup, `n < 3`, the all-collinear
  case, and `detail::Canonicalize` on the way out.
- Zero cost: policy is a template parameter, `Build` is static, everything
  inlines. No vtable, no indirection.
- The header states the payoff explicitly — *"all policies return byte-identical
  canonical output for the same input"* — and `tests/ConvexHullTest.cpp:144-147`
  **enforces** it, pinning every policy byte-equal to `MonotoneChain`.

Two observations while reading it:

1. `ConvexHull` is currently the **only** instance of this convention in the
   library. Adding a second is what turns it from a one-off into a house style,
   so the shape chosen here will be the template for later ones.
2. **`Policy` has no default** in the signature (`template <class Policy, class T = float>`),
   so every call site must name one. If "we pick a default" is meant to be part
   of the convention, it is not expressed in code yet —
   `template <class Policy = MonotoneChain, ...>` would say it.

---

## The design consequence that matters: the contract changes kind

**The byte-identical invariant cannot carry over to decomposition, and copying it
would be a design error.**

The convex hull of a point set is **unique**. That uniqueness is exactly what
lets six algorithms agree byte-for-byte after canonicalisation. A convex
decomposition is **not unique** — Bayazit and Hertel–Mehlhorn will produce
different cuts and different piece counts for the same polygon, and both are
correct.

So the wrapper's shared contract has to be **validity invariants** rather than
output equality. Every policy must satisfy:

- the union of the pieces equals the input polygon (no area gained or lost)
- every piece is convex
- every piece is CCW, and the pieces come back in a deterministic order
- no two pieces overlap in area

Piece count becomes a **measured characteristic of a policy**, not a violation.

That reshapes the test suite too, and in a direction this library is already
equipped for: `tests/` already uses **rapidcheck** alongside Catch2, and these
invariants are exactly property-test shaped. One shared invariant harness that
every policy is run through on generated polygons, plus a *piece-count
comparison* against the optimal policy instead of an equality assertion.

The wrapper should also own, mirroring `ConvexHull`'s degenerate handling:
input dedup and collinear-vertex removal, CCW normalisation, `n < 3` → empty,
**already-convex → single piece** (the common fast path), and rejection of
self-intersecting input — which is a real authoring case, since a user dragging
points *will* cross an edge.

---

## The policy set

| Policy | Character | Role |
|---|---|---|
| **Bayazit** | O(n·r), heuristic, no triangulation needed | The Box2D-ecosystem standard; good practical piece counts on hand-authored art |
| **Hertel–Mehlhorn** | Triangulate, then remove non-essential diagonals | Bounded **4-approximation** of minimal; what CGAL ships |
| **Keil–Snoeyink (optimal DP)** | Minimum piece count, expensive | The **reference** policy — plays `MonotoneChain`'s role, but as the thing others are *measured against* rather than equal to |
| **Greene DP** | The other classical approximation | Completeness of the "needed set" |
| **Approximate (ACD)** | Tolerance-based; allows slight concavity for far fewer pieces | Arguably the right **default** for hand-authored game colliders |

The default deserves thought rather than inheritance. For a *physics* collider,
fewer pieces is usually worth more than exactness — a 12-piece exact
decomposition of a hand-drawn blob costs more per frame than a 4-piece
approximate one that is visually indistinguishable. That argues for ACD as the
default with an explicit tolerance, and exactness as the opt-in. It is a product
call, not a geometry one.

---

## The sequencing question: triangulation first?

Hertel–Mehlhorn needs a triangulation, and **triangulation is itself a "needed
algorithms" family** — ear clipping, monotone partition, Delaunay. There is none
in the library today.

So there is a real fork:

- **Burying a private ear-clipper** inside `detail/HertelMehlhorn.hpp` gets
  decomposition landed sooner, but hides a primitive that several later things
  want (mesh generation, area computation, point-in-polygon acceleration,
  rendering debug shapes) and violates the spirit of the convention the moment a
  second consumer appears.
- **Landing `Triangulate<Policy>` first**, as its own instance of the same
  convention, costs an extra step but means Hertel–Mehlhorn is a thin consumer
  and the primitive is available to everything after.

Given the house rule is to implement the needed set rather than the one that
happens to be required today, the second is more consistent — and it also gives
the convention its second instance, which is what establishes it as a pattern.

---

## The Arcane side

**The split.** Decomposition is pure geometry with no physics dependency and no
engine dependency: it belongs in `Manifold2D::Geometry` beside `ConvexHull`, and
Arcane consumes it. The editor — point placement, dragging, insert/delete
interaction, undo integration, gizmo overlay — is Arcane's, and none of it
belongs in the library.

**A prerequisite Arcane must clear first.** `Collider2D::fixtures` is currently
reflected as a `std::vector<Fixture>` and is marked `Serializable(false)` in
Arcane — because Arcane's JSON bridge (`Serialization/ReflectionJson.hpp`)
cannot represent containers, and worse, its `Visit` classified by type *before*
checking whether the key was present, so an unhandled field type latched an error
even when nothing had ever written it. Adding a `Collider2D` in the editor and
saving produced a scene that could never be opened again; `Serializable(false)`
stopped the bricking, at the cost of colliders not persisting at all.

**A collider editor whose output cannot be saved is worthless**, so this is a
hard prerequisite. The durable fix is that **Astra already ships
`Reflection/ContainerTraits.hpp`** — full `IsContainer` / `IsSequence` /
`IsAssociative` / `HasContiguousStorage` detection with `ValueType` / `KeyType` /
`MappedType` — and Arcane's JSON bridge simply does not use it, enumerating a
fixed scalar type list instead. Teaching the bridge containers through those
traits makes `fixtures` round-trip, and closes the latent class hazard (any
*future* reflected field of an unhandled type bricking scenes) at the same time.

**Interaction model** worth copying from Unity: an explicit edit-mode toggle
rather than always-live handles, drag-to-move, click-on-edge-to-insert,
modifier-click-to-delete, and support for multiple disjoint paths on one
collider. Arcane already has the pieces this needs — a gizmo layer, an undo
system where one drag is one undo, and a selection outline.

**One Arcane-side caution:** the gizmo's `DecomposeTRS` currently assumes a
planar basis and refuses the drag on a tilted parent. A collider editor's handles
will need the same guard or the same refusal, for the same reason.

---

## Open questions

1. **Default policy** — ACD with a tolerance, or an exact method? Product call:
   fewer pieces vs. exactness.
2. **`Triangulate<Policy>` first, or a private ear-clipper inside
   Hertel–Mehlhorn?** Consistency argues for the former.
3. **Should `ConvexHull`'s `Policy` gain a default** (`= MonotoneChain`), so the
   convention's "pick a default" half is expressed in code before a second entry
   copies its shape?
4. **Self-intersecting input** — reject, or auto-repair? A user dragging points
   will cross an edge; rejecting is honest but needs a good editor affordance.
5. **Where does the authored polygon live?** A `Collider2D` holding the source
   path plus cached decomposed fixtures, or fixtures only with the path
   reconstructed? The first is re-editable and is what Unity and Godot both do.

---

## Related

- `Geometry/ConvexHull.hpp` — the convention this follows
- `Geometry/detail/Predicates.hpp` — the exact kernel decomposition needs
- Arcane: `docs/research/2026-08-21-3d-foundations-assessment.md` (F2 defines
  component/asset shape), `Serialization/ReflectionJson.hpp` (the container
  prerequisite)
