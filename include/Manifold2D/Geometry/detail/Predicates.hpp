#pragma once

// Manifold2D::Geometry shared predicates + the canonical-form post-processing every
// convex-hull policy runs through. Presentation-free: Geometry::Vec2 + std only
// (compiles /MD and static-CRT). Header-only templates; scalar-generic on T
// (float|double).

#include <algorithm>
#include <cmath>          // std::fma
#include <cstddef>
#include <span>
#include <type_traits>    // std::is_same_v
#include <vector>

#include <Manifold2D/Geometry/Vec2.hpp>

namespace Manifold2D::Geometry
{
    // Pt is the historical name for the hull kernel's point type; it now aliases
    // the first-party Vec2<T> (Manifold2D Phase 1). Same layout, same .x/.y,
    // same construction forms -- the policies compile unchanged.
    template <class T> using Pt = Vec2<T>;

    namespace detail
    {
        // (b-o) is left of (a-o): >0 CCW/left turn, <0 CW/right, 0 collinear.
        template <class T>
        constexpr T Cross(const Pt<T>& o, const Pt<T>& a, const Pt<T>& b) noexcept
        {
            return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
        }

        // ---- E01-5 robust orientation predicate ------------------------------
        // Exact sign of the orientation determinant
        //   Cross(o,a,b) = (a.x-o.x)*(b.y-o.y) - (a.y-o.y)*(b.x-o.x).
        // Returns +1 (b left of o->a, CCW), -1 (right/CW), 0 (collinear) EXACTLY,
        // for ANY float or double input -- no coordinate-magnitude precondition --
        // so callers get the correct side even for near-collinear / large-coord /
        // adversarial cross-scale inputs where the plain-T Cross rounds to the wrong
        // side of zero. (Cross<T> stays for magnitude consumers that need the value,
        // not just the sign.)
        //
        // One robust kernel for both types: the sign is computed EXACTLY over the
        // EXPANDED determinant
        //     (ax*by - ay*bx) + (ay*ox - ax*oy) + (bx*oy - by*ox)
        //   (the o.x*o.y terms cancel) in double, each product split by std::fma
        //   TwoProduct and summed into a non-overlapping Shewchuk expansion whose top
        //   term carries the true sign. float inputs promote LOSSLESSLY to double and
        //   run the identical kernel, so the returned sign is exact for any float too.
        //   The double error-free transforms are correct ONLY if the compiler neither
        //   contracts a*b+c into an FMA nor reassociates the recovery terms -- i.e.
        //   under this repo's /fp:strict (the guard just above TwoSum fails the build
        //   loudly under /fp:fast). std::fma is single-rounding by the standard
        //   regardless of /fp mode.

        // The double error-free transforms below (TwoSum/GrowExpansion) are correct
        // only without FMA-contraction/reassociation -- i.e. under /fp:strict (this
        // repo's mode). Fail the build loudly if someone flips Core to /fp:fast.
#if defined(_M_FP_FAST)
#error "Manifold2D::Geometry robust predicates require /fp:strict (or /fp:precise); /fp:fast breaks the error-free transforms."
#endif

        // Knuth TwoSum: a+b == x+y exactly (round-to-nearest, no reassociation).
        inline void TwoSum(double a, double b, double& x, double& y) noexcept
        {
            x = a + b;
            const double z = x - a;
            y = (a - (x - z)) + (b - z);
        }
        // std::fma TwoProduct: a*b == p+e exactly (single-rounding fma).
        inline void TwoProduct(double a, double b, double& p, double& e) noexcept
        {
            p = a * b;
            e = std::fma(a, b, -p);
        }
        // Grow a non-overlapping increasing expansion e[0..m) by scalar b -> h[0..ret).
        inline int GrowExpansion(const double* e, int m, double b, double* h) noexcept
        {
            double Q = b;
            int hi = 0;
            for (int i = 0; i < m; ++i)
            {
                double s, err;
                TwoSum(Q, e[i], s, err);
                if (err != 0.0) h[hi++] = err;
                Q = s;
            }
            if (Q != 0.0 || hi == 0) h[hi++] = Q;
            return hi;
        }
        // Exact sign of the sum of n signed doubles (n small; determinant expansion
        // stays well under 32). Non-overlapping increasing expansion => the sign is
        // its most-significant nonzero component's sign.
        inline int ExactSignOfSum(const double* comps, int n) noexcept
        {
            double buf[2][32];
            int cur = 0, m = 0;
            for (int i = 0; i < n; ++i)
            {
                const int nx = GrowExpansion(buf[cur], m, comps[i], buf[cur ^ 1]);
                cur ^= 1;
                m = nx;
            }
            for (int i = m - 1; i >= 0; --i)
                if (buf[cur][i] != 0.0) return buf[cur][i] > 0.0 ? 1 : -1;
            return 0;
        }
        inline int Orient2dExactD(const Pt<double>& o, const Pt<double>& a,
                                  const Pt<double>& b) noexcept
        {
            double p, e, comps[12];
            int n = 0;
            TwoProduct(a.x, b.y, p, e); comps[n++] =  p; comps[n++] =  e;  // + ax*by
            TwoProduct(a.y, b.x, p, e); comps[n++] = -p; comps[n++] = -e;  // - ay*bx
            TwoProduct(a.y, o.x, p, e); comps[n++] =  p; comps[n++] =  e;  // + ay*ox
            TwoProduct(a.x, o.y, p, e); comps[n++] = -p; comps[n++] = -e;  // - ax*oy
            TwoProduct(b.x, o.y, p, e); comps[n++] =  p; comps[n++] =  e;  // + bx*oy
            TwoProduct(b.y, o.x, p, e); comps[n++] = -p; comps[n++] = -e;  // - by*ox
            return ExactSignOfSum(comps, n);
        }

        template <class T>
        int Orient2d(const Pt<T>& o, const Pt<T>& a, const Pt<T>& b) noexcept
        {
            static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                          "Orient2d supports only float and double");
            // float promotes LOSSLESSLY to double, and Orient2dExactD is exact for
            // any double, so the returned sign is exact for ANY float or double
            // input -- no coordinate-magnitude precondition (one robust kernel).
            return Orient2dExactD(Pt<double>{ double(o.x), double(o.y) },
                                  Pt<double>{ double(a.x), double(a.y) },
                                  Pt<double>{ double(b.x), double(b.y) });
        }

        // Lexicographic order: x then y.
        template <class T>
        constexpr bool Less(const Pt<T>& a, const Pt<T>& b) noexcept
        {
            return a.x < b.x || (a.x == b.x && a.y < b.y);
        }

        template <class T>
        constexpr bool Equal(const Pt<T>& a, const Pt<T>& b) noexcept
        {
            return a.x == b.x && a.y == b.y;
        }

        // Copy, lexicographically sort, drop exact duplicates.
        template <class T>
        std::vector<Pt<T>> Dedup(std::span<const Pt<T>> pts)
        {
            std::vector<Pt<T>> v(pts.begin(), pts.end());
            std::sort(v.begin(), v.end(),
                      [](const Pt<T>& a, const Pt<T>& b) { return Less<T>(a, b); });
            v.erase(std::unique(v.begin(), v.end(),
                                [](const Pt<T>& a, const Pt<T>& b) { return Equal<T>(a, b); }),
                    v.end());
            return v;
        }

        // 2x signed area of an ordered closed polygon (>0 => CCW in math orientation).
        template <class T>
        T SignedArea2(const std::vector<Pt<T>>& poly)
        {
            T s = T(0);
            const std::size_t n = poly.size();
            for (std::size_t i = 0; i < n; ++i)
            {
                const Pt<T>& a = poly[i];
                const Pt<T>& b = poly[(i + 1) % n];
                s += a.x * b.y - b.x * a.y;
            }
            return s;
        }

        // Drop vertices collinear with their neighbours (and seam duplicates, whose
        // neighbour cross is also 0) from an ordered closed polygon.
        template <class T>
        std::vector<Pt<T>> StripCollinear(const std::vector<Pt<T>>& poly)
        {
            const std::size_t n = poly.size();
            if (n < 3) return poly;
            std::vector<Pt<T>> out;
            out.reserve(n);
            for (std::size_t i = 0; i < n; ++i)
            {
                const Pt<T>& prev = poly[(i + n - 1) % n];
                const Pt<T>& cur  = poly[i];
                const Pt<T>& next = poly[(i + 1) % n];
                if (Orient2d<T>(prev, cur, next) != 0)
                    out.push_back(cur);
            }
            return out;
        }

        // Canonical form of a policy's ordered boundary cycle WITHOUT re-deriving the
        // hull: strip collinear, normalise winding to CCW EXACTLY (reverse if the
        // lex-min corner turns CW -- NOT an angular re-sort, so a policy ordering bug
        // still surfaces as a wrong result), rotate to start at the lexicographically
        // smallest vertex. Winding is decided by the exact Orient2d turn at the lex-min
        // vertex -- a guaranteed convex, non-collinear corner -- rather than a summed
        // shoelace (SignedArea2) whose sign large-coordinate cancellation can flip.
        template <class T>
        std::vector<Pt<T>> Canonicalize(std::vector<Pt<T>> hull)
        {
            hull = StripCollinear<T>(hull);
            if (hull.size() < 3) return hull;
            const std::size_t n = hull.size();
            // Lex-min vertex is an extreme point => a genuine convex corner
            // (non-collinear turn). Its exact orientation gives the winding without
            // summing a cancellation-prone area.
            std::size_t m = 0;
            for (std::size_t i = 1; i < n; ++i)
                if (Less<T>(hull[i], hull[m])) m = i;
            if (Orient2d<T>(hull[(m + n - 1) % n], hull[m], hull[(m + 1) % n]) < 0)
            {
                std::reverse(hull.begin(), hull.end());
                m = n - 1 - m;                 // same vertex, new index after reverse
            }
            std::rotate(hull.begin(), hull.begin() + static_cast<std::ptrdiff_t>(m),
                        hull.end());
            return hull;
        }

        // Deterministic median-of-medians (groups of 5) selection: returns the k-th
        // (0-based) smallest element of `a` under `less`. O(n) worst case. By value.
        template <class U, class Cmp>
        U DeterministicSelect(std::vector<U> a, std::size_t k, Cmp less)
        {
            for (;;)
            {
                const std::size_t n = a.size();
                if (n <= 5)
                {
                    std::sort(a.begin(), a.end(), less);
                    return a[k];
                }
                std::vector<U> medians;
                medians.reserve((n + 4) / 5);
                for (std::size_t i = 0; i < n; i += 5)
                {
                    const std::size_t e = std::min(i + 5, n);
                    std::sort(a.begin() + static_cast<std::ptrdiff_t>(i),
                              a.begin() + static_cast<std::ptrdiff_t>(e), less);
                    medians.push_back(a[i + (e - i) / 2]);
                }
                U pivot = DeterministicSelect<U>(medians, medians.size() / 2, less);
                std::vector<U> lo, hi;
                std::size_t eq = 0;
                for (const U& x : a)
                {
                    if (less(x, pivot))      lo.push_back(x);
                    else if (less(pivot, x)) hi.push_back(x);
                    else                     ++eq;
                }
                if (k < lo.size())            { a = std::move(lo); }
                else if (k < lo.size() + eq)  { return pivot; }
                else { k -= lo.size() + eq;     a = std::move(hi); }
            }
        }
    } // namespace detail
} // namespace Manifold2D::Geometry
