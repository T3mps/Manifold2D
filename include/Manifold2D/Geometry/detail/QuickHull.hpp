#pragma once
#include <Manifold2D/Geometry/detail/Predicates.hpp>

namespace Manifold2D::Geometry::detail
{
    // Internal recursion helper (internal linkage: a detail-only header function).
    template <class T>
    static void QuickHullRec(std::span<const Pt<T>> pts, std::size_t a, std::size_t b,
                             const std::vector<std::size_t>& set, std::vector<Pt<T>>& out)
    {
        // Farthest point strictly left of a->b. The side is decided by the EXACT
        // Orient2d sign; the farthest-point ranking is a double-promoted magnitude
        // (ranking need not be exact, only monotonic among the strictly-left set).
        // The gate is exact but `mag` is not: at large exponent spread a strictly-left
        // point's mag can round to <= 0, so `mag > best` (best = 0) would never fire
        // and drop a vertex the EXACT gate accepted. Never let the inexact rank
        // contradict the exact gate: the FIRST gated candidate always sets `c`; later
        // ones replace it only if strictly farther.
        double best = 0.0;
        std::size_t c = static_cast<std::size_t>(-1);
        for (std::size_t idx : set)
        {
            if (Orient2d<T>(pts[a], pts[b], pts[idx]) <= 0) continue;   // strictly left only
            const double mag = (double(pts[b].x) - double(pts[a].x)) * (double(pts[idx].y) - double(pts[a].y))
                             - (double(pts[b].y) - double(pts[a].y)) * (double(pts[idx].x) - double(pts[a].x));
            if (c == static_cast<std::size_t>(-1) || mag > best) { best = mag; c = idx; }
        }
        if (c == static_cast<std::size_t>(-1)) { out.push_back(pts[a]); return; }

        std::vector<std::size_t> leftAC, leftCB;
        for (std::size_t idx : set)
        {
            if (idx == c) continue;
            if (Orient2d<T>(pts[a], pts[c], pts[idx]) > 0)      leftAC.push_back(idx);
            else if (Orient2d<T>(pts[c], pts[b], pts[idx]) > 0) leftCB.push_back(idx);
        }
        QuickHullRec<T>(pts, a, c, leftAC, out);
        QuickHullRec<T>(pts, c, b, leftCB, out);
    }

    template <class T>
    std::vector<Pt<T>> QuickHullBuild(std::span<const Pt<T>> pts)
    {
        const std::size_t n = pts.size();
        std::size_t minI = 0, maxI = 0;
        for (std::size_t i = 1; i < n; ++i)
        {
            if (Less<T>(pts[i], pts[minI])) minI = i;
            if (Less<T>(pts[maxI], pts[i])) maxI = i;
        }
        std::vector<std::size_t> upper, lower;
        for (std::size_t i = 0; i < n; ++i)
        {
            if (i == minI || i == maxI) continue;
            const int s = Orient2d<T>(pts[minI], pts[maxI], pts[i]);
            if (s > 0)      upper.push_back(i);
            else if (s < 0) lower.push_back(i);
        }
        std::vector<Pt<T>> out;
        QuickHullRec<T>(pts, minI, maxI, upper, out);   // minI..(upper chain)
        QuickHullRec<T>(pts, maxI, minI, lower, out);   // maxI..(lower chain)
        return out;
    }
} // namespace Manifold2D::Geometry::detail
