#pragma once

// Manifold2D::Geometry public entry: ConvexHull<Policy,T>(points). Each algorithm is a
// stateless policy tag whose Build returns an ordered boundary cycle; the wrapper
// owns the shared contract (dedup, degenerate cases, canonical CCW/pivot form), so
// all policies return byte-identical canonical output for the same input.

#include <span>
#include <vector>

#include <Manifold2D/Geometry/detail/Predicates.hpp>
#include <Manifold2D/Geometry/detail/MonotoneChain.hpp>
#include <Manifold2D/Geometry/detail/GrahamScan.hpp>
#include <Manifold2D/Geometry/detail/JarvisMarch.hpp>
#include <Manifold2D/Geometry/detail/QuickHull.hpp>
#include <Manifold2D/Geometry/detail/Chan.hpp>
#include <Manifold2D/Geometry/detail/KirkpatrickSeidel.hpp>

namespace Manifold2D::Geometry
{
    struct MonotoneChain
    {
        template <class T>
        static std::vector<Pt<T>> Build(std::span<const Pt<T>> p)
        {
            return detail::MonotoneChainBuild<T>(p);
        }
    };

    struct GrahamScan
    {
        template <class T>
        static std::vector<Pt<T>> Build(std::span<const Pt<T>> p)
        {
            return detail::GrahamScanBuild<T>(p);
        }
    };

    struct JarvisMarch
    {
        template <class T>
        static std::vector<Pt<T>> Build(std::span<const Pt<T>> p)
        {
            return detail::JarvisMarchBuild<T>(p);
        }
    };

    struct QuickHull
    {
        template <class T>
        static std::vector<Pt<T>> Build(std::span<const Pt<T>> p)
        {
            return detail::QuickHullBuild<T>(p);
        }
    };

    struct Chan
    {
        template <class T>
        static std::vector<Pt<T>> Build(std::span<const Pt<T>> p)
        {
            return detail::ChanBuild<T>(p);
        }
    };

    struct KirkpatrickSeidel
    {
        template <class T>
        static std::vector<Pt<T>> Build(std::span<const Pt<T>> p)
        {
            return detail::KirkpatrickSeidelBuild<T>(p);
        }
    };

    template <class Policy, class T = float>
    std::vector<Pt<T>> ConvexHull(std::span<const Pt<T>> points)
    {
        std::vector<Pt<T>> pts = detail::Dedup<T>(points);
        const std::size_t n = pts.size();
        if (n < 3) return pts;   // 0/1/2 points: already lexicographically ordered.

        bool collinear = true;
        for (std::size_t i = 2; i < n && collinear; ++i)
        {
            if (detail::Orient2d<T>(pts[0], pts[1], pts[i]) != 0)
                collinear = false;
        }
        if (collinear)
            return { pts.front(), pts.back() };   // two extreme endpoints.

        std::vector<Pt<T>> hull = Policy::template Build<T>(std::span<const Pt<T>>(pts));
        return detail::Canonicalize<T>(std::move(hull));
    }
} // namespace Manifold2D::Geometry
