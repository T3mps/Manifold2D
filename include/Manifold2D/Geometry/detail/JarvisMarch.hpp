#pragma once
#include <Manifold2D/Geometry/detail/Predicates.hpp>

namespace Manifold2D::Geometry::detail
{
    template <class T>
    std::vector<Pt<T>> JarvisMarchBuild(std::span<const Pt<T>> pts)
    {
        const std::size_t n = pts.size();
        std::size_t leftmost = 0;
        for (std::size_t i = 1; i < n; ++i)
            if (Less<T>(pts[i], pts[leftmost])) leftmost = i;

        std::vector<Pt<T>> hull;
        std::size_t cur = leftmost;
        do
        {
            hull.push_back(pts[cur]);
            std::size_t endp = (cur + 1) % n;
            for (std::size_t j = 0; j < n; ++j)
            {
                if (j == cur || j == endp) continue;
                const int c = Orient2d<T>(pts[cur], pts[endp], pts[j]);
                if (c > 0)
                {
                    endp = j;                       // strictly more CCW
                }
                else if (c == 0)
                {
                    const Pt<T>& b = pts[cur];      // collinear: take the farther
                    const T dj = (pts[j].x - b.x) * (pts[j].x - b.x) +
                                 (pts[j].y - b.y) * (pts[j].y - b.y);
                    const T de = (pts[endp].x - b.x) * (pts[endp].x - b.x) +
                                 (pts[endp].y - b.y) * (pts[endp].y - b.y);
                    if (dj > de) endp = j;
                }
            }
            cur = endp;
        } while (cur != leftmost);
        return hull;
    }
} // namespace Manifold2D::Geometry::detail
