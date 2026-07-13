#pragma once
#include <Manifold2D/Geometry/detail/Predicates.hpp>
#include <Manifold2D/Geometry/detail/MonotoneChain.hpp>   // CCW sub-hulls

namespace Manifold2D::Geometry::detail
{
    template <class T>
    int Turn(const Pt<T>& p, const Pt<T>& q, const Pt<T>& r) noexcept
    {
        return Orient2d<T>(p, q, r);   // {-1 CW, 0 collinear, +1 CCW}
    }
    template <class T>
    T Dist2(const Pt<T>& a, const Pt<T>& b) noexcept
    {
        const T dx = a.x - b.x, dy = a.y - b.y;
        return dx * dx + dy * dy;
    }

    // Right tangent from external/boundary point p to CCW polygon `hull`
    // (Tom Switzer). O(log n).
    template <class T>
    int RTangent(const std::vector<Pt<T>>& hull, const Pt<T>& p)
    {
        const int n = static_cast<int>(hull.size());
        int l = 0, r = n;
        // lPrev/lNext cache the turn signs at the current left boundary l. They are
        // only refreshed in the branch that advances l; the r = c branch leaves l (and
        // hence these) unchanged -- intentionally, not a stale-read bug.
        int lPrev = Turn<T>(p, hull[0], hull[(0 + n - 1) % n]);
        int lNext = Turn<T>(p, hull[0], hull[(0 + 1) % n]);
        while (l < r)
        {
            const int c = (l + r) / 2;
            const int cPrev = Turn<T>(p, hull[c], hull[(c + n - 1) % n]);
            const int cNext = Turn<T>(p, hull[c], hull[(c + 1) % n]);
            const int cSide = Turn<T>(p, hull[l], hull[c]);
            if (cPrev != -1 && cNext != -1) return c;
            if ((cSide == 1 && (lNext == -1 || lPrev == lNext)) ||
                (cSide == -1 && cPrev == -1))
            {
                r = c;
            }
            else
            {
                l = c + 1;
                lPrev = -cNext;
                lNext = Turn<T>(p, hull[l % n], hull[(l + 1) % n]);
            }
        }
        return l % n;
    }

    template <class T>
    std::pair<int, int> MinHullPt(const std::vector<std::vector<Pt<T>>>& hulls)
    {
        int hi = 0, pi = 0;
        for (int h = 0; h < static_cast<int>(hulls.size()); ++h)
        {
            int j = 0;
            for (int k = 1; k < static_cast<int>(hulls[h].size()); ++k)
                if (Less<T>(hulls[h][k], hulls[h][j])) j = k;
            if (Less<T>(hulls[h][j], hulls[hi][pi])) { hi = h; pi = j; }
        }
        return { hi, pi };
    }

    template <class T>
    std::pair<int, int> NextHullPt(const std::vector<std::vector<Pt<T>>>& hulls,
                                   std::pair<int, int> cur)
    {
        const Pt<T> p = hulls[cur.first][cur.second];
        std::pair<int, int> next = {
            cur.first, (cur.second + 1) % static_cast<int>(hulls[cur.first].size()) };
        for (int h = 0; h < static_cast<int>(hulls.size()); ++h)
        {
            if (h == cur.first) continue;
            const int s = RTangent<T>(hulls[h], p);
            const Pt<T> q = hulls[next.first][next.second];
            const Pt<T> r = hulls[h][s];
            const int t = Turn<T>(p, q, r);
            if (t == -1 || (t == 0 && Dist2<T>(p, r) > Dist2<T>(p, q)))
                next = { h, s };
        }
        return next;
    }

    template <class T>
    std::vector<Pt<T>> ChanBuild(std::span<const Pt<T>> pts)
    {
        const int n = static_cast<int>(pts.size());   // lex-sorted (wrapper)
        for (int t = 0; ; ++t)
        {
            if ((1LL << t) >= 31) return MonotoneChainBuild<T>(pts);   // overflow guard
            long long mm = 1LL << (1LL << t);
            if (mm > n) mm = n;
            const int m = static_cast<int>(mm);

            std::vector<std::vector<Pt<T>>> hulls;
            for (int i = 0; i < n; i += m)
            {
                const int e = std::min(i + m, n);
                hulls.push_back(MonotoneChainBuild<T>(
                    std::span<const Pt<T>>(pts.data() + i,
                                           static_cast<std::size_t>(e - i))));
            }

            const std::pair<int, int> start = MinHullPt<T>(hulls);
            std::vector<std::pair<int, int>> chain{ start };
            bool closed = false;
            for (int step = 0; step < m; ++step)
            {
                const std::pair<int, int> nx = NextHullPt<T>(hulls, chain.back());
                if (nx == start) { closed = true; break; }
                chain.push_back(nx);
            }
            if (closed)
            {
                std::vector<Pt<T>> out;
                out.reserve(chain.size());
                for (const auto& pr : chain) out.push_back(hulls[pr.first][pr.second]);
                return out;
            }
            if (m >= n) return MonotoneChainBuild<T>(pts);   // safety net
        }
    }
} // namespace Manifold2D::Geometry::detail
