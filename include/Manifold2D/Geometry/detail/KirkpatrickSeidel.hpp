#pragma once
#include <Manifold2D/Geometry/detail/Predicates.hpp>   // DeterministicSelect, Less, Equal

namespace Manifold2D::Geometry::detail
{
    // Upper bridge over vertical line x=a (KS pairing + median-of-slopes prune).
    // Returns (left, right) endpoints with left.x <= a < right.x. Acc = double for
    // robustness; endpoints are exact input points.
    template <class T>
    std::pair<Pt<T>, Pt<T>> UpperBridge(std::vector<Pt<T>> S, double a)
    {
        using Acc = double;
        if (S.size() == 2)
            return Less<T>(S[0], S[1]) ? std::pair{ S[0], S[1] }
                                       : std::pair{ S[1], S[0] };

        // The supporting (max-h) point must be searched over the WHOLE set,
        // including the odd point set aside for pairing -- keep a full copy.
        const std::vector<Pt<T>> all = S;

        std::vector<Pt<T>> candidates;
        std::vector<std::pair<Pt<T>, Pt<T>>> pairs;
        if (S.size() % 2 == 1) { candidates.push_back(S.back()); S.pop_back(); }
        for (std::size_t i = 0; i + 1 < S.size(); i += 2)
        {
            Pt<T> pi = S[i], pj = S[i + 1];
            if (pi.x > pj.x) std::swap(pi, pj);
            if (pi.x == pj.x) candidates.push_back(pi.y > pj.y ? pi : pj);  // vertical
            else              pairs.push_back({ pi, pj });
        }
        if (pairs.empty()) return UpperBridge<T>(candidates, a);

        // Slopes are kept as the exact rational (dy, dx>0) of each pair -- never as a
        // rounded quotient. Forming K = dy/dx as a double and testing h(p) == h(q) with
        // h = y - K*x loses co-maximizers to a single-ULP rounding of K (so the bridge,
        // which is a slope-K pair, is never recognised and the prune diverges). All
        // slope / supporting-height comparisons below cross-multiply instead, exact for
        // integer inputs and robust for float.
        struct Slope { Acc dy, dx; };   // dx > 0 (pairs are ordered by x, distinct x)
        std::vector<Slope> slopes;
        slopes.reserve(pairs.size());
        for (const auto& pr : pairs)
            slopes.push_back({ Acc(pr.second.y) - Acc(pr.first.y),
                               Acc(pr.second.x) - Acc(pr.first.x) });
        // dx > 0 for both operands, so cross-multiplication preserves the slope order.
        const Slope K = DeterministicSelect<Slope>(
            slopes, slopes.size() / 2,
            [](const Slope& p, const Slope& q) { return p.dy * q.dx < q.dy * p.dx; });

        // h(p) - h(q) under slope K, scaled by K.dx > 0:  (yp-yq)*K.dx - K.dy*(xp-xq).
        // > 0 => p strictly higher than q on the slope-K supporting line.
        auto hCmp = [&](const Pt<T>& p, const Pt<T>& q) -> Acc {
            return (Acc(p.y) - Acc(q.y)) * K.dx - K.dy * (Acc(p.x) - Acc(q.x));
        };
        // pk = highest point, tie min x; pm = highest point, tie max x.
        Pt<T> pk = all[0], pm = all[0];
        for (const auto& p : all)
        {
            Acc c = hCmp(p, pk);
            if (c > Acc(0) || (c == Acc(0) && p.x < pk.x)) pk = p;
            c = hCmp(p, pm);
            if (c > Acc(0) || (c == Acc(0) && p.x > pm.x)) pm = p;
        }
        if (double(pk.x) <= a && double(pm.x) > a) return { pk, pm };

        // pair slope vs K, exact: si.dy/si.dx ? K.dy/K.dx  ->  si.dy*K.dx ? K.dy*si.dx.
        auto cmpK = [&](const Slope& s) -> Acc { return s.dy * K.dx - K.dy * s.dx; };
        if (double(pm.x) <= a)   // supporting set left of a: LARGE+EQUAL keep right, SMALL keep both
        {
            for (std::size_t i = 0; i < pairs.size(); ++i)
                if (cmpK(slopes[i]) < Acc(0)) { candidates.push_back(pairs[i].first);
                                                candidates.push_back(pairs[i].second); }
                else                          { candidates.push_back(pairs[i].second); }
        }
        else                     // pk.x > a, supporting set right of a: SMALL+EQUAL keep left, LARGE keep both
        {
            for (std::size_t i = 0; i < pairs.size(); ++i)
                if (cmpK(slopes[i]) > Acc(0)) { candidates.push_back(pairs[i].first);
                                                candidates.push_back(pairs[i].second); }
                else                          { candidates.push_back(pairs[i].first); }
        }
        return UpperBridge<T>(candidates, a);
    }

    template <class T>
    std::vector<Pt<T>> ConnectUpper(const Pt<T>& pmin, const Pt<T>& pmax,
                                    std::vector<Pt<T>> S)
    {
        std::vector<double> xs;
        xs.reserve(S.size());
        for (const auto& p : S) xs.push_back(double(p.x));
        const double a = DeterministicSelect<double>(
            xs, (xs.size() - 1) / 2, [](double x, double y) { return x < y; });

        auto [pl, pr] = UpperBridge<T>(S, a);
        std::vector<Pt<T>> sl, sr;
        for (const auto& p : S)
        {
            if (p.x < pl.x) sl.push_back(p);
            else if (p.x > pr.x) sr.push_back(p);
        }
        sl.push_back(pl);
        sr.push_back(pr);

        std::vector<Pt<T>> out;
        if (Equal<T>(pl, pmin)) out.push_back(pmin);
        else { auto L = ConnectUpper<T>(pmin, pl, sl); out.insert(out.end(), L.begin(), L.end()); }
        if (Equal<T>(pr, pmax)) out.push_back(pmax);
        else { auto R = ConnectUpper<T>(pr, pmax, sr); out.insert(out.end(), R.begin(), R.end()); }
        return out;
    }

    template <class T>
    std::vector<Pt<T>> UpperHull(std::vector<Pt<T>> P)
    {
        Pt<T> pmin = P[0], pmax = P[0];
        for (const auto& p : P)
        {
            if (p.x < pmin.x || (p.x == pmin.x && p.y > pmin.y)) pmin = p;
            if (p.x > pmax.x || (p.x == pmax.x && p.y > pmax.y)) pmax = p;
        }
        return ConnectUpper<T>(pmin, pmax, std::move(P));
    }

    template <class T>
    std::vector<Pt<T>> KirkpatrickSeidelBuild(std::span<const Pt<T>> pts)
    {
        std::vector<Pt<T>> P(pts.begin(), pts.end());
        std::vector<Pt<T>> upper = UpperHull<T>(P);

        std::vector<Pt<T>> Pn = P;                       // lower hull via y-negation
        for (auto& p : Pn) p.y = -p.y;
        std::vector<Pt<T>> lowerNeg = UpperHull<T>(Pn);
        std::vector<Pt<T>> lower;
        lower.reserve(lowerNeg.size());
        for (const auto& p : lowerNeg) lower.push_back(Pt<T>(p.x, -p.y));

        // Concatenate into one cycle: lower (pmin..pmax) + upper reversed (pmax..pmin).
        // The chains share their endpoints (pmin, pmax appear in both), so the naive
        // concatenation has consecutive seam duplicates AND a wrap-around duplicate.
        // Collapse consecutive repeats (incl. the cycle wrap) here -- a duplicated
        // genuine corner vertex would otherwise be wholly removed by Canonicalize's
        // StripCollinear (every copy has a zero neighbour cross).
        std::vector<Pt<T>> cycle = lower;
        for (std::size_t i = upper.size(); i-- > 0;) cycle.push_back(upper[i]);

        std::vector<Pt<T>> out;
        out.reserve(cycle.size());
        for (const auto& p : cycle)
            if (out.empty() || !Equal<T>(out.back(), p)) out.push_back(p);
        while (out.size() > 1 && Equal<T>(out.front(), out.back())) out.pop_back();
        return out;
    }
} // namespace Manifold2D::Geometry::detail
