// ContactPool: a persistent, deduped, generation-safe pool of fixture-pair contacts.
#include <algorithm>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <Manifold2D/Physics/Contact.hpp>

using namespace Manifold2D::Physics;

TEST_CASE("ContactPool dedups by fixture-slot pair and is generation-safe", "[physics]")
{
    ContactPool pool;
    const FixtureHandle f1{ 3, 1 }, f2{ 7, 1 }, f3{ 9, 1 };

    // (a) EnsurePair creates exactly one contact per unordered pair, and the
    //     explicit created/existing signal distinguishes a fresh create from a
    //     dedup HIT.
    const auto r12 = pool.EnsurePair(f1, f2);
    REQUIRE(r12.id != ContactPool::kNone);
    REQUIRE(r12.created == true);                 // first create signals fresh

    const auto r12dup = pool.EnsurePair(f2, f1);  // order-independent dedup
    REQUIRE(r12dup.id == r12.id);
    REQUIRE(r12dup.created == false);             // dedup HIT signals existing
    REQUIRE(pool.Count() == 1);

    const auto r13 = pool.EnsurePair(f1, f3);
    REQUIRE(r13.id != r12.id);
    REQUIRE(r13.created == true);
    REQUIRE(pool.Count() == 2);

    // A freshly-created contact has self-evidently unfilled body slots.
    const Contact& fresh = pool.Get(r13.id);
    REQUIRE(fresh.bodyA == kInvalidSlot);
    REQUIRE(fresh.bodyB == kInvalidSlot);

    // (b) Find returns the same id; the contact carries both handles.
    REQUIRE(pool.Find(f1, f2) == r12.id);
    const Contact& cc = pool.Get(r12.id);
    REQUIRE(((cc.a.index == 3 && cc.b.index == 7) || (cc.a.index == 7 && cc.b.index == 3)));

    // (c) Destroy frees the slot + removes the key; a later EnsurePair may recycle it.
    pool.Destroy(r12.id);
    REQUIRE(pool.Count() == 1);
    REQUIRE(pool.Find(f1, f2) == ContactPool::kNone);

    // (d) A recycled fixture slot (same index, new generation) is a DISTINCT pair.
    const FixtureHandle f2b{ 7, 2 };             // slot 7 recycled, gen bumped
    const auto r12b = pool.EnsurePair(f1, f2b);
    REQUIRE(r12b.created == true);
    REQUIRE(pool.Find(f1, f2) == ContactPool::kNone);   // old gen never aliases
    REQUIRE(pool.Find(f1, f2b) == r12b.id);
}

TEST_CASE("ContactPool ForEach visits live contacts in ascending id order, skipping the dead", "[physics]")
{
    ContactPool pool;
    const FixtureHandle f1{ 1, 1 }, f2{ 2, 1 }, f3{ 3, 1 };

    const auto a = pool.EnsurePair(f1, f2);
    const auto b = pool.EnsurePair(f1, f3);
    const auto c = pool.EnsurePair(f2, f3);
    REQUIRE(pool.Count() == 3);

    // Destroy the MIDDLE one -- ForEach must skip it and keep ascending order.
    pool.Destroy(b.id);
    REQUIRE(pool.Count() == 2);

    std::vector<std::uint32_t> visited;
    pool.ForEach([&](std::uint32_t id, const Contact&) { visited.push_back(id); });

    REQUIRE(visited.size() == 2);
    REQUIRE(std::is_sorted(visited.begin(), visited.end()));   // ascending id seam
    REQUIRE(std::find(visited.begin(), visited.end(), a.id) != visited.end());
    REQUIRE(std::find(visited.begin(), visited.end(), c.id) != visited.end());
    REQUIRE(std::find(visited.begin(), visited.end(), b.id) == visited.end()); // dead skipped
}

TEST_CASE("ContactPool recycles freed ids via the LIFO free-list", "[physics]")
{
    ContactPool pool;
    const FixtureHandle f1{ 4, 1 }, f2{ 5, 1 }, f3{ 6, 1 };

    const auto c12 = pool.EnsurePair(f1, f2);
    pool.Destroy(c12.id);

    // A fresh EnsurePair must reuse the just-freed id (LIFO free-list).
    const auto c13 = pool.EnsurePair(f1, f3);
    REQUIRE(c13.created == true);
    REQUIRE(c13.id == c12.id);                  // freed id reused
    REQUIRE(pool.Count() == 1);
}

TEST_CASE("ContactPool Clear empties the pool and leaves it reusable", "[physics]")
{
    ContactPool pool;
    const FixtureHandle f1{ 1, 1 }, f2{ 2, 1 }, f3{ 3, 1 };

    pool.EnsurePair(f1, f2);
    pool.EnsurePair(f1, f3);
    REQUIRE(pool.Count() == 2);

    pool.Clear();
    REQUIRE(pool.Count() == 0);
    REQUIRE(pool.Find(f1, f2) == ContactPool::kNone);   // old pair gone

    // Pool is reusable after Clear.
    const auto re = pool.EnsurePair(f2, f3);
    REQUIRE(re.created == true);
    REQUIRE(re.id != ContactPool::kNone);
    REQUIRE(pool.Count() == 1);
    REQUIRE(pool.Find(f2, f3) == re.id);
}

TEST_CASE("ContactPool Destroy is a no-op on an already-dead id and on kNone", "[physics]")
{
    ContactPool pool;
    const FixtureHandle f1{ 8, 1 }, f2{ 9, 1 };

    const auto c = pool.EnsurePair(f1, f2);
    REQUIRE(pool.Count() == 1);

    pool.Destroy(c.id);
    REQUIRE(pool.Count() == 0);

    // Double-Destroy: already dead -> no-op, no crash, Count unchanged.
    pool.Destroy(c.id);
    REQUIRE(pool.Count() == 0);

    // Destroy(kNone): out-of-range sentinel -> no-op, no crash.
    pool.Destroy(ContactPool::kNone);
    REQUIRE(pool.Count() == 0);
}
