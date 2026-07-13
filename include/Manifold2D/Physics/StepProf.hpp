// StepProf: opt-in per-Step-phase timing. ZERO cost when ARCANE_STEPPROF is
// off (the default) -- the scoped timer compiles to an empty struct and
// Enabled() is constexpr false. Turn on for a measuring build only (it is
// NOT in the determinism/behavioral gate path). Presentation-free, header-only.
#pragma once
#include <cstdint>
#ifndef ARCANE_STEPPROF
#define ARCANE_STEPPROF 0
#endif
#if ARCANE_STEPPROF
#include <chrono>
#include <array>
#include <cstdio>
#endif
namespace Manifold2D::Physics::StepProf
{
    // Stable phase ids (one per Step stage). Keep in lockstep with the
    // ARCANE_STEPPROF_SCOPE call sites in PhysicsWorld::Step.
    enum class Phase : std::uint32_t {
        Stage1Snapshot = 0, Narrowphase, EmitConstraints, Solve,
        WarmStartWriteback, Bullet, IslandSleep, Events, Count
    };
    constexpr bool Enabled() noexcept { return ARCANE_STEPPROF != 0; }
#if ARCANE_STEPPROF
    struct Acc { std::uint64_t ns = 0; std::uint64_t calls = 0; };
    inline std::array<Acc, static_cast<std::size_t>(Phase::Count)>& Table() {
        static std::array<Acc, static_cast<std::size_t>(Phase::Count)> t{}; return t;
    }
    struct Scope {
        Phase p; std::chrono::high_resolution_clock::time_point t0;
        explicit Scope(Phase ph) : p(ph), t0(std::chrono::high_resolution_clock::now()) {}
        ~Scope() {
            const auto t1 = std::chrono::high_resolution_clock::now();
            auto& a = Table()[static_cast<std::size_t>(p)];
            a.ns += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
            a.calls += 1;
        }
    };
    inline void Reset() { for (auto& a : Table()) { a = Acc{}; } }
    inline void Dump(const char* tag) {
        std::printf("[STEPPROF] %s\n", tag);
        const char* names[] = {"stage1","narrow","emit","solve","wswb","bullet","sleep","events"};
        for (std::size_t i = 0; i < static_cast<std::size_t>(Phase::Count); ++i) {
            const auto& a = Table()[i];
            const double ms = a.calls ? (double)a.ns / 1e6 / (double)a.calls : 0.0;
            std::printf("  %-8s %8.4f ms/step  (%llu calls)\n", names[i], ms, (unsigned long long)a.calls);
        }
    }
#else
    struct Scope { explicit Scope(Phase) noexcept {} };
    inline void Reset() noexcept {}
    inline void Dump(const char*) noexcept {}
#endif
}
// Two-level macro expansion so __LINE__ is fully expanded before token-paste.
// This gives each ARCANE_STEPPROF_SCOPE a unique variable name per source line
// even when multiple scopes appear in the same function.
#define ARCANE_STEPPROF_CONCAT2(a,b) a##b
#define ARCANE_STEPPROF_CONCAT(a,b) ARCANE_STEPPROF_CONCAT2(a,b)
#define ARCANE_STEPPROF_SCOPE(phase) \
    ::Manifold2D::Physics::StepProf::Scope ARCANE_STEPPROF_CONCAT(arcane_stepprof_scope_, __LINE__){ ::Manifold2D::Physics::StepProf::Phase::phase }
