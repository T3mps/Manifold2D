#pragma once

// Manifold2D::FunctionRef<Sig> -- a non-owning, zero-allocation callable VIEW
// (void* + thunk). This is a standalone copy of a small leaf-vocabulary type
// also carried by the host engine (design: docs/superpowers/specs/2026-07-10-manifold2d-phase2-lift-design.md,
// D3) -- kept independent so Manifold2D has zero dependency on the host.
// For SYNCHRONOUS, non-escaping callbacks only -- the referent MUST outlive the
// FunctionRef. Do NOT store one (it would dangle); a stored callback uses an
// owning std::function (or a future Delegate). C++26 std::function_ref-aligned:
// this can be replaced by a using-alias when MSVC ships it.

#include <cassert>       // assert (E01-3b empty-call guard)
#include <memory>
#include <type_traits>

namespace Manifold2D
{
    template <class Sig> class FunctionRef;

    template <class R, class... Args>
    class FunctionRef<R(Args...)>
    {
        void* m_obj = nullptr;
        R (*m_thunk)(void*, Args...) = nullptr;

    public:
        FunctionRef() = default;

        // Implicit by design: call sites pass a lambda directly.
        template <class F>
            requires (!std::is_same_v<std::remove_cvref_t<F>, FunctionRef>
                      && std::is_invocable_r_v<R, F&, Args...>)
        FunctionRef(F&& f) noexcept
            // For a raw function name, F deduces as a function TYPE and addressof(f)
            // yields the function's stable address (not a temporary) -- safe. Same for
            // any lvalue callable. Do NOT bind a function-POINTER rvalue (e.g. a cast
            // result): F would deduce as a pointer type and m_obj would dangle.
            : m_obj(const_cast<void*>(static_cast<const void*>(std::addressof(f)))),
              m_thunk(+[](void* o, Args... a) -> R {
                  return (*static_cast<std::remove_reference_t<F>*>(o))(static_cast<Args&&>(a)...);
              })
        {
        }

        R operator()(Args... a) const
        {
            // E01-3b: calling through an empty (default-constructed / moved-from)
            // FunctionRef dereferences a null thunk -- UB. Debug assert on the
            // existing bool conversion; release path is unchanged (compiles out
            // under NDEBUG). Callers must check operator bool if emptiness is
            // reachable.
            assert(static_cast<bool>(*this) && "FunctionRef::operator(): call through an empty FunctionRef");
            return m_thunk(m_obj, static_cast<Args&&>(a)...);
        }

        explicit operator bool() const noexcept { return m_thunk != nullptr; }
    };
}
