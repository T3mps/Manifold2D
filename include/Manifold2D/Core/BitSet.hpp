#pragma once

// Manifold2D::BitSet -- a minimal fixed-capacity bit set (the b2BitSet equivalent).
// This is the move origin for this type (design: docs/superpowers/specs/2026-07-10-manifold2d-phase2-lift-design.md,
// D3) -- the host engine's now-duplicate copy is deleted in Task 3 once
// Physics/Geometry retarget here.
// Used by the narrowphase MT serial tail: each worker flags changed contact ids
// into its own BitSet; the tail OR-reduces (InPlaceUnion) and walks set bits in
// ascending order (ForEachSetBit, via std::countr_zero == Box2D b2CTZ64).
// Presentation-free + C++23-clean: std only.

#include <bit>        // std::countr_zero
#include <cassert>    // assert (E01-3a debug bounds guard)
#include <cstddef>
#include <cstdint>
#include <vector>

namespace Manifold2D
{
    class BitSet
    {
    public:
        // Size to hold [0, bitCount) bits. Grows the block backing; never shrinks
        // capacity (grow-only -> zero steady-state alloc when reused per step).
        void Resize(std::size_t bitCount)
        {
            const std::size_t blocks = (bitCount + 63u) / 64u;
            if (blocks > m_blocks.size())
            {
                m_blocks.resize(blocks, 0ull);
            }
            m_blockCount = blocks;
        }

        void ClearAll() noexcept
        {
            for (std::size_t i = 0; i < m_blockCount; ++i) { m_blocks[i] = 0ull; }
        }

        void Set(std::size_t i) noexcept
        {
            // E01-3a: bounds guard. i must be < capacity (m_blockCount * 64);
            // an out-of-range index is an OOB write into m_blocks (UB). Debug
            // assert only -- the release path stays branch-free (assert compiles
            // out under NDEBUG). Resize(bitCount) must precede any Set.
            assert(i < m_blockCount * 64u && "BitSet::Set: index out of range (call Resize first)");
            m_blocks[i >> 6] |= (1ull << (i & 63u));
        }

        // OR `other` into this. Both must have been Resize()d to the same bitCount.
        void InPlaceUnion(const BitSet& other) noexcept
        {
            const std::size_t n = m_blockCount < other.m_blockCount
                                      ? m_blockCount : other.m_blockCount;
            for (std::size_t i = 0; i < n; ++i) { m_blocks[i] |= other.m_blocks[i]; }
        }

        // Call fn(id) for every set bit, ascending id order (block scan + CTZ).
        template <class Fn>
        void ForEachSetBit(Fn&& fn) const
        {
            for (std::size_t k = 0; k < m_blockCount; ++k)
            {
                std::uint64_t bits = m_blocks[k];
                while (bits != 0ull)
                {
                    const std::uint32_t ctz = static_cast<std::uint32_t>(std::countr_zero(bits));
                    fn(static_cast<std::uint32_t>(64u * k) + ctz);
                    bits &= (bits - 1ull); // clear lowest set bit
                }
            }
        }

    private:
        std::vector<std::uint64_t> m_blocks;
        std::size_t                m_blockCount = 0;
    };
} // namespace Manifold2D
