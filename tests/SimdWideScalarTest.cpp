// Forced-scalar SIMD tests -- exercises the scalar backend even under global
// /arch:AVX2, so the reference oracle is always covered in one ArcaneTests run.
#define MOSAIC_SIMD_SCALAR
#define MOSAIC_SIMD_TUTAG "scalar"
#include "Simd/SimdWideTests.inl"
