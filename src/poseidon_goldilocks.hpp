#ifndef POSEIDON_GOLDILOCKS
#define POSEIDON_GOLDILOCKS

#include "poseidon_goldilocks_base.hpp"
#include "poseidon_goldilocks_avx.hpp"

#ifdef __AVX512__
#include "poseidon_goldilocks_avx512.hpp"
#endif

#endif
