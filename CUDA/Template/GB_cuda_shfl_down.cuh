
// GB_cuda_shfl_down.cuh:  warp-level reductions


//==============================================================================
//==============================================================================
// built-in types (uint64_t and int64_t) with "+"
//==============================================================================
//==============================================================================

// FIXME: So far, we only need uint64_t.  Otherwise, we could use C++ templates
// for the data type.

// FIXME: do we assume tile_sz is always 32?  Seems reasonable.

//------------------------------------------------------------------------------
// warp_ReduceSumPlus_uint64: for dot3_phase2
//------------------------------------------------------------------------------

__inline__ __device__ uint64_t warp_ReduceSumPlus_uint64
(
    thread_block_tile<tile_sz> tile,
    uint64_t val
)
{
    // Each iteration halves the number of active threads
    // Each thread adds its partial sum[i] to sum[lane+i]
    for (int i = tile.size() / 2; i > 0; i /= 2)
    {
        val += tile.shfl_down (val, i) ;
    }
    return val; // note: only thread 0 will return full sum
}

//------------------------------------------------------------------------------
// GB_warp_ReduceSumPlus_uint64_vsvs: for vsvs kernel
//------------------------------------------------------------------------------

__inline__ __device__ uint64_t GB_warp_ReduceSumPlus_uint64_vsvs
(
    thread_block_tile<tile_sz> g,
    uint64_t val
)
{
    // Each iteration halves the number of active threads
    // Each thread adds its partial sum[i] to sum[lane+i]
    /*
    #pragma unroll
    for (int i = tile_sz >> 1; i > 0; i >>= 1) {
        val +=  g.shfl_down( val, i);
    }
    */
    // assuming tile_sz is 32:
    val +=  g.shfl_down( val, 16);
    val +=  g.shfl_down( val, 8);
    val +=  g.shfl_down( val, 4);
    val +=  g.shfl_down( val, 2);
    val +=  g.shfl_down( val, 1);
    return val; // note: only thread 0 will return full sum
}

//------------------------------------------------------------------------------
// reduce_sum_int64: for vsdn
//------------------------------------------------------------------------------

// FIXME: make this uint64_t.  Should this method be a C++ template type,
// so it can do any built-in type?

// for counting zombies only (always int64_t)
__device__ int64_t reduce_sum_int64
(
    thread_block_tile<tile_sz> g,
    int64_t val
)
{
    // Each iteration halves the number of active threads
    // Each thread adds its partial sum[i] to sum[lane+i]
    for (int64_t i = g.size() / 2; i > 0; i /= 2)
    {
        val += g.shfl_down(val,i) ;
    }
    return val; // note: only thread 0 will return full sum
}


//==============================================================================
//==============================================================================
// GB_Z_TYPE reduces, using the GB_ADD operator
//==============================================================================
//==============================================================================

//------------------------------------------------------------------------------
// warp_ReduceSum_dndn: for dndn kernel
//------------------------------------------------------------------------------

// FIXME: make this the same static device function
// #include "GB_reduce_whatever.cuh"

__inline__ __device__ GB_Z_TYPE warp_ReduceSum_dndn
(
    thread_block_tile<32> g,
    GB_Z_TYPE val
)
{
    // Each iteration halves the number of active threads
    // Each thread adds its partial sum[i] to sum[lane+i]
    // FIXME: only works if sizeof(GB_Z_TYPE) <= 32 bytes
    // FIXME: the ANY monoid needs the cij_exists for each thread
    for (int i = g.size() / 2; i > 0; i /= 2)
    {
        GB_Z_TYPE next = g.shfl_down( val, i) ;
        GB_ADD( val, val, next ); 
    }
    return val; // note: only thread 0 will return full sum
}

//------------------------------------------------------------------------------
// GB_reduce_sum: for dot3 mp and spdn
//------------------------------------------------------------------------------

__device__ __inline__ GB_Z_TYPE GB_reduce_sum
(
    thread_block_tile<tile_sz> g,
    GB_Z_TYPE val
)
{
    // Each iteration halves the number of active threads
    // Each thread adds its partial sum[i] to sum[lane+i]
    // Temporary GB_Z_TYPE is necessary to handle arbirary ops
    // FIXME: only works if sizeof(GB_Z_TYPE) <= 32 bytes
    // FIXME: the ANY monoid needs the cij_exists for each thread
    #pragma unroll
    for (int i = tile_sz >> 1 ; i > 0 ; i >>= 1)
    {
        GB_Z_TYPE next = g.shfl_down (val, i) ;
        GB_ADD (val, val, next) ; 
    }
    return val;
}

//------------------------------------------------------------------------------
// GB_warp_Reduce: for cuda_reduce
//------------------------------------------------------------------------------

__inline__ __device__ GB_Z_TYPE GB_warp_Reduce
(
    thread_block_tile<tile_sz> g,
    GB_Z_TYPE val
)
{
    // Each iteration halves the number of active threads
    // Each thread adds its partial val[k] to val[lane+k]

    // FIXME: doesn't work unless sizeof(GB_Z_TYPE) <= 32 bytes

#if ( GB_Z_NBITS <= 8*32 )
    // assumes tile_size is 32:
    GB_Z_TYPE fold = g.shfl_down ( val, 16) ;
    GB_ADD ( val, val, fold ) ;
    fold = g.shfl_down ( val, 8) ;
    GB_ADD ( val, val, fold ) ;
    fold = g.shfl_down ( val, 4) ;
    GB_ADD ( val, val, fold ) ;
    fold = g.shfl_down ( val, 2) ;
    GB_ADD ( val, val, fold ) ;
    fold = g.shfl_down ( val, 1) ;
    GB_ADD ( val, val, fold ) ;
#else
    // use shared memory; do not use shfl_down
    #error "not implemented yet"
#endif
    return (val) ; // note: only thread 0 will return full val
}
