// semiring: (plus, pair, int64_t)

// monoid:
#define GB_Z_TYPE int64_t
#define GB_ADD(z,x,y) z = (x) + (y)
#define GB_UPDATE(z,y) z += y
#define GB_DECLARE_IDENTITY(z) int64_t z = 0
#define GB_DECLARE_IDENTITY_CONST(z) const int64_t z = 0
#define GB_HAS_IDENTITY_BYTE 1
#define GB_IDENTITY_BYTE 0x00
#define GB_PRAGMA_SIMD_REDUCTION_MONOID(z) GB_PRAGMA_SIMD_REDUCTION (+,z)
#define GB_Z_IGNORE_OVERFLOW 1
#define GB_Z_NBITS 64
#define GB_Z_ATOMIC_BITS 64
#define GB_Z_HAS_ATOMIC_UPDATE 1
#define GB_Z_HAS_OMP_ATOMIC_UPDATE 1
#define GB_Z_HAS_CUDA_ATOMIC_BUILTIN 1
#define GB_Z_CUDA_ATOMIC GB_cuda_atomic_add
#define GB_Z_CUDA_ATOMIC_TYPE int64_t

// multiplicative operator:
#define GB_MULT(z,x,y,i,k,j) z = 1

// multiply-add operator:
#define GB_MULTADD(z,x,y,i,k,j) z += 1

// special cases:
#define GB_IS_PLUS_PAIR_REAL_SEMIRING 1
#define GB_IS_PLUS_BIG_PAIR_SEMIRING 1
#define GB_IS_PAIR_MULTIPLIER 1

// C matrix: sparse
#define GB_C_IS_HYPER  0
#define GB_C_IS_SPARSE 1
#define GB_C_IS_BITMAP 0
#define GB_C_IS_FULL   0
#define GBP_C(Cp,k,vlen) Cp [k]
#define GBH_C(Ch,k)      (k)
#define GBI_C(Ci,p,vlen) Ci [p]
#define GBB_C(Cb,p)      1
#define GB_C_ISO 0
#define GB_C_IN_ISO 0
#define GB_C_TYPE int64_t
#define GB_PUTC(c,Cx,p) Cx [p] = c

// M matrix: sparse
#define GB_M_IS_HYPER  0
#define GB_M_IS_SPARSE 1
#define GB_M_IS_BITMAP 0
#define GB_M_IS_FULL   0
#define GBP_M(Mp,k,vlen) Mp [k]
#define GBH_M(Mh,k)      (k)
#define GBI_M(Mi,p,vlen) Mi [p]
#define GBB_M(Mb,p)      1
// structural mask:
#define GB_M_TYPE void
#define GB_MCAST(Mx,p,msize) 1
#define GB_MASK_STRUCT 1
#define GB_MASK_COMP   0
#define GB_NO_MASK     0
#define GB_MASK_SPARSE_STRUCTURAL_AND_NOT_COMPLEMENTED

// A matrix: sparse
#define GB_A_IS_HYPER  0
#define GB_A_IS_SPARSE 1
#define GB_A_IS_BITMAP 0
#define GB_A_IS_FULL   0
#define GBP_A(Ap,k,vlen) Ap [k]
#define GBH_A(Ah,k)      (k)
#define GBI_A(Ai,p,vlen) Ai [p]
#define GBB_A(Ab,p)      1
#define GB_A_ISO 1
#define GB_A_IS_PATTERN 1
#define GB_A_TYPE void
#define GB_A2TYPE void
#define GB_DECLAREA(a)
#define GB_GETA(a,Ax,p,iso)

// B matrix: sparse
#define GB_B_IS_HYPER  0
#define GB_B_IS_SPARSE 1
#define GB_B_IS_BITMAP 0
#define GB_B_IS_FULL   0
#define GBP_B(Bp,k,vlen) Bp [k]
#define GBH_B(Bh,k)      (k)
#define GBI_B(Bi,p,vlen) Bi [p]
#define GBB_B(Bb,p)      1
#define GB_B_ISO 1
#define GB_B_IS_PATTERN 1
#define GB_B_TYPE void
#define GB_B2TYPE void
#define GB_DECLAREB(b)
#define GB_GETB(b,Bx,p,iso)
