

#if 0
GrB_Info GB_subassign_jit
(
    // input/output:
    GrB_Matrix C,
    // input:
    const bool C_replace,
    const GrB_Index *I,
    const int64_t ni,
    const int64_t nI,
    const int Ikind,
    const int64_t Icolon [3],
    const GrB_Index *J,
    const int64_t nj,
    const int64_t nJ,
    const int Jkind,
    const int64_t Jcolon [3],
    const GrB_Matrix M,
    const bool Mask_comp,
    const bool Mask_struct,
    const GrB_BinaryOp accum,
    const GrB_Matrix A,
    const void *scalar,
    const GrB_Type scalar_type,
    const int assign_kind,
    const int assign_kernel,
    GB_Werk Werk
)
{

}

#endif
