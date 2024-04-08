#include "GraphBLAS_cuda.hpp"
#include "GB_cuda.hpp"

bool GB_cuda_apply_binop_branch
(
    const GrB_Type ctype,
    const GrB_BinaryOp op,
    const GrB_Matrix A,
    const bool bind1st,
)
{
    bool ok = GB_cuda_type_branch (ctype) && GB_cuda_type_branch (A->type) ;
    if (bind1st)
    {
        ok = ok && GB_cuda_type_branch (op->xtype) ;
    }
    else
    {
        ok = ok && GB_cuda_type_branch (op->ytype) ;
    }

    if (!ok)
    {
        return false;
    }
    return true;
}
