#include "GraphBLAS_cuda.hpp"
#include "GB_cuda.hpp"

bool GB_cuda_apply_unop_branch
(
    const GrB_Type ctype,
    const GrB_Matrix A
)
{
    bool ok = GB_cuda_type_branch (ctype) && GB_cuda_type_branch (A->type) ;

    if (!ok)
    {
        return false;
    }
    return true;
}