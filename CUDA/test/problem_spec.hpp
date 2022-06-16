#pragma once

#include <cassert>
#include <cmath>
#include <random>
#include <algorithm>
#include <iostream>
//#include "GB_binary_search.h"
#include "GpuTimer.h"
#include "GB_cuda_buckets.h"
#include "../../rmm_wrap/rmm_wrap.h"
#include <gtest/gtest.h>
#include "test_data.hpp"
extern "C" {
#include "GB.h"
}

#include "../jitFactory.hpp"
#include "dataFactory.hpp"

template<typename T_C, typename T_M, typename T_A, typename T_B>
class mxm_problem_spec {

public:
    mxm_problem_spec(GrB_Monoid monoid, GrB_BinaryOp binop, int64_t N, int TB) :
        mymxmfactory(GB_cuda_mxm_factory ( )), mysemiring(), G(N, N), Annz(N*N), Bnnz(N*N),
        mask_struct(false) {

        // FIXME: This should be getting set automatically somehow.
        bool flipxy = false;
        bool mask_comp = false;

        switch(TB) {
            case GB_BUCKET_VSSP:
                Annz = N * 2;
                Bnnz = N * 10;
                break;
            case GB_BUCKET_VSVS:
                Annz = N * 2;
                Bnnz = N * 4;
                break;
            case GB_BUCKET_MERGEPATH:
                Annz = N * 5;
                Bnnz = N * 2;
                break;
            default:
                printf("Bucket not yet being tested!\\n");
            exit(1);
        }

        Cnz = N;
        float Cnzpercent = (float) Cnz/(N*N);

        // TODO: Allocate and fill arrays for buckets and nano buckets
        G.init_A(Annz, GxB_SPARSE, GxB_BY_ROW);
        G.init_B(Bnnz, GxB_FULL, GxB_BY_ROW);
        G.init_C(Cnzpercent);
        G.fill_buckets( TB ); // all elements go to testbucket= TB

        GrB_Matrix C = G.getC();
        GrB_Matrix M = G.getM();
        GrB_Matrix A = G.getA();
        GrB_Matrix B = G.getB();

        /************************
         * Create mxm factory
         */
        auto grb_info = GrB_Semiring_new(&mysemiring, monoid, binop);
        GRB_TRY (grb_info) ;

        bool C_iso = false ;
        int C_sparsity = GB_sparsity (M) ;
        GrB_Type ctype = binop->ztype ;

        mymxmfactory.mxm_factory (
                C_iso, C_sparsity, ctype,
                M, mask_struct, mask_comp,
                mysemiring, flipxy,
                A, B) ;
    }

    ~mxm_problem_spec() {
        G.del();
    }

    GrB_Matrix getC(){ return G.getC(); }
    GrB_Matrix getM(){ return G.getM(); }
    GrB_Matrix getA(){ return G.getA(); }
    GrB_Matrix getB(){ return G.getB(); }

    auto getG() { return G; }

    int64_t getCnz() { return Cnz; }
    GB_cuda_mxm_factory get_mxm_factory() { return mymxmfactory; }
    GrB_Semiring get_semiring() { return mysemiring; }

    bool get_mask_struct() { return mask_struct; }

private:

    bool mask_struct;
    int64_t Annz;
    int64_t Bnnz;
    int64_t Cnz;
    GrB_Semiring  mysemiring;
    GB_cuda_mxm_factory mymxmfactory;
    SpGEMM_problem_generator<T_C, T_M, T_A, T_B> G;
};