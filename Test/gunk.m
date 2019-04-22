
clear all
load gunk
% test18
% here = 13 ; save gunk C Mask accum op A B dnn here
                                C0 = GB_spec_eWiseMult_Matrix ...
                                    (C, Mask, accum, op, A, B, dnn);
                                C1 = GB_mex_eWiseMult_Matrix ...
                                    (C, Mask, accum, op, A, B, dnn);
                                GB_spec_compare (C0, C1) ;


