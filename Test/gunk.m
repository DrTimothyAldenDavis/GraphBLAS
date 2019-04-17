
% test103
% save gunk C M A desc
clear
load gunk

                    C2a = GB_spec_transpose (C, M, 'plus', A, desc) ;
                    C2b = GB_mex_transpose  (C, M, 'plus', A, desc, 'test') ;
                    GB_spec_compare (C2a, C2b) ;
%{
% test103
% save gunk C M A desc
clear
load gunk
                    C2a = GB_spec_transpose (C, M, 'plus', A, desc) ;
                    C2b = GB_mex_transpose  (C, M, 'plus', A, desc, 'test') ;
                    GB_spec_compare (C2a, C2b) ;

% test14
% save gunk cin op A A_hack
clear
load gunk
        c1 = GB_spec_reduce_to_scalar (cin, [ ], op, A_hack) ;
        c2 = GB_mex_reduce_to_scalar  (cin, [ ], op, A) ;
    c1
    c2
    c1-c2
            abs (c1-c2)
            eps (A.class) *  norm (A.matrix,1)
            abs (c1-c2) < 4 * eps (A.class) *  (abs(c1) + 1)
%        assert (isequal (c1, c2)) ;

%test18
clear
load gunk
% C = A+B with mask
%save gunk C Mask accum op A B dnn
C0 = GB_spec_eWiseAdd_Matrix (C, Mask, accum, op, A, B, dnn);
C1 = GB_mex_eWiseAdd_Matrix  (C, Mask, accum, op, A, B, dnn);
GB_spec_compare (C0, C1) ;

% C = A+B with mask
% save gunk C Mask accum op A B dnn
clear
load gunk
C0 = GB_spec_eWiseAdd_Matrix (C, Mask, accum, op, A, B, dnn);
C1 = GB_mex_eWiseAdd_Matrix  (C, Mask, accum, op, A, B, dnn);
GB_spec_compare (C0, C1) ;

% test18
% C = A+B with mask
% save gunk C Mask accum op A B dnn
clear
C0 = GB_spec_eWiseAdd_Matrix (C, Mask, accum, op, A, B, dnn);
C1 = GB_mex_eWiseAdd_Matrix  (C, Mask, accum, op, A, B, dnn);
GB_spec_compare (C0, C1) ;

% test103
load gunk
% save gunk C M A desc
                    C2a = GB_spec_transpose (C, M, 'plus', A, desc) ;
                    C2b = GB_mex_transpose  (C, M, 'plus', A, desc, 'test') ;
                    GB_spec_compare (C2a, C2b) ;

%}
