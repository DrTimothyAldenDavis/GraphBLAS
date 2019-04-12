
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
%{
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
