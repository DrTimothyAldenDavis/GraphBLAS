
% test103
load gunk
% save gunk C M A desc
                    C2a = GB_spec_transpose (C, M, 'plus', A, desc) ;
                    C2b = GB_mex_transpose  (C, M, 'plus', A, desc, 'test') ;
                    GB_spec_compare (C2a, C2b) ;

