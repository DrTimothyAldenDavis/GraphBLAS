function test128
%TEST128 test eWiseMult, different cases in emult_phase0

% C = GB_mex_eWiseMult_Matrix (C, Mask, accum, mult, A, B, desc)

fprintf ('\ntest128: test eWiseMult\n') ;
rng ('default') ;

m = 100 ;
n = 100 ;

Mmat = sparse (m,n) ;
Mmat (1,:) = 1 ;

Amat = sparse (rand (m,n)) ;
Amat (:,2) = 0 ;
Amat (:,5) = 0 ;
Amat (1,5) = 1 ;

Bmat = sparse (rand (m,n)) ;
Bmat (:, 3:4) = 0 ;
Bmat (:, 6) = 0 ;
Bmat (1, 6) = 1 ;

clear M
M.matrix  = Mmat ;
M.pattern = logical (spones (Mmat)) ;
M.class = 'logical' ;

clear A
A.matrix  = Amat ;
A.pattern = logical (spones (Amat)) ;
A.class = 'double' ;

clear B
B.matrix  = Bmat ; 
B.pattern = logical (spones (Bmat)) ;
B.class = 'double' ;

S = sparse (m,n) ;
X = sparse (rand (m,n)) ;

for M_hyper = 0:1
    for B_hyper = 0:1
        for A_hyper = 0:1
            M.is_hyper = M_hyper ;
            A.is_hyper = A_hyper ;
            B.is_hyper = B_hyper ;
            C0 = Amat .* Bmat .* Mmat ;
            C1 = GB_spec_eWiseMult_Matrix (S, M, [ ], 'times', A, B, [ ]) ;
            C2 = GB_mex_eWiseMult_Matrix  (S, M, [ ], 'times', A, B, [ ]) ;
            C3 = GB_mex_eWiseMult_Matrix  (S, M, [ ], 'times', B, A, [ ]) ;
            GB_spec_compare (C1, C2) ;
            GB_spec_compare (C1, C3) ;
            assert (isequal (C0, C2.matrix)) ;

            C1 = GB_spec_eWiseMult_Matrix (X, M, [ ], 'times', A, B, [ ]) ;
            C2 = GB_mex_eWiseMult_Matrix  (X, M, [ ], 'times', A, B, [ ]) ;
            GB_spec_compare (C1, C2) ;
        end
    end
end

fprintf ('test128: all tests passed\n') ;
