function gbtest29
%GBTEST29 test logical indexing

rng ('default') ;
for trial = 1:40
    fprintf ('.')
    for m = 0:5
        for n = 0:5
            A = sprand (m, n, 0.5) ;
            C = sprand (m, n, 0.5) ;
            M = sprand (m, n, 0.5) ~= 0 ;
            G = gb (A) ;

            x1 = A (M) ;
            x2 = G (M) ;

            % In MATLAB, if A and M are row vectors then A(M) is also a
            % row vector.  That contradicts this blog post:
            % https://www.mathworks.com/company/newsletters/articles/matrix-indexing-in-matlab.html
            % and also this documentation:
            % https://www.mathworks.com/help/matlab/math/array-indexing.html
            % Both those pages state that A(M) where M is a logical matrix
            % always returns a column vector.  GraphBLAS always returns a
            % column vector.  So x1(:) and x2(:) are compared below:
            assert (isequal (x1 (:), double (x2 (:)))) ;

            % Both GraphBLAS and MATLAB can accept either row or column vectors
            % x on input to C (M) = x.  MATLAB treats this as C(M)=x(:), and so
            % does GraphBLAS.  So the subsasgn is in agreement between MATLAB
            % and GraphBLAS, and all these uses work:

            C1 = C ;
            C1 (M) = A (M) ;        % C1(M) is MATLAB, A(M) is MATLAB

            C2 = gb (C) ;
            C2 (M) = A (M) ;        % C2(M) is gb, A(M) is MATLAB

            C3 = gb (C) ;
            C3 (M) = G (M) ;        % C3(M) is gb, and G(M) is gb

            % this uses the MATLAB subasgn, after typecasting G(M) from class
            % gb to class double, using the double(G) function in gb.m:
            C4 = C ;
            C4 (M) = G (M) ;

            assert (isequal (C1, double (C2))) ;
            assert (isequal (C1, double (C3))) ;
            assert (isequal (C1, C4)) ;

            % also try with a gb mask matrix M.  In this case, A(M) where A
            % is a MATLAB matrix and M is a gb logical matrix fails.
            % the subsref function in gb.m can handle this case, but MATLAB
            % doesn't call it.  It tries using the built-in subsref instead,
            % and it doesn't know what to do with a gb logical matrix M.

            M = gb (M) ;
            C5 = gb (C) ;
            C5 (M) = G (M) ;

            assert (isequal (C1, double (C5))) ;

            % test scalar assigment with logical indexing
            K = logical (M) ;
            C1 (K) = pi ;
            C2 (M) = pi ;
            C3 (M) = gb (pi) ;
            C4 (K) = gb (pi) ;
            C5 (M) = pi ;

            assert (isequal (C1, double (C2))) ;
            assert (isequal (C1, double (C3))) ;
            assert (isequal (C1, C4)) ;
            assert (isequal (C1, double (C5))) ;

        end
    end
end

fprintf ('\ngbtest29: all tests passed\n') ;

