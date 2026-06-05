function [x,p] = gbtest_argminmax (A, ismin, dim)
%GBTEST_ARGMINMAX simple computation of argmin and argmax

S = spones (A) ;
[m n] = size (A) ;
type = GrB.type (A) ;

if (dim == 2)

    [x,p] = gbtest_argminmax (A', ismin, 1) ;

elseif (dim == 1)

    x = GrB (n, 1, type) ;
    p = GrB (n, 1, 'int64') ;
    for j = 1:n
        first = true ;
        for i = 1:m
            if (S (i,j) == 1)
                if (first)
                    x (j) = A (i,j) ;
                    p (j) = i ;
                    first = false ;
                else
                    if (ismin)
                        if (A (i,j) < x (j))
                            x (j) = A (i,j) ;
                            p (j) = i ;
                        end
                    else
                        if (A (i,j) > x (j))
                            x (j) = A (i,j) ;
                            p (j) = i ;
                        end
                    end
                end
            end
        end
    end

else % dim == 0

    first = true ;
    for i = 1:m
        for j = 1:m
            if (S (i,j) == 1)
                if (first)
                    x = A (i,j) ;
                    p = [i j] ;
                    first = false ;
                else
                    if (ismin)
                        if (A (i,j) < x)
                            x = A (i,j) ;
                            p = [i j] ;
                        end
                    else
                        if (A (i,j) > x)
                            x = A (i,j) ;
                            p = [i j] ;
                        end
                    end
                end
            end
        end
    end
    p = GrB (p', 'int64') ;

end

