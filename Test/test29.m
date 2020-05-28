function test29
%TEST29 GrB_reduce with zombies

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

[~, ~, add_ops, types, ~, ~] = GB_spec_opsall ;
types = types.all ;

fprintf ('\ntest29: ----------------- GrB_reduce with zombies\n') ;

for m = [1 5 10]
    for n = [1 5 10]

        rng ('default') ;

        for k3 = 1:length (types)
            atype = types {k3}  ;

            clear C
            C.matrix = 100 * sparse (rand (m,n)) ;
            C.class = atype ;
            C.pattern = logical (spones (C.matrix)) ;

            for A_is_hyper = 0:1
            for A_is_csc   = 0:1

            A = GB_spec_random (m,n,0.1,100,atype, A_is_csc, A_is_hyper) ;

            if (isequal (atype, 'logical'))
                ops = {'or', 'and', 'xor', 'eq'} ;
            else
                ops = {'min', 'max', 'plus', 'times'} ;
            end

            fprintf ('.') ;
            for kk4 = 1:length(add_ops)

                op = ops {kk4} ;
                try
                    GB_spec_operator (op, atype) ;
                    cin = GB_spec_identity (op, atype) ;
                catch
                    continue
                end

                [C3,c1,c3] = GB_mex_subassign (C, [ ], [ ], A, ...
                    [ ], [ ], [ ], op) ;
                c2 = GB_mex_reduce_to_scalar (cin, '', op, C3) ;

                if (isfloat (c1))
                    assert (isequal (c1,c2) || ...
                        (abs (c1-c2) <= 8 * eps (c2)))  ;
                else
                    assert (isequal (c1,c2))
                end

                op_plus.opname = 'plus' ;
                op_plus.optype = 'double' ;
                c4 = GB_mex_reduce_to_scalar (0, '', op_plus, C3) ;

                if (isfloat (c3))
                    assert (isequal (c3,c4) || ...
                        (abs (c3-c4) <= 8 * eps (c4)))  ;
                else
                    assert (isequal (c3,c4))
                end


            end
            end
            end
        end
    end
end

fprintf ('\ntest29: all tests passed\n') ;

