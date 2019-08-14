function gbtest5
%TEST5 test gb.descriptorinfo

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

list_out  = { [ ], 'default', 'replace' } ;
list_in   = { [ ], 'default', 'transpose' } ;
list_mask = { [ ], 'default', 'complement' } ;
list_axb  = { [ ], 'default', 'gustavson', 'heap', 'dot' } ;

ntrials = 0;

d = struct

for k1 = 1:length (list_out)
    out = list_out {k1} ;
    for k2 = 1:length (list_in)
        in0 = list_in {k2} ;
        for k3 = 1:length (list_in)
            in1 = list_in {k3} ;
            for k4 = 1:length (list_mask)
                mask = list_mask {k4} ;
                for k5 = 1:length (list_axb)
                    axb = list_axb {k5} ;

                    for nthreads = [0 2]
                        for chunk = [0 10000]

                            clear d
                            d = struct ;

                            if (~isempty (out))
                                d.out = out ;
                            end

                            if (~isempty (mask))
                                d.mask = mask ;
                            end

                            if (~isempty (in0))
                                d.in0 = in0 ;
                            end

                            if (~isempty (in1))
                                d.in1 = in1 ;
                            end

                            if (~isempty (axb))
                                d.axb = axb ;
                            end

                            if (nthreads > 0)
                                d.nthreads = nthreads ;
                            end

                            if (chunk > 0)
                                d.chunk = chunk ;
                            end

                            d
                            gb.descriptorinfo (d) ;
                            ntrials = ntrials + 1 ;
                        end
                    end
                end
            end
        end
    end
end

fprintf ('test error handling (errors expected):\n') ;

clear d
d.in0 = 'crud'
try
    gb.descriptorinfo (d) ;
    ok = false ;
catch me
    ok = true ;
    me
end
assert (ok) ;

clear d
d.in0 = 42
try
    gb.descriptorinfo (d) ;
    ok = false ;
catch me
    ok = true ;
    me
end
assert (ok) ;

clear d
d.nthreads = 'something'
try
    gb.descriptorinfo (d) ;
    ok = false ;
catch me
    ok = true ;
    me
end
assert (ok) ;

clear d
d.chunk = 'something'
try
    gb.descriptorinfo (d) ;
    ok = false ;
catch me
    ok = true ;
    me
end
assert (ok) ;

gb.descriptorinfo

ntrials
fprintf ('gbtest5: all tests passed\n') ;

