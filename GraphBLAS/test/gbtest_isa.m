function s = gbtest_isa (ghb, G)
switch (ghb)
    case 0
        s = (isa (G, 'GrB')) ;
    case 1
        s = (isa (G, 'GhB')) ;
    otherwise
        s = (isa (G, 'GhB') || isa (G, 'GrB')) ;
end

