
clear
% tricount (gbgraph (1))
% gb.threads (8)
gb.format ('by col')

probs = [2662 2459]

for id = probs
    Prob = ssget (id) 

    A = gb (Prob.A, 'logical', 'by row') ;
    % A = gb (Prob.A, 'logical') ;
    H = gbgraph (A | A') ;

    tic ;
    s1 = tricount (H) ;
    t = toc ;
    fprintf ('s1 %d time %d (original)\n', s1, t) ;

    %{
    p = colperm (double (H)) ;
    H2 = reordernodes (H, p) ;
    tic ;
    s2 = tricount (H) ;
    t = toc ;
    fprintf ('s2 %d time %d (colperm)\n', s2, t) ;
    assert (s1 == s2) ;

    H2 = reordernodes (H, p (end:-1:1)) ;
    tic ;
    s2 = tricount (H) ;
    t = toc ;
    fprintf ('s2 %d time %d (reverse colperm)\n', s2, t) ;
    assert (s1 == s2) ;

    [~,p] = etree (H) ;
    H2 = reordernodes (H, p) ;
    tic ;
    s2 = tricount (H) ;
    t = toc ;
    fprintf ('s2 %d time %d (etree)\n', s2, t) ;
    assert (s1 == s2) ;

    p = amd (H) ;
    H2 = reordernodes (H, p) ;
    tic ;
    s2 = tricount (H) ;
    t = toc ;
    fprintf ('s2 %d time %d (amd)\n', s2, t) ;
    assert (s1 == s2) ;

    p = symrcm (H) ;
    H2 = reordernodes (H, p) ;
    tic ;
    s2 = tricount (H) ;
    t = toc ;
    fprintf ('s2 %d time %d (symrcm)\n', s2, t) ;
    assert (s1 == s2) ;
    %}

end

