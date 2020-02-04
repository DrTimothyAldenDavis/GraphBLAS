function [c times best] = tric (A, cgood)
% A must be logical, symmetric, and stored by row

assert (GrB.isbyrow (A)) ;
assert (isequal (GrB.type (A), 'logical')) ;

rng ('default') ;

desc_s.mask = 'structural' ;

desc_st.mask = 'structural' ;
desc_st.in1 = 'transpose' ;

semiring = '+.pair.int64' ;
monoid = '+.int64' ;

n = size (A,1) ;
Z = GrB (n, n, 'int64', 'by row') ;

if (nargin < 2)
    tstart = tic ;
    cgood = GrB.tricount (A) ;
    tgood = toc (tstart) ;
    fprintf ('tricount time: %g   triangles %d\n', tgood, cgood) ;
end

degree = full (double (GrB.entries (A, 'row', 'degree'))) ;
fprintf ('degree: min: %d max: %d mean: %g std: %g\n', ...
    min (degree), max (degree), mean (degree), std (degree)) ;

times = inf (12, 1) ;

dot = [3 4 7 8 11 12] ;
for trial = dot % 1:12

    tstart = tic ;
    c = -1 ;

    try

        if (trial == 1)

            % Sandia method: C<L>=L*L with saxpy method
            L = tril (A, -1) ;
            C = GrB.mxm (Z, L, semiring, L, L, desc_s) ;

        elseif (trial == 2)

            % Sandia2 method: C<U>=U*U with saxpy method
            U = triu (A, 1) ;
            C = GrB.mxm (Z, U, semiring, U, U, desc_s) ;

        elseif (trial == 3)

            % SandiaDot: C<L>=L*U': dot method
            L = tril (A, -1) ;
            U = triu (A, 1) ;
            C = GrB.mxm (Z, L, semiring, L, U, desc_st) ;

        elseif (trial == 4)

            % SandiaDot2: C<U>=U*L': dot method
            L = tril (A, -1) ;
            U = triu (A, 1) ;
            C = GrB.mxm (Z, U, semiring, U, L, desc_st) ;



        elseif (trial == 5)

            % sort degree, low to hi: saxpy method (Sandia)
            [~,p] = sort (degree, 'ascend') ;
            L = tril (A (p,p), -1) ;
            C = GrB.mxm (Z, L, semiring, L, L, desc_s) ;

        elseif (trial == 6)

            % sort degree, hi to low: saxpy method (Sandia)
            [~,p] = sort (degree, 'descend') ;
            L = tril (A (p,p), -1) ;
            C = GrB.mxm (Z, L, semiring, L, L, desc_s) ;

        elseif (trial == 7)

            % sort degree, low to hi: dot method (SandiaDot)
            [~,p] = sort (degree, 'ascend') ;
            S = A (p,p) ;
            L = tril (S, -1) ;
            U = triu (S, 1) ;
            C = GrB.mxm (Z, L, semiring, L, U, desc_st) ;

        elseif (trial == 8)

            % sort degree, hi to low: dot method (SandiaDot)
            [~,p] = sort (degree, 'descend') ;
            S = A (p,p) ;
            L = tril (S, -1) ;
            U = triu (S, 1) ;
            C = GrB.mxm (Z, L, semiring, L, U, desc_st) ;



        elseif (trial == 9)

            % sort degree, low to hi: saxpy method
            [~,p] = sort (degree, 'ascend') ;
            [i j ~] = find (A) ;
            % if p = 1:n, the rule is i > j, which is tril (A)
            keep = p(i) > p(j) ;
            i = i (keep) ;
            j = j (keep) ;
            S = GrB.build (i, j, 1, n, n, '|', 'logical') ;
            C = GrB.mxm (Z, S, semiring, S, S, desc_s) ;

        elseif (trial == 10)

            % sort degree, hi to low: saxpy method
            [~,p] = sort (degree, 'descend') ;
            [i j ~] = find (A) ;
            % if p = 1:n, the rule is i > j, which is tril (A)
            keep = p(i) > p(j) ;
            i = i (keep) ;
            j = j (keep) ;
            S = GrB.build (i, j, 1, n, n, '|', 'logical') ;
            clear keep i j
            C = GrB.mxm (Z, S, semiring, S, S, desc_s) ;

        elseif (trial == 11)

            % sort degree, low to hi: dot method
            [~,p] = sort (degree, 'ascend') ;
            [i j ~] = find (A) ;
            % if p = 1:n, the rule is i > j, which is tril (A)
            keep = p(i) > p(j) ;
            ilo = i (keep) ;
            jlo = j (keep) ;
            L = GrB.build (ilo, jlo, 1, n, n, '|', 'logical') ;
            keep = p(i) < p(j) ;
            ihi = i (keep) ;
            jhi = j (keep) ;
            U = GrB.build (ihi, jhi, 1, n, n, '|', 'logical') ;
            clear keep i j ilo jlo ihi jhi
            C = GrB.mxm (Z, L, semiring, L, U, desc_st) ;

        elseif (trial == 12)

            % sort degree, hi to low: dot method
            [~,p] = sort (degree, 'descend') ;
            [i j ~] = find (A) ;
            % if p = 1:n, the rule is i > j, which is tril (A)
            keep = p(i) > p(j) ;
            ilo = i (keep) ;
            jlo = j (keep) ;
            L = GrB.build (ilo, jlo, 1, n, n, '|', 'logical') ;
            keep = p(i) < p(j) ;
            ihi = i (keep) ;
            jhi = j (keep) ;
            U = GrB.build (ihi, jhi, 1, n, n, '|', 'logical') ;
            clear keep i j ilo jlo ihi jhi
            C = GrB.mxm (Z, L, semiring, L, U, desc_st) ;

        end

        c = full (double (GrB.reduce (monoid, C))) ;

    catch me
        fprintf ('error: %s\n', me.message) ;
    end

    t = toc (tstart) ;
    if (c == -1)
        t = inf ;
    else
        assert (c == cgood)
    end

    times (trial) = t ;
    fprintf ('%2d: %10.4f\n', trial, t) ;
    clear C S L U ilo jlo ihi jhi keep i j p

end

fprintf ('\n') ;
[tbest, best] = min (times) ;
best = best (1) ;

for trial = 1:12
    t = times (trial) ;
    fprintf ('%2d: %10.4f relative: %10.4f ', trial, t, t / tbest) ;
    if (trial == best)
        fprintf ('best') ;
    end
    fprintf ('\n') ;
end

fprintf ('\n') ;

