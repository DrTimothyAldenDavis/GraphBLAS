function iset = mis_gb (A)
%MIS_GB variant of Luby's maximal independent set algorithm, using GraphBLAS
%
% Given a logical n x n adjacency matrix A of an unweighted and undirected
% graph (where the value true represents an edge), compute a maximal set of
% independent nodes and return it in a boolean n-vector, 'iset' where iset(i)
% of true implies node i is a member of the set.
%
% The graph cannot have any self edges, and it must be symmetric.  These
% conditions are not checked.  Self-edges will cause the method to stall.
%
% Singletons require special treatment.  Since they have no neighbors, their
% prob is never greater than the max of their neighbors, so they never get
% selected and cause the method to stall.  To avoid this case they are removed
% from the candidate set at the begining, and added to the iset.
%
% [Luby 1985] TODO cite

[m n] = size (A) ;
if (m ~= n)
    error ('A must be square') ;
end

prob = gb (n, 1) ;
neighbor_max = gb (n, 1) ;
new_members = gb (n, 1, 'logical') ;
new_neighbors = gb (n, 1, 'logical') ;
candidates = gb (n, 1, 'logical') ;

% Initialize independent set vector
iset = gb (n, 1, 'logical') ;

% descriptor: C_replace
r_desc.out = 'replace' ;

% descriptor: C_replace + structural complement of mask
sr_desc.mask = 'complement' ;
sr_desc.out  = 'replace' ;

% create the mis_score binary operator
% GrB_BinaryOp_new (&set_random, mis_score2, GrB_FP64, GrB_UINT32, GrB_FP64) ;

% compute the degree of each nodes
degrees = gb.vreduce ('+.double',  A) ;

% singletons are not candidates; they are added to iset first instead
% candidates[degree != 0] = 1
candidates = gb.assign (candidates, degrees, true) ;

% add all singletons to iset
% iset[degree == 0] = 1
iset = gb.assign (iset, degrees, true, sr_desc) ; 

% Iterate while there are candidates to check.
nvals = gb.nvals (candidates) ;
last_nvals = nvals ;
kk = 0 ;

while (nvals > 0)

    % compute a random probability scaled by inverse of degree
    % NOTE: this is slower than it should be; rand may not be parallel,
    % See GraphBLAS/Demo/Source/mis.c and the prand_* functions.
    prob = 0.0001 + rand (n,1) ./ (1 + 2 * degrees) ;
    prob = gb.assign (prob, candidates, prob, r_desc) ;

    % compute the max probability of all neighbors
    neighbor_max = gb.mxm (neighbor_max, candidates, ...
        'max.second.double', A, prob, r_desc) ;

    % select node if its probability is > than all its active neighbors
    new_members = gb.eadd ('>', prob, neighbor_max) ;

    % add new members to independent set.
    iset = gb.eadd ('|', iset, new_members) ;

    % remove new members from set of candidates
    candidates = gb.apply (candidates, new_members, 'identity', ...
        candidates, sr_desc) ;

    nvals = gb.nvals (candidates) ;
    if (nvals == 0)
        break ;                    % early exit condition
    end

    % Neighbors of new members can also be removed from candidates
    new_neighbors = gb.mxm (new_neighbors, candidates, ...
        '|.&.logical', A, new_members) ;

    candidates = gb.apply (candidates, new_neighbors, 'identity', ...
        candidates, sr_desc) ;

    nvals = gb.nvals (candidates) ;

    % this will not occur, unless the input is corrupted somehow
    if (last_nvals == nvals)
        error ('stall!\n') ;
    end
    last_nvals = nvals ;
end

% drop explicit false values
iset = gb.apply (iset, iset, 'identity', iset, r_desc) ;

