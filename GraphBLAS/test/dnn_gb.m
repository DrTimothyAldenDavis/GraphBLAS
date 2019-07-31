function Y = dnn_gb (W, bias, Y0)
% Performs ReLU inference using input feature vector(s) Y0, DNN weights W,
% and bias vectors.

% set # of threads to use in GraphBLAS
% gb.threads (1) ;  % or 40, or whatever.  Default is omp_get_max_threads( ).

n = size (Y0, 2) ;

% convert to GraphBLAS (this is optional, gb.* methods can take inputs
% that are either gb objects or MATLAB sparse matrices)
Y0 = gb (Y0, 'single') ;
for i=1:length(W)
    W {i} = gb (W {i}, 'single') ;
    % build a gb GrB_FP32 matrix from tuples, using '+' as the dup operator
    bias {i} = gb.build (1:n, 1:n, bias {i}, n, n, '+', 'single') ;
end

% TODO allow thunk to be a full scalar instead of a sparse one
thunk = sparse (32) ;

Y = Y0 ; % Initialize feature vectors.
for i=1:length(W) % Loop through each weight layer W{i}

    % Propagate through layer.
    Y = gb.mxm ('+.*', Y, W {i}) ;

    % Apply bias to non-zero entries:
    Y = gb.mxm ('+.+', Y, bias {i}) ;

    % Threshold negative values.
    Y = gb.select ('>=0', Y) ;

    % Threshold maximum values.
    M = gb.select ('>thunk', Y, thunk) ;

    if (0) % if done in GraphBLAS, it will look like this:
        % Y = gb.assign (Y, M, thunk) ;
    else
        % (not yet implemented in the GraphBLAS interface ...)
        % first convert the gb objects to MATLAB
        Y_matlab = sparse (Y) ;
        M_matlab = logical (sparse (M)) ;
        % do the work in MATLAB
        Y_matlab (M_matlab) = 32 ;
        % convert back to gb object
        Y = gb (Y_matlab, 'single') ;
    end
end

% convert result back to MATLAB sparse matrix (also optional)
Y = sparse (Y) ;

