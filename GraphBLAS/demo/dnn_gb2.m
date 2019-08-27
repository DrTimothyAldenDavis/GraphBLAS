function Y = dnn_gb (W, bias, Y0)
Y = Y0 ;
for i=1:length(W)
    % Propagate through layer, apply bias, and threshold negative values.
    Y = gb.select ('>0', gb.mxm ('+.+', Y * W {i}, bias {i})) ;
    % Threshold maximum values.
    Y (Y > 32) = 32 ;
end

