classdef graphblas
%GRAPHBLAS a GraphBLAS sparse matrix object

properties % (SetAccess = protected)
    opaque = [ ] ;
end

    methods

        function G = graphblas (varargin)
        G.opaque = gb (varargin {:}) ;
        end

    end

    methods (Static)

        function stuff = ghe (gotcha)
        stuff = gotcha + 1 ;
        end

    end
end
