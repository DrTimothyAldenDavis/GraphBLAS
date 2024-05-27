function gbmake (what)
%GBMAKE compile @GrB interface for SuiteSparse:GraphBLAS
%
% Usage:
%   gbmake
%
% gbmake compiles the @GrB interface for SuiteSparse:GraphBLAS.  The
% GraphBLAS library must already be compiled and installed.
% MATLAB 9.4 (R2018a) or Octave 7 later is required.
%
% For the Mac, the GraphBLAS library must be installed in /usr/local/lib/ as
% libgraphblas_matlab.dylib.  It cannot be used where it is created in
% ../build, because of the default Mac security settings.  For Unix/Linux, the
% library used is ../build/libgraphblas_matlab.so if found, or in
% /usr/local/lib if not found there.
%
% See also mex, version, GrB.clear.
%
% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('Note: the libgraphblas_matlab dynamic library must already be\n') ;
fprintf ('compiled prior to running this script.\n') ;

have_octave = (exist ('OCTAVE_VERSION', 'builtin') == 5) ;

if (have_octave)
    % Octave can use the normal libgraphblas.so
    need_rename = false ;
    if verLessThan ('octave', '7')
        error ('GrB:mex', 'Octave 7 or later is required') ;
    end
else
    if verLessThan ('matlab', '9.4')
        error ('GrB:mex', 'MATLAB 9.4 (R2018a) or later is required') ;
    end
    % MATLAB 9.10 (R2021a) and following include a built-in GraphBLAS library
    % that conflicts with this version, so rename this version.
    % Earlier versions of MATLAB can use this renamed version too, so
    % for simplicity, use libgraphblas_matlab.so for all MATLAB versions.
    need_rename = true ;
end

if (nargin < 1)
    what = '' ;
end

make_all = (isequal (what, 'all')) ;

% use -R2018a for the new interleaved complex API
flags = '-O -R2018a -DGBNCPUFEAT' ;

if ispc
    % First do the following in GraphBLAS/build, in the Windows console:
    %
    %   cmake ..
    %   devenv graphblas.sln /build "release|x64" /project graphblas
    %
    % The above commands require MS Visual Studio.  The graphblas.lib is
    % compiled and placed in GraphBLAS/build/Release.  Then in the
    % Command Window do:
    %
    %   gbmake
    %
    if (need_rename)
        library = sprintf ('%s/../../build/Release', pwd) ;
    else
        library = sprintf ('%s/../../../build/Release', pwd) ;
    end
else
    % First do one the following in GraphBLAS (use JOBS=n for a parallel
    % build, which is faster):
    %
    %   make
    %   make JOBS=8
    %   sudo make install
    %
    % If you can't do "sudo make install" then add the GraphBLAS/build
    % folder to your LD_LIBRARY_PATH.  Then in this folder in the
    % Command Window do:
    %
    %   gbmake
    %
    if (need_rename)
        library = sprintf ('%s/../../build', pwd) ;
    else
        library = sprintf ('%s/../../../build', pwd) ;
    end
end

if (have_octave)
    % Octave does not have the new MEX classdef object and as of version 7, the
    % mex command doesn't handle compiler options the same way.
    flags = [flags ' -std=c11 -fopenmp -fPIC -Wno-pragmas' ] ;
else
    % remove -ansi from CFLAGS and replace it with -std=c11
    try
        if (strncmp (computer, 'GLNX', 4))
            cc = mex.getCompilerConfigurations ('C', 'Selected') ;
            env = cc.Details.SetEnv ;
            c1 = strfind (env, 'CFLAGS=') ;
            q = strfind (env, '"') ;
            q = q (q > c1) ;
            if (~isempty (c1) && length (q) > 1)
                c2 = q (2) ;
                cflags = env (c1:c2) ;  % the CFLAGS="..." string
                ansi = strfind (cflags, '-ansi') ;
                if (~isempty (ansi))
                    cflags = [cflags(1:ansi-1) '-std=c11' cflags(ansi+5:end)] ;
                    flags = [flags ' ' cflags] ;
                    fprintf ('using -std=c11 instead of default -ansi\n') ;
                end
            end
        end
    catch
    end
    % revise compiler flags for MATLAB
    if (ismac)
        cflags = '' ;
        ldflags = '-fPIC' ;
        rpath = '-rpath ' ;
    elseif (isunix)
        cflags = '-fopenmp' ;
        ldflags = '-fopenmp -fPIC' ;
        rpath = '-rpath=' ;
    end
    if (ismac || isunix)
        rpath = sprintf (' -Wl,%s''''%s'''' ', rpath, library) ;
        flags = [ flags ' CFLAGS=''$CFLAGS ' cflags ' -Wno-pragmas'' '] ;
        flags = [ flags ' CXXFLAGS=''$CXXFLAGS ' cflags ' -Wno-pragmas'' '] ;
        flags = [ flags ' LDFLAGS=''$LDFLAGS ' ldflags rpath ' '' '] ;
    end
end

if ispc
    % Windows
    object_suffix = '.obj' ;
else
    % Linux, Mac
    object_suffix = '.o' ;
end

inc = '-Iutil -I../../../Include -I../../../Source -I../../../Source/Shared ' ;
inc = [inc '-I../../../Source/Template '] ;

    inc = [inc '-I../../.. ' ] ;
    inc = [inc '-I../../../Source/ij ' ] ;
    inc = [inc '-I../../../Source/math ' ] ;
    inc = [inc '-I../../../Source/cast ' ] ;
    inc = [inc '-I../../../Source/binaryop ' ] ;
    inc = [inc '-I../../../Source/transpose ' ] ;
    inc = [inc '-I../../../Source/helper ' ] ;

% FIXME: some of these are not needed:
%{
    inc = [inc '-I../../../Source/jitifyer ' ] ;
    inc = [inc '-I../../../Source/assign ' ] ;
    inc = [inc '-I../../../Source/cumsum ' ] ;
    inc = [inc '-I../../../Source/ok ' ] ;
    inc = [inc '-I../../../Source/omp ' ] ;
    inc = [inc '-I../../../Source/aliased ' ] ;
    inc = [inc '-I../../../Source/op ' ] ;
    inc = [inc '-I../../../Source/matrix ' ] ;
    inc = [inc '-I../../../Source/werk ' ] ;
    inc = [inc '-I../../../Source/transplant ' ] ;
    inc = [inc '-I../../../Source/callback ' ] ;
    inc = [inc '-I../../../Source/hyper ' ] ;
    inc = [inc '-I../../../Source/clear ' ] ;
    inc = [inc '-I../../../Source/type ' ] ;
    inc = [inc '-I../../../Source/slice ' ] ;
    inc = [inc '-I../../../Source/wait ' ] ;
    inc = [inc '-I../../../Source/print ' ] ;
    inc = [inc '-I../../../Source/compatible ' ] ;
    inc = [inc '-I../../../Source/nvals ' ] ;
    inc = [inc '-I../../../Source/cpu ' ] ;
    inc = [inc '-I../../../Source/dup ' ] ;
    inc = [inc '-I../../../Source/kronecker ' ] ;
    inc = [inc '-I../../../Source/positional ' ] ;
    inc = [inc '-I../../../Source/iso ' ] ;
    inc = [inc '-I../../../Source/unaryop ' ] ;
    inc = [inc '-I../../../Source/split ' ] ;
    inc = [inc '-I../../../Source/concat ' ] ;
    inc = [inc '-I../../../Source/global ' ] ;
    inc = [inc '-I../../../Source/lz4_wrapper ' ] ;
    inc = [inc '-I../../../Source/zstd_wrapper ' ] ;
    inc = [inc '-I../../../Source/builtin ' ] ;
    inc = [inc '-I../../../Source/extract ' ] ;
    inc = [inc '-I../../../Source/pending ' ] ;
    inc = [inc '-I../../../Source/indexunaryop ' ] ;
    inc = [inc '-I../../../Source/apply ' ] ;
    inc = [inc '-I../../../Source/context ' ] ;
    inc = [inc '-I../../../Source/builder ' ] ;
    inc = [inc '-I../../../Source/mask ' ] ;
    inc = [inc '-I../../../Source/element ' ] ;
    inc = [inc '-I../../../Source/memory ' ] ;
    inc = [inc '-I../../../Source/serialize ' ] ;
    inc = [inc '-I../../../Source/get_set ' ] ;
    inc = [inc '-I../../../Source/semiring ' ] ;
    inc = [inc '-I../../../Source/extractTuples ' ] ;
    inc = [inc '-I../../../Source/convert ' ] ;
    inc = [inc '-I../../../Source/ewise ' ] ;
    inc = [inc '-I../../../Source/reduce ' ] ;
    inc = [inc '-I../../../Source/import_export ' ] ;
    inc = [inc '-I../../../Source/select ' ] ;
    inc = [inc '-I../../../Source/mxm ' ] ;
    inc = [inc '-I../../../Source/resize ' ] ;
%}

if (need_rename)
    % use the renamed library for MATLAB
    flags = [flags ' -DGBMATLAB=1 ' ] ;
    inc = [inc ' -I../../rename ' ] ;
    libgraphblas = '-lgraphblas_matlab' ;
else
    % use the regular library for Octave
    libgraphblas = '-lgraphblas' ;
end

Lflags = sprintf ('-L''%s''', library) ;

fprintf ('compiler flags: %s\n', flags) ;
fprintf ('compiler incs:  %s\n', inc) ;
fprintf ('library:        %s\n', libgraphblas) ;

hfiles = [ dir('*.h') ; dir('util/*.h') ] ;

cfiles = dir ('util/*.c') ;

% Find the last modification time of any hfile.
% These are #include'd into source files.
htime = 0 ;
for k = 1:length (hfiles)
    t = datenum (hfiles (k).date) ;
    htime = max (htime, t) ;
end

% compile any source files that need compiling
any_c_compiled = false ;
objlist = '' ;
for k = 1:length (cfiles)

    % get the full cfile filename and modification time
    cfile = [(cfiles (k).folder) filesep (cfiles (k).name)] ;
    tc = datenum (cfiles(k).date) ;

    % get the object file name
    ofile = cfiles(k).name ;
    objfile = [ ofile(1:end-2) object_suffix ] ;

    % get the object file modification time
    objlist = [ objlist ' ' objfile ] ;     %#ok
    dobj = dir (objfile) ;
    if (isempty (dobj))
        % there is no object file; the cfile must be compiled
        tobj = 0 ;
    else
        tobj = datenum (dobj.date) ;
    end

    % compile the cfile if it is newer than its object file, or any hfile
    if (make_all || tc > tobj || htime > tobj)
        % compile the cfile
        % fprintf ('%s\n', cfile) ;
        fprintf ('.') ;
        mexcmd = sprintf ('mex -c %s -silent %s ''%s''', flags, inc, cfile) ;
        % fprintf ('%s\n', mexcmd) ;
        eval (mexcmd) ;
        any_c_compiled = true ;
    end
end

mexfunctions = dir ('mexfunctions/*.c') ;

% compile the mexFunctions
for k = 1:length (mexfunctions)

    % get the mexFunction filename and modification time
    mexfunc = mexfunctions (k).name ;
    mexfunction = [(mexfunctions (k).folder) filesep mexfunc] ;
    tc = datenum (mexfunctions(k).date) ;

    % get the compiled mexFunction modification time
    mexfunction_compiled = [ mexfunc(1:end-2) '.' mexext ] ;
    dobj = dir (mexfunction_compiled) ;
    if (isempty (dobj))
        % there is no compiled mexFunction; it must be compiled
        tobj = 0 ;
    else
        tobj = datenum (dobj.date) ;
    end

    % compile if it is newer than its object file, or if any cfile was compiled
    if (make_all || tc > tobj || any_c_compiled)
        % compile the mexFunction
        mexcmd = sprintf ('mex %s -silent %s %s ''%s'' %s %s', ...
            Lflags, flags, inc, mexfunction, objlist, libgraphblas) ;
        % fprintf ('%s\n', mexcmd) ;
        fprintf (':') ;
        eval (mexcmd) ;
    end
end

fprintf ('\n') ;

fprintf ('Compilation of the @GrB interface to GraphBLAS is complete.\n') ;
fprintf ('Add the following commands to your startup.m file:\n\n') ;
here1 = cd ('../..') ;
addpath (pwd) ;
fprintf ('  addpath (''%s'') ;\n', pwd) ;
cd ('..') ;
if (need_rename)
    cd ('GraphBLAS') ;
end
if ispc
    lib_path = sprintf ('%s/build/Release', pwd) ;
    fprintf ('  addpath (''%s'') ;\n', lib_path) ;
    addpath (lib_path) ;
end
cd (here1) ;

fprintf ('\nFor a quick demo of GraphBLAS, type the following commands:\n\n') ;
fprintf ('  cd ../../demo\n') ;
fprintf ('  gbdemo\n') ;

fprintf ('\nTo test GraphBLAS, type the following commands:\n\n') ;
fprintf ('  cd ../../test\n') ;
fprintf ('  gbtest\n') ;

