# GraphBLAS/GraphBLAS: MATLAB/Octave interface for SuiteSparse:GraphBLAS

SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
SPDX-License-Identifier: Apache-2.0

The @GrB class provides an easy-to-use interface to SuiteSparse:GraphBLAS.
This README.md file explains how to install it for use in MATLAB/Octave on
Linux, Mac, or Windows.

--------------------------------------------------------------------------------
# For Linux/Mac
--------------------------------------------------------------------------------

    To install GraphBLAS for use in MATLAB/Octave, do the following inside the
    MATLAB/Octave Command Window:

        cd /home/me/GraphBLAS/GraphBLAS
        graphblas_install
        addpath (pwd)
        cd test
        gbtest

    That should be enough.  However, the above script may fail to compile the
    libgraphblas_matlab.so (for MATLAB) or libgraphblas.so (for Octave).  If so,
    then you will need to first compile the GraphBLAS library outside of
    MATLAB/Octave.

    Suppose your copy of GraphBLAS is in /home/me/GraphBLAS.  For MATLAB on
    Linux/Mac, compile libgraphblas_matlab.so (.dylib on the Mac) with:

        cd /home/me/GraphBLAS/GraphBLAS
        make

    For Octave on Linux/Mac, compile libgraphblas.so (.dylib on the Mac) with:

        cd /home/me/GraphBLAS
        make

    If the 'make' command above fails, do this instead (assuming you are in the
    /home/me/GraphBLAS/GraphBLAS folder for MATLAB, or /home/me/GraphBLAS for
    Octave), outside of MATLAB/Octave:

        cd build
        cmake  ..
        cmake --build . --config Release -j40

    Then inside MATLAB/Octave, do this:

        cd /home/me/GraphBLAS/GraphBLAS/@GrB/private
        gbmake

--------------------------------------------------------------------------------
# For Windows
--------------------------------------------------------------------------------

    On Windows 10, on the Search bar type env and hit enter; (or you can
    right-click My Computer or This PC and select Properties, and then select
    Advanced System Settings).  Select "Edit the system environment variables",
    then "Environment Variables".  Under "System Variables" select "Path" and
    click "Edit".  These "New" to add a path and then "Browse".  Browse to the
    folder (for example: C:/Users/me/Documents/GraphBLAS/build/Release) and add
    it to your path.  For MATLAB, you must use the libgraphblas_matlab.dll, in:
    /User/me/SuiteSparse/GraphBLAS/GraphBLAS/build/Release instead.  Then close
    the editor, sign out of Windows and sign back in again.

    Then do this inside of MATLAB/Octave:

        cd /home/me/GraphBLAS/GraphBLAS/@GrB/private
        gbmake

--------------------------------------------------------------------------------
# After installation on Linux/Mac/Windows
--------------------------------------------------------------------------------

    Add this command to your startup.m file:

        % add the MATLAB/Octave interface to the MATLAB/Octave path
        addpath ('/home/me/GraphBLAS/GraphBLAS') :

    where the path /home/me/GraphBLAS/GraphBLAS is the full path to this
    folder.

    The name "GraphBLAS/GraphBLAS" is used for this folder so that this can be
    done in MATLAB/Octave:

        help GraphBLAS

    To get additional help, type:

        methods GrB
        help GrB

    To run the demos, go to the GraphBLAS/GraphBLAS/demo folder and type:

        gbdemo
        gbdemo2

    To test your installation, go to GraphBLAS/GraphBLAS/test and type:

        gbtest

    If everything is successful, it should report 'gbtest: all tests passed'.
    Note that gbtest tests all features of the MATLAB/Octave interface to
    SuiteSparse/GraphBLAS, including error handling, so you can expect to see
    error messages during the test.  This is expected.

--------------------------------------------------------------------------------
# MATLAB vs Octave
--------------------------------------------------------------------------------

    You cannot use a single copy of the GraphBLAS source distribution to use in
    both MATLAB and Octave on the same system at the same time.  The .o files
    in GraphBLAS/GraphBLAS/@GrB/private compiled by the graphblas_install.m
    will conflict with each other.  To switch between MATLAB and Octave, use a
    second copy of the GraphBLAS source distribution, or do a clean
    installation (via "make purge" in the GraphBLAS/GraphBLAS/@GrB/private
    folder, outside of MATLAB/Octave) and redo the above instructions.  There
    is no need to recompile the libgraphblas.so (or dylib on the Mac) since
    Octave uses GraphBLAS/build/libgraphblas.so while MATLAB uses
    GraphBLAS/GraphBLAS/build/libgraphblas_matlab.so.  Both MATLAB and Octave
    can share the same compiled JIT kernels.

--------------------------------------------------------------------------------
# FUTURE: Not yet supported for GrB matrices in MATLAB/Octave:
--------------------------------------------------------------------------------

    linear indexing, except for C=A(:) to index the whole matrix A
        or C(:)=A to index the whole matrix C.
    2nd output for [x,i] = max (...) and [x,i] = min (...):
        use GrB.argmin and GrB.argmax instead.
    'includenan' for min and max
    min and max for complex matrices
    singleton expansion
    saturating element-wise binary and unary operators for integers.
        See also the discussion in the User Guide.

These functions are supported, but are not yet as fast as they could be:

    eps, ishermitian, issymmetric, spfun.

