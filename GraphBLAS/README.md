# GraphBLAS/GraphBLAS: MATLAB interface for SuiteSparse:GraphBLAS

SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

The gb class provides an easy-to-use MATLAB interface to SuiteSparse:GraphBLAS.

To install it for use in MATLAB, first compile the GraphBLAS library,
-lgraphblas.  See the instructions in the top-level GraphBLAS folder for
details.  Be sure to use OpenMP for best performance.

Next, start MATLAB and go to this GraphBLAS/GraphBLAS folder.  Type

    addpath (pwd)

to add the GraphBLAS interface to your path.  Then do

    savepath

Or, if that function is not allowed because of file permissions, add a command
to your startup.m file:

    addpath /whereever/GraphBLAS/GraphBLAS

where the path /whereever/GraphBLAS/GraphBLAS is the full path to this folder.
The name "GraphBLAS/GraphBLAS" is used so that this can be done in MATLAB:

    help GraphBLAS

Next, go to the GraphBLAS/GraphBLAS/private folder and compile the MATLAB
mexFunctions:

    cd private
    gbmake

To run the demos, go to the GraphBLAS/GraphBLAS/demo folder and type:

    gbdemo
    gbdemo2

The output of these demos on a Dell XPS 13 laptop and an NVIDIA DGX Station can
also be found in GraphBLAS/GraphBLAS/demo/html, in both PDF and HTML formats.

To test your installation, go to GraphBLAS/GraphBLAS/test and type:

    gbtestall

If everything is successful, it should report 'gbtestall: all tests passed'.

