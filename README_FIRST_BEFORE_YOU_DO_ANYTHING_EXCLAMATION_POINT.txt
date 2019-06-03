
This is a draft version of GraphBLAS V3.0.0, not yet suitable for public
release.  I've given access to a few people via the private
github.tamu.edu/GraphBLAS site, and via my GraphBLAS_drafts/ folder on Dropbox.

These drafts are fairly decent, from my level of quality control,
but sometimes I introduce bugs that can affect your results.
This code may not have passed all of my rigorous tests yet.

For example, from May 29 to June 3, there was a bug in the GxB_select
function that limited the # of threads it used to 1, regardless of the
omp_get_max_threads ( ) setting.  Oops.

This had a big effect on the performance of the Sparse DNN code in
LAGraph_dnn.c.

As a result of these kinds of bugs ...

    DO

        NOT

            BENCHMARK

                THIS

                    CODE

                        WITHOUT

                            MY

                                PERMISSION.

Got it?  Your license to use this early draft is contingent on this
constraint.

You might be benchmarking a bug ...  or you might be benchmarking a
code with debugging enabled (see "#define GB_DEBUG" in the Source/*
files).  Turning on debugging causes the code to become exceedingly slow,
since it rigorously checks all its matrices at every step.

Also, if you post performance results of this code, you cannot simply
say "Version 3.0.0" of GraphBLAS.  I'm not yet updating the version
number of GraphBLAS, so all of these codes are V3.0.0: both the ones
with the bug in GxB_select, and those without.

You must instead refer to the exact *date* of the release.
This is *NOT* always reflected in the DATE string in the
CMakeLists.txt file.  It is usually in the name of the tar.gz file
where you got this code.

If in any doubt ... ask me.

Tim Davis
davis@tamu.edu

