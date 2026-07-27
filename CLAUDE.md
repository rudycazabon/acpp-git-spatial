# Background
The two directories in this project, sycl and slang, are prove out SYCL and SLANG kernels applied to the calculation of geospatial coordinate transformations in 32-bit floating point and 64-bit floating point.
One specific use case is that of applying the Helmert 7-parameter datum shift in both SLANG and SYCL where two 32-bit floats are used to mimick native 64-bit floating point and track the divergence in errors.

My initial queries began by using Google search to ask for example codes, but those were initially fauly, but after continuing queries and reporting of build and execution errors the the solutions in the SYCL folder stabilized yielding consistent results.

However, the SLANG implementation modeling the same transformations are still faulty.

# Purpose 
The purpose of this project is to analyze the project slang/ and help me solidify a solution that is numerically consistent with the sycl/ implmentation.

The file google-search.md contains the text of my queries and replies and suggested fixes that Google search has provided as context.

The file results.md contains the results of the from the executables in sycl/ and in slang/

The sycl/ solution is correct. The slang/ solution at this point compiles with no errors, but upon execution throws a segmentation fault.
