module DataDecomposition

# package code goes here

importall MLKernels

export lda_components!,
       lda_matrices!,
       lda!

include("common.jl")
include("klda.jl")

end # module
