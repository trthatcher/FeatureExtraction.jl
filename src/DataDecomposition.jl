module DataDecomposition

# package code goes here

importall MLKernels

export lda_components!,
       lda_matrices!,
       lda!,
       transform_lda,
       klda!,
       transform_klda,
       pca,
       transform_pca,
       kpca,
       transform_kpca

include("common.jl")
include("kpca.jl")
include("klda.jl")

end # module
