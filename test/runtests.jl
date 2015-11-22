using FeatureExtraction
using MLKernels
using Base.Test

MOD = FeatureExtraction

FloatingPointTypes = (Float32, Float64)

# Unit Tests
include("test_common.jl")
include("test_kpca.jl")
