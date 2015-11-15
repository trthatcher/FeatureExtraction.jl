n = 20  # Number of observations
p = 5   # Number of parameters

X = rand(n, p)
μ = mean(X,1)

H = (X .- μ)
Σ = H'H

info("Testing ", DataDecomposition.pca_components_eig!)
for T in FloatingPointTypes
    for α in (zero(T), convert(T,0.5), one(T)), ϵ in (zero(T), one(T))
        Σ_T = convert(Array{T}, Σ)
        V1, D1 = DataDecomposition.pca_components_eig!(convert(Array{T},H), α, ϵ)
        V2, D2 = DataDecomposition.components_eig!(((1-α)*Σ_T + (α/p)*trace(Σ_T)*I) + ϵ*I)
        @test_approx_eq V1 V2
        @test_approx_eq D1 D2
    end
end

info("Testing ", DataDecomposition.pca_components_svd!)
for T in FloatingPointTypes
    H_T = convert(Array{T},H)
    V1, D1 = DataDecomposition.pca_components_svd!(copy(H_T))
    V2, D2 = DataDecomposition.components_svd!(copy(H_T))

    @test_approx_eq V1 V2
    @test_approx_eq D1 D2
end

#κ = PolynomialKernel(one(T),zero(T),one(T))  # Equivalent to dot product
#K = centerkernelmatrix!(kernelmatrix(κ, X))




#V, D = pca!(copy(X))
#V, D = kpca!(copy(X),κ)

#@test_approx_eq diag(V'Xc'Xc*V)*(n-1) D
