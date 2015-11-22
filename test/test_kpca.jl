n = 20  # Number of observations
p = 5   # Number of parameters

info("Testing ", MOD.pca_components_eig!)
for T in FloatingPointTypes
    X = rand(T, n, p)
    μ = mean(X,1)
    H = (X .- μ)
    Σ = H'H
    for α in (zero(T), convert(T,0.5), one(T)), ϵ in (zero(T), one(T))
        V1, D1 = MOD.pca_components_eig!(copy(H), α, ϵ)
        V2, D2 = MOD.components_eig!(MOD.perturb!(MOD.regularize!(copy(Σ), α, trace(Σ)/p), ϵ))
        @test_approx_eq V1 V2
        @test_approx_eq D1 D2
    end
end

info("Testing ", MOD.pca_components_svd!)
for T in FloatingPointTypes
    X = rand(T, n, p)
    μ = mean(X,1)
    H = (X .- μ)
    V1, D1 = MOD.pca_components_svd!(copy(H))
    V2, D2 = MOD.components_svd!(copy(H))
    @test_approx_eq V1 V2
    @test_approx_eq D1 D2
end

info("Testing ", MOD.pca)
for T in FloatingPointTypes
    X = rand(T, n, p)
    μ = mean(X,1)
    H = (X .- μ)
    Σ = H'H/(n-1)
    for α in (zero(T), convert(T,0.5), one(T)), ϵ in (zero(T), one(T))
        V1, D1 = MOD.pca(copy(X), α, ϵ)
        D2, V2 = eig(MOD.perturb!(MOD.regularize!(copy(Σ), α, trace(Σ)/p), ϵ))
        @test_approx_eq abs(V1) abs(V2[:,end:-1:1])
        @test_approx_eq abs(D1) abs(D2[end:-1:1])
        @test_approx_eq abs(transform_pca(V1,X)) abs(X*(V2[:,end:-1:1]))
    end
end

info("Testing ", MOD.kpca)
for T in FloatingPointTypes
    X = rand(T, n, p)
    Xc = X .- mean(X,1)
    Σ = Xc'Xc/(n-1)
    κ = PolynomialKernel(one(T),zero(T),one(T))  # Equivalent to dot product
    for ϵ in (zero(T), one(T))
        V1, D1 = MOD.kpca(copy(X), κ, zero(T), ϵ)
        V2, D2 = MOD.pca(copy(X), zero(T), ϵ)
        @test_approx_eq (V1'Xc*Xc'V1/(n-1))[1:p,1:p] (V2'Σ*V2)
        @test_approx_eq abs(D1)[1:p] abs(D2)
    end
end
