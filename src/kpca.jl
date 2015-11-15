#==========================================================================
  PCA/KPCA Solvers
==========================================================================#


# PCA ###

function pca_components_eig!{T<:AbstractFloat}(H::Matrix{T}, α::T, ϵ::T)
    Σ = syml!(BLAS.syrk('U', 'T', one(T), H))  # Σ = H'H
    α == 0 || regularize!(Σ, α, trace(Σ)/size(Σ,1))
    ϵ == 0 || perturb!(Σ, ϵ)
    components_eig!(Σ)
end

pca_components_svd!{T<:AbstractFloat}(H::Matrix{T}) = components_svd!(H::Matrix{T})

function pca!{T<:AbstractFloat}(X::Matrix{T}, α::T = zero(T), ϵ::T = zero(T))
    n, p = size(X)
    μ = vec(mean(X,1))
    H = translate!(X, -μ)
    V, Λ = (α == 0 && ϵ == 0) ? pca_components_svd!(H) : pca_components_eig!(H, α, ϵ)
    (V, scale!(Λ, one(T)/(n-1)))  # Sampling correction of n-1
end

doc"""
`pca(X, α, ϵ)`
Computes the principal components of matrix `X`. `α` is a regularization that
shrinks the covariance matrix `Σ` towards the identity matrix scaled to the
average eigenvalue of `Σ`. `ϵ` perturbs `Σ` by `ϵI`.
"""
pca{T<:AbstractFloat}(X::Matrix{T}, α::T = zero(T), ϵ::T = zero(T)) = pca!(copy(X), α, ϵ)

doc"""
`transform_pca(W, Z)`
Applies the principal components `W` to Z.
"""
transform_pca{T<:AbstractFloat}(W::Matrix{T}, Z::Matrix{T}) = Z * W


# KPCA ###

doc"""
`kpca(X, κ, α, ϵ)`
Computes the kernel principal components of matrix `X` with respect to kernel 
κ.  `α` is a regularization that shrinks the kernel matrix `K` towards the 
identity matrix scaled to the average eigenvalue of `K`. `ϵ` perturbs `K` by 
`ϵI`.
"""
function kpca{T<:AbstractFloat}(X::Matrix{T}, κ::Kernel{T}, α::T = zero(T), ϵ::T = zero(T))
    n, p = size(X)
    K = kernelmatrix(κ, X)
    centerkernelmatrix!(K)
    α == 0 || regularize!(K, α, trace(K)/size(K,1))
    ϵ == 0 || perturb!(K, ϵ)
    V, Λ = components_eig!(K)
    (V, scale!(Λ, one(T)/(n-1)))
end

doc"""
`transform_kpca(X, κ, W, Z)`
Applies the kernel principal components `W` to `Z`. `X` and `κ` must be the 
original data matrix and kernel input into the solver.
"""
function transform_kpca{T<:AbstractFloat}(X::Matrix{T}, κ::Kernel{T}, W::Matrix{T}, Z::Matrix{T})
    K_zx = kernelmatrix(Z, X)
    μ_x = vec(mean(K_zx, 2))  # Need to center X in Hilbert space
    scale!(-μ, K_zx) * W
end
