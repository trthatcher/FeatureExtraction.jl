#==========================================================================
  KPCA Solvers
==========================================================================#

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
    (V, scale!(Λ, one(T)/(n-1)))  # Sampling correction
end

function kpca!{T<:AbstractFloat}(X::Matrix{T}, κ::Kernel{T}, α::T = zero(T), ϵ::T = zero(T))
    K = kernelmatrix(κ, X)
    centerkernelmatrix!(K)
    α == 0 || regularize!(K, α, trace(K)/size(K,1))
    ϵ == 0 || perturb!(K, ϵ)
    V, Λ = components_eig!(K)
    (V, scale!(Λ, one(T)/(n-1)))
end

# W is components returned
# Z must be the new matrix to be transformed
transform_pca{T<:AbstractFloat}(W::Matrix{T}, Z::Matrix{T}) = Z * W

# X must be original data matrix
# κ must be original kernel
# W is components returned
# Z must be the new matrix to be transformed
function transform_kpca{T<:AbstractFloat}(X::Matrix{T}, κ::Kernel{T}, W::Matrix{T}, Z::Matrix{T})
    K_zx = kernelmatrix(Z, X)
    μ_x = vec(mean(K_zx, 2))
    scale!(-μ, K_zx) * W
end
