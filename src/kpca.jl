#==========================================================================
  KPCA Solvers
==========================================================================#

function pca_components_eig!{T<:AbstractFloat}(H::Matrix{T}, α::T, ϵ::T)
    Σ = syml!(BLAS.syrk('U', 'T', one(T), H_b))  # Σ = H'H
    α == 0 || regularize!(Σ, α, trace(Σ)/size(Σ,1))
    ϵ == 0 || perturb!(Σ, ϵ)
    components_eig!(Σ)
end

pca_components_svd!(H::Matrix{T}) = components_svd!(H::Matrix{T})

function pca!{T<:AbstractFloat}(X::Matrix{T}, α::T = zero(T), ϵ:: = zero(T))
    n, p = size(H)
    μ = mean(X,1)
    H = translate!(X, μ)
    scale!(H, one(T)/sqrt(n-1))  # Scaling constant
    if α == 0 && ϵ == 0
        return pca_components_svd!(H)
    else
        return pca_components_eig!(H, α, ϵ)
    end
end

#= 

function pca!{T<:AbstractFloat}(
        X::Matrix{T},
        algorithm::Symbol = :auto,
        α::T = zero(T),
        ϵ::T = zero(T),
        μ::Vector{T} = vec(mean(X,1))
        tolerance::Union(T,Symbol) = :auto,
        max_dimension::Union(Int64,Symbol) = :auto
    )
    n = size(X,1)
    p = size(X,2)
    length(μ) == p || length(μ) == 0 || throw(DimensionMismatch("Mean vector must be of dimension zero or p."))
    H = length(μ) == 0 ? X : translate!(X, -μ)
    if algorithm == :auto
        algorithm = ϵ == 0 && α == 0 ? :svd : :eig
    end
    tol = isa(tolerance, T) ? tolerance : eps(T) * maximum(size(H)) * maximum(H)
    max_dim = isa(max_dimension, Int64) ? max_dimension : p
    β = 1/(n - one(T))  # Scaling constant for Σ = H'H/√(n-1)
    params = ParametersPCA(algorithm, α, ϵ, μ, tol, max_dim)
    if algorithm == :svd
        scale!(H, sqrt(β))  
        V, D = pca_svd!(H, params)
        return ModelPCA(params, ComponentsPCA(V, D))
    elseif algorithm == :eig
        Σ = BLAS.syrk('U', 'T', β, H)
        V, D = pca_eig!(Σ, params)
        return ModelPCA(params, ComponentsPCA(V, D))
    else 
        error("Unrecognized algorithm.")
    end
end

function kpca!{T<:AbstractFloat}(
        K::Matrix{T},
        algorithm::Symbol = :auto,
        α::T = zero(T),
        ϵ::T = zero(T),
        tolerance::Union(T,Symbol) = :auto,
        max_dimension::Union(Int64,Symbol) = :auto
    )
    (n = size(K,1)) == size(K,2) || throw(DimensionMismatch("Kernel matrix must be square."))
    μ = vec(mean(K,1))
    translate!(K, -μ)
    tol = isa(tolerance, T) ? tolerance : eps(T) * maximum(size(H)) * maximum(H)
    max_dim = isa(max_dimension, Int64) ? max_dimension : n
    β = 1/(n - one(T))  # Scaling constant for Σ = H'H/√(n-1)
    params = ParametersPCA(algorithm, α, ϵ, μ, tol, max_dim)
    if algorithm == :svd
        scale!(H, sqrt(β))  
        V, D = pca_svd!(H, params)
        return ModelPCA(params, ComponentsPCA(V, D))
    elseif algorithm == :eig
        Σ = BLAS.syrk('U', 'T', β, H)
        V, D = pca_eig!(Σ, params)
        return ModelPCA(params, ComponentsPCA(V, D))
    else 
        error("Unrecognized algorithm.")
    end
end

=#


#===================================================================================================
  Interface
===================================================================================================#

#=

function pca{T<:AbstractFloat}(
        X::Matrix{T};
        alpha::T = zero(T),
        epsilon::T = zero(T),
        override::Int64 = 0
    )
    n, p = size(X)
    Parameters = PCA_Parameters(n, p, alpha, epsilon)
    Components = pca!(copy(X), Parameters, override)
    PCA_Model(Parameters, Components)
end
=#

#=
function kpca{T<:AbstractFloat}(
		X::Matrix{T}; 
		kernel::KERNEL.MercerKernel=KERNEL.LinearKernel(),
		dimensions::Integer=0, 
		alpha::Real=0, 
		epsilon::Real=0,
	) 
	n,p = size(X)
	Parameters = PCA_Parameters(n,p,alpha,epsilon)
	K = KERNEL.kernelmatrix(X,kernel)
	Components = kpca!(K,dimensions,Parameters)
	return KPCA_Model(Parameters,Components,copy(X),kernel)
end

function transform(Model::KPCA_Model,Z::Array{Float64})
	size(Z,2) == size(Model.X,2) || error("Dimension mismatch")
	K = KERNEL.kernelmatrix(Z,Model.X,Model.κᵩ)
	μₖ = mean(K,2)
	MATRIX.col_add!(K,-μₖ)  # Center kernel matrix in X
	return BLAS.gemm('N','N', K, Model.Components.W)
end
=#
