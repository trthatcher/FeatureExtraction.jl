#===================================================================================================
  Principal Components Analysis
===================================================================================================#

using MLKernels

#==========================================================================
  PCA Objects
==========================================================================#

immutable ParametersPCA{T<:FloatingPoint}
    algorithm::Symbol
    alpha::T  # Regularization parameter for covariance matrix
    epsilon::T   # Perturbation parameter for covariance matrix
    mu::Vector{T}
    tolerance::T
    max_dimension::Int64
    function ParametersPCA(algorithm::Symbol, α::T, ϵ::T, μ::Vector{T}, tolerance::T, max_dimension::Int64)
        ϵ >= 0 || throw(ArgumentError("ϵ = $(ϵ) must be a non-negative number."))
        0 <= α <= 1 || throw(ArgumentError("α = $(α) must be in the inverval [0,1]."))
        length(μ) == p || length(μ) == 0 || throw(ArgumentError("Mean vector must be length p = $(p) or p = 0."))
        tolerance >= 0 || throw(ArgumentError("tol = $(tol) must be a non-negative number."))
        max_dimension >= 1 || throw(ArgumentError("Must select at least one component."))
        max_dimension <= p || throw(ArgumentError("Max dimension must not exceed p = $(p)."))
        new(algorithm, n, p, α, ϵ, μ, tolerance, max_dimension)
    end
end
ParametersPCA{T<:FloatingPoint}(n::Int64, p::Int64, α::T, ϵ::T, tol::T, max_comp::Int64) = ParametersPCA{T}(n, p, α, ϵ, tol, max_comp)

immutable ComponentsPCA{T<:FloatingPoint}
    V::Matrix{T}
    D::Vector{T}
end
ComponentsPCA{T<:FloatingPoint}(μ::Vector{T}, V::Matrix{T}, d::Int64) = ComponentsPCA{T}(μ, V, d)

immutable ModelPCA{T<:FloatingPoint}
	parameters::ParametersPCA{T}
	components::ComponentsPCA{T}
end

immutable ModelKPCA{T<:FloatingPoint}
    kernel::Kernel{T}
	parameters::ParametersPCA{T}
	components::ComponentsPCA{T}
end



#===================================================================================================
  Computational Routines
===================================================================================================#


function pca_eig!{T<:FloatingPoint}(Σ::Matrix{T}, parameters::ParametersPCA{T})
    parameters.α == 0 || regularize!(Σ, parameters.α)
    parameters.ϵ == 0 || perturb!(Σ, parameters.ϵ)
    components_eig!(Σ, parameters.tolerance, parameters.max_dimension)
end

pca_svd!(H::Matrix{T}, param::ParametersPCA{T} = components_svd!(H, parameters.tolerance, parameters.max_dimension)

# IS the matrix a covariance matrix?
# What is the extraction method?
# Components or tolerance

function pca!{T<:FloatingPoint}(
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

function kpca!{T<:FloatingPoint}(
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



#===================================================================================================
  Interface
===================================================================================================#

function pca{T<:FloatingPoint}(
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


#=
function kpca{T<:FloatingPoint}(
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
