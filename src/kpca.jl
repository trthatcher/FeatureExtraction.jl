#===================================================================================================
  Principal Components Analysis
===================================================================================================#

#==========================================================================
  PCA Objects
==========================================================================#

immutable ParametersPCA{T<:FloatingPoint}
    alg::Symbol
    n::Int64  # Observation count
    p::Int64  # Original number of parameters
    alpha::T  # Regularization parameter for covariance matrix
    epsilon::T   # Perturbation parameter for covariance matrix
    mu::Vector{T}
    tolerance::T
    max_components::Int64
    function ParametersPCA(alg::Symbol, n::Int64, p::Int64, α::T, ϵ::T, μ::Vector{T}, tol::T, max_comp::Int64)
        n > 0 || throw(ArgumentError("n = $(n) must be positive a positive integer."))
        p > 0 || throw(ArgumentError("p = $(p) must be positive a positive integer."))
        ϵ >= 0 || throw(ArgumentError("ϵ = $(ϵ) must be a non-negative number."))
        0 <= α <= 1 || throw(ArgumentError("α = $(α) must be in the inverval [0,1]."))
        tol >= 0 || throw(ArgumentError("tol = $(tol) must be a non-negative number."))
        max_comp >= 1 || throw(ArgumentError("Must select at least one component."))
        new(alg, n, p, α, ϵ, μ, tol, max_components)
    end
end
ParametersPCA{T<:FloatingPoint}(n::Int64, p::Int64, α::T, ϵ::T, tol::T, max_comp::Int64) = ParametersPCA{T}(n, p, α, ϵ, tol, max_comp)

immutable ComponentsPCA{T<:FloatingPoint}
    V::Matrix{T}
    d::Int64
end
ComponentsPCA{T<:FloatingPoint}(μ::Vector{T}, V::Matrix{T}, d::Int64) = ComponentsPCA{T}(μ, V, d)

immutable ModelPCA{T<:FloatingPoint}
	parameters::ParametersPCA{T}
	components::ComponentsPCA{T}
end


#===================================================================================================
  Computational Routines
===================================================================================================#

function center_data!(X::Matrix{T})
    μ = mean(X, 1)
    H = X .- μ
    (μ, H)
end



function apply_transform!{T<:FloatingPoint}(Σ::Matrix{T}, param::ParametersPCA{T})
    if param.α != 0
        regularize!(Σ, param.α)
    end
    if param.ϵ != 0
        perturb!(Σ, param.ϵ)
    end
    Σ
end

function pca_svd!(H::Matrix{T}, tol::T)
    data_svd!(H, tol)
end

pca_svd!(H::Matrix{T})
    


function pca_data!{T<:FloatingPoint}(X::Matrix{T}, param::ParametersPCA{T}, μ::Vector{T} = vec(mean(X,1)))
    H = length(μ) == 0 ? X : X .- μ'
    if param.alg == :svd

    H = X .- μ


# IS the matrix a covariance matrix?
# What is the extraction method?
# Components or tolerance

function pca_cov!{T<:FloatingPoint}(Σ::Matrix{T}, param::ParametersPCA{T})
    if method

function pca!{T<:FloatingPoint}(X::Matrix{T}, param::ParametersPCA{T}, algorithm::Symbol, is_cov::Bool)
    if algorithm == :svd
        μ, H = center_data!(X)
        if 

    elseif algorithm == :eig

    else 
        error("")
    end

    μ = mean(X, 1)
    H = X .- μ
    if (override == 0 && param.α == param.ϵ == 0)
        scale!(H, one(T)/sqrt(param.n - one(T)))
        tol = eps(T) * maximum(size(H)) * maximum(H)
        VD = data_eigfact!(H, tol)
    else
        tol = eps(T) * maximum(size(H)) * maximum(H)
		Σ = BLAS.syrk('U', 'T', one(T) / d, H)  # Σ = H'H/d
		Model.α == 0 || regularize!(Σ, Model.α)
		Model.ϵ == 0 || perturb!(Σ, Model.ϵ)
		VD = cov_eigfact!(Σ, tol)
    end
    VD
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
