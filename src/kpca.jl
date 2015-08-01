#===================================================================================================
  Principal Components Analysis
===================================================================================================#

#==========================================================================
  PCA Objects
==========================================================================#

immutable PCA_Parameters{T<:FloatingPoint}
    n::Int64      # Observations
    p::Int64      # Dimensions
    α::T   # Regularization parameter for Σ ∝ MᵀM
    ϵ::T   # Perturbation parameter for Σ
    function PCA_Parameters(n::Int64, p::Int64, α::T, ϵ::T)
        n > 0 || throw(ArgumentError("n = $(n) must be positive a positive integer."))
        p > 0 || throw(ArgumentError("p = $(p) must be positive a positive integer."))
        ϵ >= 0 || throw(ArgumentError("ϵ = $(ϵ) must be a non-negative number."))
        0 <= α <= 1 || throw(ArgumentError("α = $(α) must be in the inverval [0,1]."))
        new(n, p, α, ϵ)
    end
end

function PCA_Parameters{T<:FloatingPoint}(n::Int64, p::Int64, α::T, ϵ::T)
    PCA_Parameters{T}(n, p, α, ϵ)
end

immutable PCA_Model{T<:FloatingPoint}
	Parameters::PCA_Parameters{T}
	Components::DataEigen{T}
end


#===================================================================================================
  Computational Routines
===================================================================================================#

function pca!{T<:FloatingPoint}(X::Matrix{T}, Model::PCA_Parameters{T}, override::Int64 = 0)
    μ = mean(X, 1)
    H = row_sub!(X, μ)
    d = convert(Float64, Model.n - 1)
    if (override == 0 && Model.α == Model.ϵ == 0) || override == 1
        BLAS.scal!(length(H), one(T) / sqrt(d), H, 1)  # Hₓ := Hₓ/√d
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
