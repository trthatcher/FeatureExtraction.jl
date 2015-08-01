#===================================================================================================
  Kernel Principal Components Analysis
===================================================================================================#

#==========================================================================
  LDA Objects
==========================================================================#

immutable LDA_Parameters{T<:FloatingPoint}
    n::Int64      # Observations
    p::Int64      # Dimensions
    k::Int64      # Number of classes
    f::Vector{T}  # Class frequencies
    α::T   # Regularization parameter for Σ ∝ MᵀM
    αₓ::T  # Regularization parameter for Σₓ ∝ XᵀX
    ϵ::T   # Perturbation parameter for Σ
    ϵₓ::T  # parameter for Σₓ
    function LDA_Parameters(n::Int64, p::Int64, k::Int64, f::Vector{T}, α::T, αₓ::T, ϵ::T, ϵₓ::T)
        n > 0 || throw(ArgumentError("n = $(n) must be positive a positive integer."))
        p > 0 || throw(ArgumentError("p = $(p) must be positive a positive integer."))
        k > 0 || throw(ArgumentError("k = $(k) must be positive a positive integer."))
        k == length(f) || throw(ArgumentError("Class frequencies vector f ∈ ℝ$(length(f)) must " * (
                                              "have k = $(k) entries.")))
        ϵ >= 0 || throw(ArgumentError("ϵ = $(ϵ) must be a non-negative number."))
        ϵₓ >= 0 || throw(ArgumentError("ϵₓ = $(ϵₓ) must be a non-negative number."))
        0 <= α <= 1 || throw(ArgumentError("α = $(α) must be in the inverval [0,1]."))
        0 <= αₓ <= 1 || throw(ArgumentError("αₓ = $(αₓ) must be in the inverval [0,1]."))
        new(n, p, k, copy(f), α, αₓ, ϵ, ϵₓ)
    end
end

function LDA_Parameters{T<:FloatingPoint}(n::Int64, p::Int64, k::Int64, f::Vector{T}, α::T, αₓ::T,
                                          ϵ::T, ϵₓ::T)
    LDA_Parameters{T}(n, p, k, f, α, αₓ, ϵ, ϵₓ)
end

immutable LDA_Model{T<:FloatingPoint}
	Parameters::LDA_Parameters{T}
	Components::DataEigen{T}
end


#==========================================================================
  Computational Routines
==========================================================================#

function lda!{T<:FloatingPoint}(y::Factor, X::Matrix{T}, Model::LDA_Parameters{T},
                                override::Int64 = 0)
    M = class_means(y, X)  # M = [μ₁; μ₂; ...]
    Hₓ = center_rows!(y, X, M)
    μ = vec(BLAS.gemm('N', 'N', one(T), reshape(Model.f, 1, Model.k), M))  # μ = Σ fᵢ * μᵢ
    H = row_sub!(M, μ)  # Center M
    d = convert(T, Model.n - Model.k)  # d = (n - k) is the sampling correction for Σₓ
    dgmm!(sqrt(Model.f), H)  # H := diag(√(f*d))*H
    if override == 1
        BLAS.scal!(length(Hₓ), one(T) / sqrt(d), Hₓ, 1)  # Hₓ := Hₓ/√d
        tol = eps(T) * maximum(size(Hₓ)) * maximum(Hₓ)
        VΛ = data_eigfact!(H, Hₓ, tol)
    else
        Σ = syml!(BLAS.syrk('U', 'T', one(T), H))  # Σ = H'*diag(f)*H (weighted covariance)
        Model.α == 0 || regularize!(Σ, Model.α)
        Model.ϵ == 0 || perturb!(Σ, Model.ϵ)
        tol = eps(T) * maximum(size(Hₓ)) * maximum(Hₓ)
        Σₓ = syml!(BLAS.syrk('U', 'T', one(T) / d, Hₓ))  # Σₓ = Hₓ'Hₓ/d ∝ Hₓ'Hₓ
        Model.αₓ == 0 || regularize!(Σₓ, Model.αₓ)
        Model.ϵₓ == 0 || perturb!(Σₓ, Model.ϵₓ)
        VΛ = cov_eigfact!(Σ, Σₓ, tol)
    end
    VΛ
end


#===================================================================================================
  Interface
===================================================================================================#

function lda{T<:FloatingPoint}(
        y::Factor,
        X::Matrix{T};
        frequencies::Vector{T} = convert(Vector{T}, class_counts(y) ./ y.n),
        alpha::(T,T) = (zero(T), zero(T)),
        epsilon::(T,T) = (zero(T), zero(T)),
        override::Int64 = 0
    )
    n, p = size(X)
    n == y.n || throw(ArgumentError("X and y must have the same number of rows."))
    Parameters = LDA_Parameters(n, p, y.k, frequencies, alpha..., epsilon...)
    Components = lda!(y, copy(X), Parameters, override)
    LDA_Model(Parameters, Components)
end


#=
function klda{T₁<:Integer,T₂<:FloatingPoint}(
		y::Vector{T₁},
		X::Matrix{T₂};
		kernel::KERNEL.MercerKernel=KERNEL.LinearKernel(),
		frequencies::Vector{T₂} = convert(Array{T₂},class_counts(y)/length(y)),
		dimensions::Integer = maximum(y)-1,
		alpha::(Real,Real) = (0,0),
		epsilon::(Real,Real) = (0,0),
		override::Integer=0
	) 
	Parameters = LDA_Parameters(y,X,frequencies,alpha,epsilon)
	dimensions >= 0 || error("Dimensions = $dimensions must be non-negative.")
	Components = lda!(y,KERNEL.kernelmatrix(X,kernel),convert(Int64,dimensions),Parameters,override)
	return KLDA_Model(Parameters,Components,copy(X),kernel)
end

function transform{T<:FloatingPoint}(Model::KLDA_Model{T},Z::Matrix{T})
	K = KERNEL.kernelmatrix(Z,Model.X,Model.κᵩ)
	return BLAS.gemm('N','N', K, Model.Components.W)
end
=#
