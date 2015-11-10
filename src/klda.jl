#===================================================================================================
  Kernel Principal Components Analysis
===================================================================================================#

#==========================================================================
  LDA Objects
==========================================================================#

immutable LDA_Parameters{T<:AbstractFloat,U<:Integer}
    n::U  # Observations
    p::U  # Dimensions
    k::U  # Number of classes
    freq::Nullable{Vector{T}}  # Class frequencies
    alpha_mu  ::Nullable{T}  # Regularization parameter for Σ_μ ∝ MᵀM
    alpha_x   ::Nullable{T}  # Regularization parameter for Σ_x ∝ XᵀX
    epsilon_mu::Nullable{T}  # Perturbation parameter for Σ_μ
    epsilon_x ::Nullable{T}  # parameter for Σ_x
    function LDA_Parameters(n::U, 
                            p::U,
                            k::U, 
                            freq::Nullable{Vector{T}}, 
                            α_μ::Nullable{T}, 
                            α_x::Nullable{T}, 
                            ϵ_μ::Nullable{T}, 
                            ϵ_x::Nullable{T})
        get(n) > 0 || error("n = $(n) must be positive a positive integer.")
        p > 0 || error("p = $(p) must be positive a positive integer.")
        k > 0 || error("k = $(k) must be positive a positive integer.")
        #k == length(freq) || error("Class frequencies vector must have k = $(k) entries.")
        if !isnull(ϵ_μ)
            get(ϵ_μ) >= 0 || error("ϵ = $(get(ϵ_μ)) must be a non-negative number.")
        end
        ϵ_c >= 0 || error("ϵₓ = $(ϵ_x) must be a non-negative number.")
        0 <= α_μ <= 1 || error("α = $(α_μ) must be in the inverval [0,1].")
        0 <= α_x <= 1 || error("αₓ = $(α_x) must be in the inverval [0,1].")
        new(n, p, k, freq, α_μ, α_x, ϵ_μ, ϵ_x)
    end
end

#function LDA_Parameters{T<:AbstractFloat}(n::Int64, p::Int64, k::Int64, f::Vector{T}, α::T, αₓ::T,
#                                          ϵ::T, ϵₓ::T)
#    LDA_Parameters{T}(n, p, k, f, α, αₓ, ϵ, ϵₓ)
#end

immutable LDA_Model{T<:AbstractFloat}
	Parameters::LDA_Parameters{T}
	Components::DataEigen{T}
end


#==========================================================================
  Computational Routines
==========================================================================#

function lda!{T<:AbstractFloat}(y::Factor, X::Matrix{T}, Model::LDA_Parameters{T},
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

function lda{T<:AbstractFloat}(
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
function klda{T₁<:Integer,T₂<:AbstractFloat}(
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

function transform{T<:AbstractFloat}(Model::KLDA_Model{T},Z::Matrix{T})
	K = KERNEL.kernelmatrix(Z,Model.X,Model.κᵩ)
	return BLAS.gemm('N','N', K, Model.Components.W)
end
=#
