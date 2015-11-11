#==========================================================================
  LDA Solvers
==========================================================================#

function class_counts{T<:Integer}(y::Vector{T}, k::T)
    counts = zeros(Int64, k)
    for i = 1:length(y)
        i < k || error("Index out of range.")
        counts[y[i]] += 1
    end
    counts
end

function class_means{T<:AbstractFloat,U<:Integer}(X::Matrix{T}, y::Vector{U}, k::U = maximum(y))
    n, p = size(X)
    length(y) == n || throw(DimensionMismatch("X and y must have the same number of rows."))
    scale_factor = one(T) ./ class_counts(y, k)
    M = Array(T, k, p)
    for j = 1:p, i = 1:n
        M[y[i],j] += X[i,j]
    end
    scale!(M, scale_factor)
end

#function center_rows

# Takes intermediate matrices such that H_b'H_b = Σ_b, H_w'H_w = Σ_w
# and returns (lda_components, eigenvalues)
function lda_components!{T<:AbstractFloat}(H_b::Matrix{T}, α_b::T, ϵ_b::T, H_w::Matrix{T}, α_w::T, ϵ_w::T)
    Σ_b = syml!(BLAS.syrk('U', 'T', one(T), H_b))  # Σ_b = H_b'*diag(freq)*H_b
    α_b == 0 || regularize!(Σ_b, α_b)
    ϵ_b == 0 || perturb!(Σ_b, ϵ_b)
    #tol = eps(T) * maximum(size(H_w)) * maximum(H_w)
    Σ_w = syml!(BLAS.syrk('U', 'T', one(T), H_w))  # Σₓ = Hₓ'Hₓ/d ∝ Hₓ'Hₓ
    α_w == 0 || regularize!(Σ_w, α_w)
    ϵ_w == 0 || perturb!(Σ_w, ϵ_w)
    components_eig(Σ_b, Σ_b)
end

#=
function lda!{T<:AbstractFloat,U<:Integer}(X::Matrix{T}, y::Vector{U})
    M = class_means(X, y)  # M = [μ1; μ2; ...]
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
=#


#===================================================================================================
  Interface
===================================================================================================#

#=
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

=#

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
