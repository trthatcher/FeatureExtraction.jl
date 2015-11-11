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

function class_totals{T<:AbstractFloat,U<:Integer}(X::Matrix{T}, y::Vector{U}, k::U)
    n, p = size(X)
    length(y) == n || throw(DimensionMismatch("X and y must have the same number of rows."))
    M = zeros(T, k, p)
    for j = 1:p, i = 1:n
        M[y[i],j] += X[i,j]
    end
    M
end

function center_rows!{T<:AbstractFloat,U<:Integer}(X::Matrix{T}, M::Matrix{T}, y::Vector{u})
    n, p = size(X)
    for j = 1:p, i = 1:n
        X[i,j] -= M[y[i],j]
    end
    X
end

# Takes intermediate matrices such that H_b'H_b = Σ_b, H_w'H_w = Σ_w
# and returns (lda_components, eigenvalues)
function lda_components!{T<:AbstractFloat}(H_b::Matrix{T}, α_b::T, ϵ_b::T, H_w::Matrix{T}, α_w::T, ϵ_w::T)
    Σ_b = syml!(BLAS.syrk('U', 'T', one(T), H_b))  # Σ_b = H_b'*diag(freq)*H_b
    α_b == 0 || regularize!(Σ_b, α_b)
    ϵ_b == 0 || perturb!(Σ_b, ϵ_b)
    #tol = eps(T) * maximum(size(H_w)) * maximum(H_w)
    Σ_w = syml!(BLAS.syrk('U', 'T', one(T), H_w))  # Σ_w = H_w'H_w
    α_w == 0 || regularize!(Σ_w, α_w)
    ϵ_w == 0 || perturb!(Σ_w, ϵ_w)
    components_eig(Σ_b, Σ_b)
end

function lda!{T<:AbstractFloat,U<:Integer}(X::Matrix{T}, y::Vector{U}, k::U, frequency::Vector{T})
    n, p = size(X)
    M = class_totals(X, y, k)
    scale!(one(T) ./ class_counts(y), M)
    H_w = center_rows!(X, M, y)  # M = [μ1; μ2; ...]
    scale!(H_w, one(T)/(n-k))  # sampling correction factor for Σ_w
    μ = vec(frequency'M)
    translate!(μ, M)  # M:= M .- μ'
    H_b = scale!(sqrt(frequency), M)  # M := freq .* M (applies weights to class mean)
    (H_b, H_m)
end

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
