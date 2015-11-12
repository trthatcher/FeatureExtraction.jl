#==========================================================================
  LDA Solvers
==========================================================================#

function class_counts{T<:Integer}(y::Vector{T}, k::T)
    counts = zeros(Int64, k)
    for i = 1:length(y)
        y[i] <= k || error("Index $i out of range.")
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

function center_rows!{T<:AbstractFloat,U<:Integer}(X::Matrix{T}, M::Matrix{T}, y::Vector{U})
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
    components_geig!(Σ_b, Σ_b)
end

function lda_matrices!{T<:AbstractFloat,U<:Integer}(X::Matrix{T}, y::Vector{U}, k::U, frequency::Vector{T})
    n, p = size(X)
    M = class_totals(X, y, k)
    scale!(one(T) ./ class_counts(y, k), M)
    H_w = center_rows!(X, M, y)  # M = [μ1; μ2; ...]
    scale!(H_w, one(T)/(n-k))  # sampling correction factor for Σ_w
    μ = vec(frequency'M)
    translate!(M,-μ)  # M:= M .- μ'
    H_b = scale!(sqrt(frequency), M)  # M := freq .* M (applies weights to class mean)
    (H_b, H_w, μ)
end

function lda!{T<:AbstractFloat,U<:Integer}(X::Matrix{T}, y::Vector{U}, k::U, frequency::Vector{T},
                                           α_b::T = zero(T), ϵ_b::T = zero(T), α_w::T = zero(T), ϵ_w::T = zero(T))
    H_b, H_w = lda_matrices!(X, y, k, frequency)
    lda_components!(H_b, α_b, ϵ_b, H_w, α_w, ϵ_w)
end

# W is components returned
# Z must be the new matrix to be transformed
transform{T<:AbstractFloat}(W::Matrix{T}, Z::Matrix{T}) = Z * W

#=
function klda!{T<:AbstractFloat,U<:Integer}(X::Matrix{T}, κ::Kernel{T}, y::Vector{U}, k::U, frequency::Vector{T},
                                           α_b::T = zero(T), ϵ_b::T = zero(T), α_w::T = zero(T), ϵ_w::T = zero(T))
    H_b, H_w = lda_matrices!(kernelmatrix(κ, X), y, k, frequency)
    lda_components!(H_b, α_b, ϵ_b, H_w, α_w, ϵ_w)
end
=#

# X must be original data matrix
# κ must be original kernel
# W is components returned
# Z must be the new matrix to be transformed
#=
function transform{T<:AbstractFloat}(X::Matrix{T}, κ::Kernel{T}, W::Matrix{T}, Z::Matrix{T})
	kernelmatrix(Z, X) * W
end
=#
