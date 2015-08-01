#===================================================================================================
  Matrix Functions
===================================================================================================#

#==========================================================================
  Matrix Operations
==========================================================================#

# Diagonal-General Matrix Multiply: overwrites A with DA (scales A[i,:] by D[i,i])
function dgmm!{T<:FloatingPoint}(D::Array{T}, A::Matrix{T})
    n, p = size(A)
    if n != length(D)
        d = length(D)
        throw(ArgumentError(
            "The length(D) = $(d) of vector D representing a $(d)×$(d) diagonal matrix must " * (
            "equal the number of rows in Matrix A ∈ ℝ$(n)×$(p) to compute diagonal matrix " * (
            "product B = DA"))
        ))
    end
    @inbounds for j = 1:p
        for i = 1:n
            A[i, j] *= D[i]
        end
    end
    A
end
dgmm(D,A) = dgmm!(D, copy(A))

# General-Diagonal Matrix Multiply: overwrites A with AD
function gdmm!{T<:FloatingPoint}(A::Matrix{T}, D::Array{T})
    n, p = size(A)
    if p != length(D)
        d = length(D)
        throw(ArgumentError(
            "The length(D) = $(d) of vector D representing a $(d)×$(d) diagonal matrix must " * (
            "equal the number of columns in Matrix A ∈ ℝ$(n)×$(p) to compute diagonal matrix " * (
            "product B = AD"))
        ))
    end
    @inbounds for j = 1:p
        for i = 1:n
            A[i, j] *= D[j]
        end
    end
    A
end
gdmm(A,D) = gdmm!(copy(A), D)

# Symmetrize the lower off-diagonal of matrix S using the upper half of S
function syml!(S::Matrix)
    p = size(S, 1)
    p == size(S, 2) || throw(ArgumentError("S ∈ ℝ$(p)×$(size(S, 2)) should be square"))
    if p > 1 
        @inbounds for j = 1:(p - 1) 
            for i = (j + 1):p 
                S[i,j] = S[j,i]
            end
        end
    end
    return S
end
syml(S::Matrix) = syml!(copy(S))

# Symmetrize the upper off-diagonal of matrix S using the lower half of S
function symu!(S::Matrix)
	p = size(S,1)
	p == size(S,2) || throw(ArgumentError("S ∈ ℝ$(p)×$(size(S, 2)) must be square"))
	if n > 1 
		@inbounds for j = 2:p
			for i = 1:j-1
				S[i,j] = S[j,i]
			end 
		end
	end
	return S
end
symu(S::Matrix) = symu!(copy(S))

# Regularize a square matrix S, overwrites S with (1-α)*S + α*β*I
function regularize!{T<:FloatingPoint}(S::Matrix{T}, α::T, β::T=trace(S)/size(S,1))
	(p = size(S,1)) == size(S,2) || throw(ArgumentError("S ∈ ℝ$(p)×$(size(S, 2)) must be square"))
	0 <= α <= 1 || throw(ArgumentError("α=$(α) must be in the interval [0,1]"))
	@inbounds for j = 1:p
        for i = 1:p
            S[i,j] *= one(T) - α
        end
        S[j,j] += α*β
	end
	S
end
regularize{T<:FloatingPoint}(S::Matrix{T}, α::T) = regularize!(copy(S), α)
regularize{T<:FloatingPoint}(S::Matrix{T}, α::T, β::T) = regularize!(copy(S), α, β)

# Perturb a symmetric matrix S, overwrites S with S + ϵ*I
function perturb!{T<:FloatingPoint}(S::Matrix{T}, ϵ::T = 100*eps(T)*size(S,1))
	(p = size(S,1)) == size(S,2) || throw(ArgumentError("S ∈ ℝ$(p)×$(size(S, 2)) must be square"))
	0 <= ϵ || throw(ArgumentError("ϵ = $(ϵ) must be in [0,∞)"))
	@inbounds for i = 1:p
		S[i,i] += ϵ
	end
	S
end
perturb{T<:FloatingPoint}(S::Matrix{T}, ϵ::T) = perturb!(copy(S),ϵ)
perturb{T<:FloatingPoint}(S::Matrix{T}) = perturb!(copy(S))

# Add or substract a row vector from each row in a matrix
function row_add!{T<:FloatingPoint}(X::Matrix{T}, z::Array{T})
    n, p = size(X)
    p == length(z) || throw(ArgumentError("Dimensions do not conform"))
    @inbounds for j = 1:p
        for i = 1:n
            X[i,j] += z[j]
        end
    end
    X
end

function row_sub!{T<:FloatingPoint}(X::Matrix{T}, z::Array{T})
    n, p = size(X)
    p == length(z) || throw(ArgumentError("Dimensions do not conform"))
    @inbounds for j = 1:p
        for i = 1:n
            X[i,j] -= z[j]
        end
    end
    X
end


#==========================================================================
  Factorizations
==========================================================================#

# Returns count of non-zero eigenvalues
function count_nonzero{T<:FloatingPoint}(Λ::Array{T}, tol::T)
    d::Int64 = 0
    @inbounds for i = 1:length(Λ)
        if  Λ[i] >= tol
            d += 1
        end
    end
    d
end

function square_vec!{T<:FloatingPoint}(v::Array{T})
    @inbounds for i = 1:length(v)
        v[i] = v[i]^2
    end
    v
end

immutable DataEigen{T<:FloatingPoint}
    d::Int64
    Λ::Vector{T}
    V::Matrix{T}
    DataEigen(d, Σ, Vᵀ) = new(d, Σ, Vᵀ)
end

function data_eigfact!{T<:FloatingPoint}(X::Matrix{T}, tol::T = eps(T)*maximum(size(X))*maximum(X))
    # X = UΣVᵀ
	_U::Matrix{T}, Dₓ::Vector{T}, Vᵀ::Matrix{T} = LAPACK.gesdd!('S', X)
    Λ::Vector{T} = square_vec!(Dₓ)
    d::Int64 = count_nonzero(Λ, tol)
    DataEigen{T}(d, Λ[1:d], transpose(Vᵀ[1:d,:]))  # Trim zero-valued eigenvalues
end

function cov_eigfact!{T<:FloatingPoint}(Σₓ::Matrix{T}, 
                                        tol::T = eps(T)*maximum(size(Σₓ))*maximum(Σₓ))
    Λ::Vector{T}, V::Matrix{T} = LAPACK.syev!('V', 'U', Σₓ)  # V*diag(λ)*Vᵀ = Σₓ
    d::Int64 = count_nonzero(Λ, tol)
    DataEigen{T}(d, Λ[d:-1:1], V[:,d:-1:1])  # Trim zero-valued eigenvalues
end

# todo: Test sensitivity to k != 0 assumption
function data_eigfact!{T<:FloatingPoint}(M::Matrix{T}, X::Matrix{T}, 
                                         tol::T = eps(T)*maximum(size(X))*maximum(X))
    m, p = size(M)
    p == size(X, 2) || throw(ArgumentError("X and M must have the same number of columns."))
    n = size(X, 1)
    m <= n || throw(ArgumentError("M must have fewer rows than X."))
    # UᵀMQ = Σ₁[0 R], WᵀXQ = Σ₂[0 R]
    _U, _W, Q, D, Dₓ, k, l, R = LAPACK.ggsvd!('N', 'N', 'Q', M, X)
    if k != 0 
        error(
            "Infinite eigenvalues detected; ∃x: x ∈ range(M) and x ∈ null(X). The assumption " * (
            "range(M) ⊆ range(X) must hold for the solution to exist.")
        ) 
    end
    d = min(convert(Int64,l), m)  # Trim the trivial eigenvalues
    Λ::Vector{T} = square_vec!(D[1:d] ./ Dₓ[1:d])
    σ = sortperm(Λ, alg = QuickSort, rev = true)
    d = max(1, count_nonzero(Λ, tol))
    σ = σ[1:d]  # Choose indices with d greatest eigenvalues
    Λ = Λ[σ]    # Trim the zero eigenvalues
    r = size(R, 1)
    LAPACK.trtri!('U','N',R)  # Invert R
    V::Matrix{T} = BLAS.gemm('N', 'N', Q, r == p ? R[:,σ] : [zeros(T,p-r,d) ; R[:,σ]])
    gdmm!(V, 1 ./ Dₓ[σ])   # Normalize rows to ensure Σₓ orthogonality
    DataEigen{T}(d, Λ, V)  # Trim zero-valued eigenvalues?
end


function cov_eigfact!{T<:FloatingPoint}(Σ::Matrix{T}, Σₓ::Matrix{T},
                                        tol::T = eps(T)*maximum(size(Σₓ))*maximum(Σₓ))
    p = size(Σₓ,1)
    Λ::Vector{T}, V::Matrix{T}, _U::Matrix{T} = LAPACK.sygvd!(1, 'V', 'U', Σ, Σₓ)
    d::Int64 = count_nonzero(Λ, tol)
    DataEigen{T}(d, Λ[p:-1:(p-d+1)], V[:,p:-1:(p-d+1)])
end
