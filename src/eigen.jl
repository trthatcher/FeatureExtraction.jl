
# Assumes X is one observation per row
# tol <= 0.0 trims nothing
function data_svd!{T<:FloatingPoint}(X::Matrix{T}, tol::T = eps(T)*maximum(size(X))*maximum(X))
    _U, D, Vᵀ = LAPACK.gesdd!('S', X)
    d = 0
    @inbounds @simd for i = 1:length(D)
        if D[i] >= tol
            d += 1
            D[i] = D[i]^2
        else
            break
        end
    end
    (transpose(Vᵀ[1:d,:]), D[1:d])
end

function cov_eig!{T<:FloatingPoint}(Σ::Matrix{T}, tol::T = eps(T)*maximum(size(Σ))*maximum(Σ))
    Λ, V = LAPACK.syev!('V', 'U', Σ)  # V*diag(λ)*Vᵀ = Σ
    n = length(Λ)
    d = 0
    @inbounds for i = n:-1:1
        if Λ[i] >= tol
            d += 1
        end
    end
    (V[:,n:-1:(n-d+1)], Λ[n:-1:(n-d+1)])
end


# Assumes X is one observation per row
# Assumes M is one mean per row
function data_gsvd!{T<:FloatingPoint}(M::Matrix{T}, X::Matrix{T}, tol::T = eps(T)*maximum(size(X))*maximum(X))
    (p = size(X,2)) == size(M, 2) || throw(ArgumentError("X and M must have the same number of columns."))
    n = size(X, 1)
    (m = size(M, 1)) <= n || throw(ArgumentError("M must have fewer rows than X."))
    _U, _W, Q, Dm, Dx, k, l, R = LAPACK.ggsvd!('N', 'N', 'Q', M, X)  # UᵀMQ = Σ₁[0 R], WᵀXQ = Σ₂[0 R]
    k == 0 || error("Generalised SVD failed because range(M) is not a subset of range(X)")  # M must have been computed incorrectly or FP precision...
    d = min(l, m)  # Trim the trivial eigenvalues
    Λ = Array(T, d)
    @inbounds @simd for i = 1:d
        Λ[i] = (Dm[i] / Dx[i])^2
    end
    σ = sortperm(Λ, alg = QuickSort, rev = true)
    Λ = Λ[σ]    # Trim the zero eigenvalues
    r = size(R, 1)
    LAPACK.trtri!('U', 'N', R)  # Invert R
    V = BLAS.gemm('N', 'N', Q, r == p ? R[:,σ] : [zeros(T,p-r,d) ; R[:,σ]])
    scale!(V, 1 ./ Dx[σ])   # Normalize rows to ensure Σx orthogonality
    (V, Λ)
end


function cov_geig!{T<:FloatingPoint}(S_m::Matrix{T}, S_x::Matrix{T}, tol::T = eps(T)*maximum(size(Σ))*maximum(Σ))
    (p = size(S_x, 1)) == size(S_x, 2) || throw(DimensionMismatch("Covariance matrix for X must be square."))
    size(S_m, 2) == size(S_m, 2) || throw(DimensionMismatch("Covariance matrix for M must be square."))
    p == size(S_m, 2) || throw(DimensionMismatch("Covariance matrices for X and M must be of the same order."))
    Λ, V, _U = LAPACK.sygvd!(1, 'V', 'U', S_m, S_x)
    d = 0
    @inbounds for i = n:-1:1
        if Λ[i] >= tol
            d += 1
        end
    end
    (V[:,n:-1:(n-d+1)], Λ[n:-1:(n-d+1)])
end
