
# Assumes X is one observation per row
function data_svd!{T<:FloatingPoint}(X::Matrix{T}, tol::T = eps(T)*maximum(size(X))*maximum(X))
    _U::Matrix{T}, D::Vector{T}, Vᵀ::Matrix{T} = LAPACK.gesdd!('S', X)
    @inbounds @simd for i = 1:length(D)
        D[i] = D[i]^2
    end
    (transpose(Vᵀ), D)
end

# Assumes X is one observation per row
# Assumes M is one mean per row
function data_svd!{T<:FloatingPoint}(M::Matrix{T}, X::Matrix{T}, tol::T = eps(T)*maximum(size(X))*maximum(X))
    (p = size(X,2)) == size(M, 2) || throw(ArgumentError("X and M must have the same number of columns."))
    n = size(X, 1)
    (m = size(M, 1)) <= n || throw(ArgumentError("M must have fewer rows than X."))
    _U, _W, Q, Dm, Dx, k, l, R = LAPACK.ggsvd!('N', 'N', 'Q', M, X)  # UᵀMQ = Σ₁[0 R], WᵀXQ = Σ₂[0 R]
    k != 0 || error("Generalised SVD failed because range(M) is not a subset of range(X)")  # M must have been computed incorrectly or FP precision...
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
    (V, Λ)  # Trim zero-valued eigenvalues?
end
