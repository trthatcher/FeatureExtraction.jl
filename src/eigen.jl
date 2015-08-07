# Commonly used 


const liblapack = Base.liblapack_name

import Base.blasfunc

import Base.LinAlg: BlasFloat, BlasChar, BlasInt, blas_int, LAPACKException,
    DimensionMismatch, SingularException, PosDefException, chkstride1, chksquare

#Generic LAPACK error handlers
macro assertargsok() #Handle only negative info codes - use only if positive info code is useful!
    :(info[1]<0 && throw(ArgumentError("invalid argument #$(-info[1]) to LAPACK call")))
end
macro lapackerror() #Handle all nonzero info codes
    :(info[1]>0 ? throw(LAPACKException(info[1])) : @assertargsok )
end

macro assertnonsingular()
    :(info[1]>0 && throw(SingularException(info[1])))
end
macro assertposdef()
    :(info[1]>0 && throw(PosDefException(info[1])))
end

#Check that upper/lower (for special matrices) is correctly specified
macro chkuplo()
    :((uplo=='U' || uplo=='L') || throw(ArgumentError("""invalid uplo = $uplo
Valid choices are 'U' (upper) or 'L' (lower).""")))
end

for (syevd, elty) in
    ((:dsyevd_, :Float64),
     (:ssyevd_, :Float32))
    @eval begin
        #       SUBROUTINE dsyevd( JOBZ, UPLO, N, A, LDA, W, WORK, LWORK, IWORK, LIWORK, INFO )
        # *     .. Scalar Arguments ..
        #       CHARACTER          JOBZ, UPLO
        #       INTEGER            INFO, LDA, LIWORK, LWORK, N
        # *     .. Array Arguments ..
        #       INTEGER            IWORK( * )
        #       DOUBLE PRECISION   A( LDA, * ), W( * ), WORK( * )
        function syevd!(jobz::BlasChar, uplo::BlasChar, A::StridedMatrix{$elty})
            chkstride1(A)
            n = chksquare(A)
            W     = similar(A, $elty, n)
            work  = Array($elty, 1)
            lwork = blas_int(-1)
            iwork  = Array(BlasInt, 1)
            liwork = blas_int(-1)
            info  = Array(BlasInt, 1)
            for i in 1:2
                ccall(($(blasfunc(syevd)), liblapack), Void,
                     (Ptr{BlasChar}, Ptr{BlasChar}, Ptr{BlasInt}, Ptr{$elty}, Ptr{BlasInt},
                      Ptr{$elty}, Ptr{$elty}, Ptr{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}),
                      &jobz, &uplo, &n, A, &max(1,stride(A,2)), W, work, &lwork, iwork, &liwork, info)
                @lapackerror
                if lwork < 0
                    lwork = blas_int(real(work[1]))
                    work = Array($elty, lwork)
                    liwork = iwork[1]
                    iwork = Array(BlasInt, liwork)
                end
            end
            jobz=='V' ? (W, A) : W
        end
    end
end


function data_svd!{T<:FloatingPoint}(X::Matrix{T})
    _U, D, Vᵀ = LAPACK.gesdd!('S', X)
    (transpose(Vᵀ), d)
end

function data_eig!{T<:FloatingPoint}(Σ::Matrix{T}, tol::T = eps(T)*maximum(size(Σ))*maximum(Σ))
    Λ, V = LAPACK.syev!('V', 'U', Σ)  # V*diag(λ)*Vᵀ = Σ
    (V, Λ[length(Λ):-1:1])
end


# Assumes X is one observation per row
# Assumes M is one mean per row

function data_gsvd!{T<:FloatingPoint}(M::Matrix{T}, X::Matrix{T}, tol::T = eps(T)*maximum(size(X))*maximum(X))
    (p = size(X,2)) == size(M, 2) || throw(ArgumentError("X and M must have the same number of columns."))
    n = size(X, 1)
    (m = size(M, 1)) <= n || throw(ArgumentError("M must have fewer rows than X."))
    _U, _W, Q, Dm, Dx, k, l, R = LAPACK.ggsvd!('N', 'N', 'Q', M, X)  # UᵀMQ = Σ₁[0 R], WᵀXQ = Σ₂[0 R]
    k == 0 || error("Generalised SVD failed because range(M) is not a subset of range(X)")
    D = Array(T, l)
    for i = 1:l
        D[i] = Dm[i] / Dx[i]
    end
    σ = sortperm(D, alg = QuickSort, rev = true)
    D = D[σ]  # Sort eigenvalues
    LAPACK.trtri!('U', 'N', R)  # Invert R
    if tol <= 0
        error("not written")
    else
        d = 0
        for i = 1:l
            if D[i] >= tol
                d += 1
            else
                break
            end
        end
        if d < l
            D = D[1:d]
        end
        V = BLAS.gemm('N', 'N', Q, l == p ? R[:,σ] : [zeros(T,p-l,l) ; R[:,σ[1:l]]])
    end

    if tol <= 0
        D = Array(T, p)
        if l < p
            for i = 1:l
                D[i] = Dm[i] / Dx[i]
            end
            for i = (l+1):p
                D[i] = zero(T)
            end
        else
            for i = 1:p
                D[i] = Dm[i] / Dx[i]
            end
        end
        σ = sortperm(D, alg = QuickSort, rev = true)
        D = D[σ]    # sort eigenvalues
        LAPACK.trtri!('U', 'N', R)  # Invert R
        V = BLAS.gemm('N', 'N', Q, l == p ? R[:,σ] : [zeros(T,p-l,l) ; R[:,σ[1:l]]])
        scale!(V, 1 ./ Dx[σ[1:l]])   # Normalize rows to ensure Σx orthogonality
        return (V, D)
    else
        error("whoops")
    end
end

#=
function data_gsvd!{T<:FloatingPoint}(M::Matrix{T}, X::Matrix{T}, tol::T = eps(T)*maximum(size(X))*maximum(X))
    (p = size(X,2)) == size(M, 2) || throw(ArgumentError("X and M must have the same number of columns."))
    n = size(X, 1)
    (m = size(M, 1)) <= n || throw(ArgumentError("M must have fewer rows than X."))
    _U, _W, Q, Dm, Dx, k, l, R = LAPACK.ggsvd!('N', 'N', 'Q', M, X)  # UᵀMQ = Σ₁[0 R], WᵀXQ = Σ₂[0 R]
    k == 0 || error("Generalised SVD failed because range(M) is not a subset of range(X)")  # M must have been computed incorrectly or FP precision...
    if tol <= 0
        D = Array(T, p)
        if l < p
            for i = 1:l
                D[i] = Dm[i] / Dx[i]
            end
            for i = (l+1):p
                D[i] = zero(T)
            end
        else
            for i = 1:p
                D[i] = Dm[i] / Dx[i]
            end
        end
        σ = sortperm(D, alg = QuickSort, rev = true)
        D = D[σ]    # sort eigenvalues
        LAPACK.trtri!('U', 'N', R)  # Invert R
        V = BLAS.gemm('N', 'N', Q, l == p ? R[:,σ] : [zeros(T,p-l,l) ; R[:,σ[1:l]]])
        scale!(V, 1 ./ Dx[σ[1:l]])   # Normalize rows to ensure Σx orthogonality
        return (V, D)
    else
        error("whoops")
    end
end
=#

#=
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
=#


function cov_geig!{T<:FloatingPoint}(S_m::Matrix{T}, S_x::Matrix{T}, tol::T = eps(T)*maximum(size(S_x))*maximum(S_x))
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

function wsvd!{T<:FloatingPoint}(X::Matrix{T}, Wu::Matrix{T}, Wv::Matrix{T})
    (n = size(X, 1)) == size(Wu, 1) == size(Wu, 2) || throw(DimensionMismatch("The order of Wu must match the number of rows of X."))
    (m = size(X, 2)) == size(Wv, 1) == size(Wv, 2) || throw(DimensionMismatch("The order of Wv must match the number of columns of X."))
    Λu, Qu = LAPACK.syev!('V', 'U', Wu)  # QΛQᵀ = Wu
    Λv, Qv = LAPACK.syev!('V', 'U', Wv)  # QΛQᵀ = Wv
    U, D, Vᵀ = LAPACK.gesdd!('S', S_u * X * S_v)
end
