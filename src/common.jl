# TODO:
#   Figure out ARPACK's calling:
#     Symmetric Eigenvectors/values
#     Generalised Symmetric Eigenvectors/values
#   Created weighted svd for MCA

function translate!{T<:AbstractFloat}(A::Array{T}, b::T)
    @inbounds for i = 1:length(A)
        A[i] += b
    end
    A
end
translate!{T<:AbstractFloat}(b::T, A::Array{T}) = translate!(A, b)

function translate!{T<:AbstractFloat}(b::Vector{T}, A::Matrix{T})
    (n = size(A,1)) == length(b) || throw(DimensionMismatch("first dimension of A does not match length of b"))
    @inbounds for j = 1:size(A,2), i = 1:n
        A[i,j] += b[i]
    end
    A
end

function translate!{T<:AbstractFloat}(A::Matrix{T}, b::Vector{T})
    (n = size(A,2)) == length(b) || throw(DimensionMismatch("second dimension of A does not match length of b"))
    @inbounds for j = 1:n, i = 1:size(A,1)
        A[i,j] += b[j]
    end
    A
end

# A := A + ϵI
function perturb!{T<:AbstractFloat}(A::Matrix{T}, ϵ::T)
    (n = size(A,1)) == size(A,2) || throw(DimensionMismatch("Matrix A must be square."))
    @inbounds for i = 1:n
        A[i,i] += ϵ
    end
    A
end

# A := (1-α)A + αβ
function regularize!{T<:AbstractFloat}(A::Matrix{T}, α::T, β::T)
    (n = size(A,1)) == size(A,2) || throw(DimensionMismatch("Matrix A must be square."))
    @inbounds for j = 1:n
        for i = 1:n
            A[i,j] *= (one(T)-α)
        end
        A[j,j] += α*β
    end
    A
end

# Symmetrize the lower half of matrix S using the upper half of S
function syml!(S::Matrix)
    (p = size(S,1)) == size(S,2) || throw(ArgumentError("S ∈ ℝ$(p)×$(size(S, 2)) must be square"))
    @inbounds for j = 1:(p - 1), i = (j + 1):p 
        S[i, j] = S[j, i]
    end
    S
end
syml(S::Matrix) = syml!(copy(S))

# Symmetrize the upper off-diagonal of matrix S using the lower half of S
function symu!(S::Matrix)
    (p = size(S,1)) == size(S,2) || throw(ArgumentError("S ∈ ℝ$(p)×$(size(S, 2)) must be square"))
    @inbounds for j = 2:p, i = 1:(j-1)
        S[i,j] = S[j,i]
    end
    S
end
symu(S::Matrix) = symu!(copy(S))


#========================
 Import Divide & Conquer BLAS Algorithm
========================#

const liblapack = Base.liblapack_name

import Base.blasfunc

import Base.LinAlg: BlasFloat, BlasInt, LAPACKException,
    DimensionMismatch, SingularException, PosDefException, chkstride1, chksquare

typealias BlasChar Char

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
            lwork = convert(BlasInt, -1)
            iwork  = Array(BlasInt, 1)
            liwork = convert(BlasInt, -1)
            info  = Array(BlasInt, 1)
            for i in 1:2
                ccall(($(blasfunc(syevd)), liblapack), Void,
                     (Ptr{BlasChar}, Ptr{BlasChar}, Ptr{BlasInt}, Ptr{$elty}, Ptr{BlasInt},
                      Ptr{$elty}, Ptr{$elty}, Ptr{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}),
                      &jobz, &uplo, &n, A, &max(1,stride(A,2)), W, work, &lwork, iwork, &liwork, info)
                @lapackerror
                if lwork < 0
                    lwork = convert(BlasInt, real(work[1]))
                    work = Array($elty, lwork)
                    liwork = iwork[1]
                    iwork = Array(BlasInt, liwork)
                end
            end
            jobz=='V' ? (W, A) : W
        end
    end
end


#========================
  Eigen Solvers
========================#

# Singular Value Decomposition
#   Vectors in increasing order
function components_svd!{T<:AbstractFloat}(X::Matrix{T})
    _U, D, Vᵀ = LAPACK.gesdd!('S', X)
    (transpose(Vᵀ), D)
end

# Divide and conquer singular value decomposition
function components_eig!{T<:AbstractFloat}(S::Matrix{T})
    (p = size(A,1)) == size(A,2) || throw(DimensionMismatch("Matrix A must be square."))
    D, V = LAPACK.syev!('V', 'U', S)  # S = VDVᵀ
    V, D[end:-1:1]
end

#tol::T = eps(T)*maximum(size(S_x))*maximum(S_x)
function components_geig!{T<:AbstractFloat}(S_m::Matrix{T}, S_x::Matrix{T})
    (p = size(S_x, 1)) == size(S_x, 2) || throw(DimensionMismatch("Covariance matrix for X must be square."))
    size(S_m, 2) == size(S_m, 2)       || throw(DimensionMismatch("Covariance matrix for M must be square."))
    p == size(S_m, 2)                  || throw(DimensionMismatch("Covariance matrices for X and M must be of the same order."))
    D, V, _U = LAPACK.sygvd!(1, 'V', 'U', S_m, S_x)
    (V[:,end:-1:1], D[end:-1:1])
end

# MCA - WIP weighted svd
#=
function wsvd!{T<:AbstractFloat}(X::Matrix{T}, Wu::Matrix{T}, Wv::Matrix{T})
    (n = size(X, 1)) == size(Wu, 1) == size(Wu, 2) || throw(DimensionMismatch("The order of Wu must match the number of rows of X."))
    (m = size(X, 2)) == size(Wv, 1) == size(Wv, 2) || throw(DimensionMismatch("The order of Wv must match the number of columns of X."))
    Λu, Qu = LAPACK.syev!('V', 'U', Wu)  # QΛQᵀ = Wu
    Λv, Qv = LAPACK.syev!('V', 'U', Wv)  # QΛQᵀ = Wv
    U, D, Vᵀ = LAPACK.gesdd!('S', S_u * X * S_v)
end
=#
