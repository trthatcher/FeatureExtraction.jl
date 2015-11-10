# TODO:
#   Figure out ARPACK's calling:
#     Symmetric Eigenvectors/values
#     Generalised Symmetric Eigenvectors/values
#   Created weighted svd for MCA


# Singular Value Decomposition
#   Vectors in increasing order
function components_svd!{T<:AbstractFloat}(X::Matrix{T}, tolerance::T, max_dimension::Integer)
    _U, D, Vᵀ = LAPACK.gesdd!('S', X)
    d = tolerance == 0 ? max_dimension : max(count_nonzero(D, tolerance), max_dimension)
    m = length(D)
    if d < m
        V = Array(T, m, d)
        for j = 1:d, i = 1:m
            V[i,j] = Vᵀ[j,i]
        end
        return (V, D[1:d])
    else
        return (transpose(Vᵀ), D)
    end
end


function components_eig!{T<:AbstractFloat}(S::Matrix{T}, tolerance::T, max_dimension::Integer)
    (p = size(A,1)) == size(A,2) || throw(DimensionMismatch("Matrix A must be square."))
    D, V = LAPACK.syev!('V', 'U', S)  # S = VDVᵀ
    d = tol == 0 ? max_dimension : max(count_nonzero(D, tolerance), max_dimension)
    if d < p
        σ = p:-1:(p-d+1)
        return (V[:,σ], D[σ])
    else
        return (V, D)
    end
end


#tol::T = eps(T)*maximum(size(S_x))*maximum(S_x)
function components_geig!{T<:AbstractFloat}(S_m::Matrix{T}, S_x::Matrix{T}, tolerance::T, max_dimension::Integer)
    (p = size(S_x, 1)) == size(S_x, 2) || throw(DimensionMismatch("Covariance matrix for X must be square."))
    size(S_m, 2) == size(S_m, 2)       || throw(DimensionMismatch("Covariance matrix for M must be square."))
    p == size(S_m, 2)                  || throw(DimensionMismatch("Covariance matrices for X and M must be of the same order."))
    D, V, _U = LAPACK.sygvd!(1, 'V', 'U', S_m, S_x)
    d = tol == 0 ? max_dimension : max(count_nonzero(D, tolerance), max_dimension)
    (V[:,n:-1:(n-d+1)], Λ[n:-1:(n-d+1)])
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
