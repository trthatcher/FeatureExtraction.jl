#===================================================================================================
  Auxiliary Functions
===================================================================================================#

#==========================================================================
  Class Matrices
==========================================================================#

immutable Factor
    code::Array{Int64}
    k::Int64
    n::Int64
    function Factor{T<:Integer}(y::Array{T})
        min_code = y[1]
        max_code = y[1]
        n = length(y)
        @inbounds for i = 2:n
            if y[i] < min_code min_code = y[i] end
            if y[i] > max_code max_code = y[i] end
        end
        min_code == 1 || throw(ArgumentError("Vector must have minimum value = 1."))
        @inbounds for j = 1:max_code
            for i = 1:n
                if y[i] == j break end
                i != n || throw(ArgumentError("Vector does not contain value $(j)."))
            end
        end
        new(convert(Array{Int64}, y), convert(Int64, max_code), n)
    end
end

# Center X using the rows of M and index y[i]
function center_rows!{T<:FloatingPoint}(y::Factor, X::Matrix{T}, M::Matrix{T})
    n, p = size(X)
    y.k == size(M,1) || throw(ArgumentError("M must have as many rows as classes."))
    idx = y.code
    @inbounds for j = 1:p 
        for i = 1:n
            X[i,j] -= M[idx[i],j]
        end
    end
    X
end

function class_sums{T<:FloatingPoint}(y::Factor, X::Matrix{T})
    n, p = size(X)
    n == y.n || throw(ArgumentError("Factor y and matrix X must have the same number of rows"))
    M = zeros(T, y.k, p)
    idx = y.code
    @inbounds for j = 1:p
        for i = 1:n
            M[idx[i],j] += X[i,j]
        end
    end
    M
end

function class_counts(y::Factor)
    c = zeros(Int64, y.k)
    idx = y.code
    @inbounds for i = 1:y.n
        c[idx[i]] += 1
    end
    c
end

function class_means{T<:FloatingPoint}(y::Factor, X::Matrix{T})
    M::Matrix{T} = class_sums(y, X)
    c::Vector{Int64} = class_counts(y)
    dgmm!(one(T) ./ c, M)
end
