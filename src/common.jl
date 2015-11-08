function translate!{T<:AbstractFloat}(A::Array{T}, b::T)
    @inbounds for i = 1:length(A)
        A[i] += b
    end
    A
end
translate!{T<:AbstractFloat}(b::T, A::Array{T}) = translate!(A, b)

function translate!{T<:AbstractFloat}(A::Matrix{T}, b::Vector{T})
    n, p = size(A)
    p == length(b) || throw(DimensionMismatch("Vector b must have the same length as A has columns."))
    for j = 1:p, i = 1:n
        A[i,j] += b[j]
    end
    A
end

function translate!{T<:AbstractFloat}(b::Vector{T}, A::Matrix{T})
    n, p = size(A)
    n == length(b) || throw(DimensionMismatch("Vector b must have the same length as A has rows."))
    for j = 1:p, i = 1:n
        A[i,j] += b[i]
    end
    A
end

function count_nonzero{T<:AbstractFloat}(v::Vector{T}, tol::T)
    d = 0
    @inbounds for i = 1:length(D)
        if D[i] > tol
            d += 1
        else
            break
        end
    end
    d
end
