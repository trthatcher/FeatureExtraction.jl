immutable LDA_Parameters{T<:AbstractFloat,U<:Integer}
    n::U  # Observations
    p::U  # Dimensions
    k::U  # Number of classes
    freq::Nullable{Vector{T}}  # Class frequencies
    alpha_mu  ::Nullable{T}  # Regularization parameter for Σ_μ ∝ MᵀM
    alpha_x   ::Nullable{T}  # Regularization parameter for Σ_x ∝ XᵀX
    epsilon_mu::Nullable{T}  # Perturbation parameter for Σ_μ
    epsilon_x ::Nullable{T}  # parameter for Σ_x
    function LDA_Parameters(n::U, 
                            p::U,
                            k::U, 
                            freq::Nullable{Vector{T}}, 
                            α_μ::Nullable{T}, 
                            α_x::Nullable{T}, 
                            ϵ_μ::Nullable{T}, 
                            ϵ_x::Nullable{T})
        get(n) > 0 || error("n = $(n) must be positive a positive integer.")
        p > 0 || error("p = $(p) must be positive a positive integer.")
        k > 0 || error("k = $(k) must be positive a positive integer.")
        #k == length(freq) || error("Class frequencies vector must have k = $(k) entries.")
        if !isnull(ϵ_μ)
            get(ϵ_μ) >= 0 || error("ϵ = $(get(ϵ_μ)) must be a non-negative number.")
        end
        ϵ_c >= 0 || error("ϵₓ = $(ϵ_x) must be a non-negative number.")
        0 <= α_μ <= 1 || error("α = $(α_μ) must be in the inverval [0,1].")
        0 <= α_x <= 1 || error("αₓ = $(α_x) must be in the inverval [0,1].")
        new(n, p, k, freq, α_μ, α_x, ϵ_μ, ϵ_x)
    end
end

#function LDA_Parameters{T<:AbstractFloat}(n::Int64, p::Int64, k::Int64, f::Vector{T}, α::T, αₓ::T,
#                                          ϵ::T, ϵₓ::T)
#    LDA_Parameters{T}(n, p, k, f, α, αₓ, ϵ, ϵₓ)
#end

immutable LDA_Model{T<:AbstractFloat}
	Parameters::LDA_Parameters{T}
	Components::DataEigen{T}
end


#==========================================================================
  PCA Objects
==========================================================================#

immutable ParametersPCA{T<:AbstractFloat}
    algorithm::Symbol
    alpha::T  # Regularization parameter for covariance matrix
    epsilon::T   # Perturbation parameter for covariance matrix
    mu::Vector{T}
    tolerance::T
    max_dimension::Int64
    function ParametersPCA(algorithm::Symbol, α::T, ϵ::T, μ::Vector{T}, tolerance::T, max_dimension::Int64)
        ϵ >= 0 || throw(ArgumentError("ϵ = $(ϵ) must be a non-negative number."))
        0 <= α <= 1 || throw(ArgumentError("α = $(α) must be in the inverval [0,1]."))
        length(μ) == p || length(μ) == 0 || throw(ArgumentError("Mean vector must be length p = $(p) or p = 0."))
        tolerance >= 0 || throw(ArgumentError("tol = $(tol) must be a non-negative number."))
        max_dimension >= 1 || throw(ArgumentError("Must select at least one component."))
        max_dimension <= p || throw(ArgumentError("Max dimension must not exceed p = $(p)."))
        new(algorithm, n, p, α, ϵ, μ, tolerance, max_dimension)
    end
end
ParametersPCA{T<:AbstractFloat}(n::Int64, p::Int64, α::T, ϵ::T, tol::T, max_comp::Int64) = ParametersPCA{T}(n, p, α, ϵ, tol, max_comp)

immutable ComponentsPCA{T<:AbstractFloat}
    V::Matrix{T}
    D::Vector{T}
end
ComponentsPCA{T<:AbstractFloat}(μ::Vector{T}, V::Matrix{T}, d::Int64) = ComponentsPCA{T}(μ, V, d)

immutable ModelPCA{T<:AbstractFloat}
	parameters::ParametersPCA{T}
	components::ComponentsPCA{T}
end

immutable ModelKPCA{T<:AbstractFloat}
    kernel::Kernel{T}
	parameters::ParametersPCA{T}
	components::ComponentsPCA{T}
end






