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


