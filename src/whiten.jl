#===================================================================================================
  Whitening Module
===================================================================================================#

module WHITEN

require("base_types.jl")
require("lib_AUX.jl")



#===================================================================================================
  Whiten Types
===================================================================================================#

type TransformCenterWhiten <: DataTransform
	W::Matrix{Float64}	# Whiten matrix
	μ::Matrix{Float64}	# Mean vector (row)
end

type TransformWhiten <: DataTransform
	W::Matrix{Float64}	# Decorrelation matrix
end

transform(obj::TransformCenterWhiten, Z::Matrix{Float64}) =  (Z .- obj.μ) * obj.W
transform(obj::TransformWhiten, Z::Matrix{Float64}) = Z * obj.W


#===================================================================================================
  Covariance Matrix Whitening
===================================================================================================#

# white_vals_eig!:
#	Return the decorrelation matrix W for matrix Σ using an eigendecomposition
#	 • Σ is an n×n matrix proportional to a covariance matrix
#	 • symmetric == true returns W such that WWᵗ = W², otherwise W such that WWᵗ is diagonal
#	 • w is a scaling factor such that Σ/w is a covariance matrix
function whiten_eig!(Σ::Matrix{Float64}, symmetric::Bool=false, w::Real=1; tol::Float64=1e-8)
	tol >= 0.0 || error("tol=$tol is less than zero")
	n,p = size(Σ)
	n == p || error("Dimensions do not conform")
	λ,V = LAPACK.syev!('V','U',Σ)   # V*diag(λ)*Vᵗ = Σ
	for i=1:p
		λ[i] = abs(λ[i]) < tol ? 0.0 : sqrt(float64(w) / λ[i]) 
	end
	if symmetric
		return BLAS.gemm('N', 'N', 1.0, V, AUX.ldmm!(λ,transpose(V)))
	else
		return AUX.rdmm!(V, λ)
	end
end

# white_vals_eig!:
#	Return the decorrelation matrix W for matrix Σ using a Cholesky decomposition
#	 • Σ is an n×n matrix proportional to a covariance matrix
#	 • w is a scaling factor such that Σ/w is a covariance matrix
function whiten_chol!(Σ::Matrix{Float64}, w::Real=1)
	n,p = size(Σ)
	n == p || error("Dimensions do not conform")
	C = (LAPACK.potrf!('U',Σ))[1]   # C'C = Σ
	triu!(C)
	LAPACK.trtri!('U','N',C)
	if w != 1 BLAS.scal!(p^2, sqrt(float64(w)),C,1) end
	return C
end

# whiten_cov!:
#	Return the decorrelation matrix W for matrix Σ
#	 • Σ is an n×n covariance matrix
#	 • α is a regularization parameter
#	 • ϵ is a perturbation parameter
#	 • w is a scaling factor such that Σ/w is a covariance matrix
function whiten_cov!(Σ::Matrix{Float64}, α::Real=0, ϵ::Real=0, method::String="pcw", w::Real=1)
	0 <= α <= 1 || error("α=$α not in [0,1]")
	0 <= ϵ || error("ϵ=$ϵ less than zero")
	if α != 0 AUX.regularize!(Σ,α) end
	if ϵ != 0 AUX.perturb!(Σ,ϵ) end
	if method == "pcw"
		return whiten_eig!(Σ,false,w)
	elseif method == "zcw" 
		return whiten_eig!(Σ,true,w)
	elseif method == "cdw"
		return whiten_chol!(Σ,w)
	else 
		error("Method not recognized") 
	end
end

whiten_cov(Σ::Matrix{Float64}; alpha::Real=0, epsilon::Real=0, method::String="pcw") = TransformWhiten(whiten_cov!(copy(Σ), alpha, epsilon, method))
pcw_cov(Σ::Matrix{Float64}; alpha::Real=0, epsilon::Real=0) =  whiten_cov(Σ, alpha, epsilon, "pcw")
zcw_cov(Σ::Matrix{Float64}; alpha::Real=0, epsilon::Real=0) =  whiten_cov(Σ, alpha, epsilon, "zcw")
cdw_cov(Σ::Matrix{Float64}; alpha::Real=0, epsilon::Real=0) =  whiten_cov(Σ, alpha, epsilon, "cdw")


#===================================================================================================
  Data Matrix Whitening
===================================================================================================#

# whiten_svd!:
#	Return the decorrelation matrix W for data matrix X using a singular value decomposition
#	 • X is an n×p data matrix
#	 • symmetric == true returns W such that WWᵗ = W², otherwise W such that WWᵗ is diagonal
#	 • w is a scaling factor such that XᵗX/w is a covariance matrix
function whiten_svd!(X::Matrix{Float64}, symmetric::Bool=false, w::Real=(size(X)[1]-1), tol=1e-8)
	σ,Vᵗ = LAPACK.gesdd!('S',X)[2:3]   # X = U*diag(σ)*Vᵗ -> V*diag(σ)^2*Vᵗ ~ Σ
	for i=1:length(σ) 
		σ[i] = abs(σ[i]) < tol ? 0.0 : sqrt(float64(w)) / σ[i] 
	end
	if symmetric
		return BLAS.gemm('N', 'N', 1.0, AUX.rdmm!(transpose(Vᵗ),σ),Vᵗ)
	else
		return AUX.rdmm!(transpose(Vᵗ),σ)
	end
end

# whiten_qr!:
#	Return the decorrelation matrix W for data matrix X using a QR decomposition
#	 • X is an n×p data matrix
#	 • w is a scaling factor such that XᵗX/w is a covariance matrix
function whiten_qr!(X::Matrix{Float64}, w::Real=1)
	n,p = size(X)
	R = (LAPACK.geqrf!(X)[1])[1:p,:] 
	# X = Q*R -> V*diag(σ)^2*Vᵗ ~ Σ
	triu!(R)
	LAPACK.trtri!('U','N',R)
	if w != 1 scal!(p^2, float64(w),R,1) end
	return R
end

# whiten!:
#	Return the decorrelation matrix W for data matrix X
#	 • X is an n×p data matrix
#	 • α is a regularization parameter
#	 • ϵ is a perturbation parameter
#	 • method is a string defining which whitening method to employ
function whiten!(X::Matrix{Float64}, α::Real=0, ϵ::Real=0; method::String="pcw")
	n,p = size(X)
	if α == 0 & ϵ == 0
		if method == "pcw"
			return whiten_svd!(X, false, n)
		elseif method == "zcw" 
			return whiten_svd!(X, true, n)
		elseif method == "cdw"
			return whiten_qr!(X,n)
		else 
			error("Method not recognized") 
		end
	else 
		Σ = BLAS.syrk('U', 'T', 1, X)   # XᵗX
		return whiten_cov!(Σ,α,ϵ,method,n-1)
	end
end

# whiten: Wrapper for whiten!
function whiten(X::Matrix{Float64}; alpha::Real=0, epsilon::Real=0, method::String="pcw", center::Bool=true)
	if center == true
		μ = mean(X,1)
		return TransformCenterWhiten(whiten!(X .- μ, alpha, epsilon, method), μ)
	else
		return TransformWhiten(whiten!(copy(X), alpha, epsilon, method))
	end
end
pcw(X::Matrix{Float64}; alpha::Real=0, epsilon::Real=0, center::Bool=true) =  W_Trans(whiten_data!(copy(Z), alpha, epsilon, "pcw", center))
zcw(X::Matrix{Float64}; alpha::Real=0, epsilon::Real=0, center::Bool=true) =  W_Trans(whiten_data!(copy(Z), alpha, epsilon, "zcw", center))
cdw(X::Matrix{Float64}; alpha::Real=0, epsilon::Real=0, center::Bool=true) =  W_Trans(whiten_data!(copy(Z), alpha, epsilon, "cdw", center))

end # WHITEN

#function white_vals_qr(X::Matrix{Float64}, w::Real)
#	n,p = size(X)
#	R = (LAPACK.geqrf!(copy(X))[2])[1:p,:]
#	triu!(R)
#	try LAPACK.trtri!('U','N',R)
#	catch error
#		if error > 0 
#			warn("Inverse inable")
#			# do something
#		else
#			error("Error inverting")
#		end
#	end
#	if w != 1 scal!(p^2, float64(w),R,1) end
#	return R
#end


