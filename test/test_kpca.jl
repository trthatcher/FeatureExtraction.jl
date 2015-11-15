using MLKernels;
using DataDecomposition;

n = 20  # Number of observations
p = 5   # Number of parameters


#for T in

X = rand(n,p)
μ = mean(X,1)

Xc = (X .- μ)

κ = PolynomialKernel(1.0,0.0,1.0)

K = centerkernelmatrix!(kernelmatrix(κ, X))

V, D = pca!(copy(X))
V, D = kpca!(copy(X),κ)

#@test_approx_eq diag(V'Xc'Xc*V)*(n-1) D
