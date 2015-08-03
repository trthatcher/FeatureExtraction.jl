n = 20
n1 = convert(Int64,trunc(n/2))
n2 = n - n1

p = 3

X = rand(n,p)

mu_1 = mean(X[1:n1,:],1)
mu_2 = mean(X[(n1+1):end,:],1)

M = [mu_1; mu_2]

println("Test data_svd!(X)")

V, lambda = DataDecomposition.data_svd!(copy(X))

@test_approx_eq diagm(lambda) V'X'X*V

println("Test data_svd!(M,X)")

V, lambda = DataDecomposition.data_gsvd!(copy(M),copy(X))

@test_approx_eq diagm(lambda) V'M'M*V

println("Test X'X orthogonality")
@test_approx_eq eye(length(lambda)) V'X'X*V
