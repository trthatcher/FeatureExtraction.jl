n = 20
p = 5

X = rand(n,p)
μ = mean(X,1)

Xc = (X .- μ)

V, D = pca!(copy(X))
