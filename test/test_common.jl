info("Testing ", MOD.translate!)
for T in FloatingPointTypes
    A = T[1 2;
          3 4;
          5 6]

    b = T[1;
          2]

    c = T[1;
          2;
          3]

    @test_approx_eq MOD.translate!(copy(A), one(T)) (A .+ one(T))
    @test_approx_eq MOD.translate!(one(T), copy(A)) (A .+ one(T))
    @test_approx_eq MOD.translate!(copy(A), b) (A .+ b')
    @test_approx_eq MOD.translate!(c, copy(A)) (A .+ c)
end

info("Testing ", MOD.perturb!)
for T in FloatingPointTypes
    A = T[1 2 3;
          4 5 6;
          7 8 9]

    B = T[1 2;
          3 4;
          5 6]

    @test_approx_eq MOD.perturb!(copy(A), one(T)) (A + one(T)*I)
    @test_throws DimensionMismatch MOD.perturb!(B, one(T))
end

info("Testing ", MOD.regularize!)
for T in FloatingPointTypes
    A = T[1 2 3;
          4 5 6;
          7 8 9]

    B = T[1 2;
          3 4;
          5 6]

    a = trace(A)/3

    @test_approx_eq MOD.regularize!(copy(A), zero(T), a) A
    @test_approx_eq MOD.regularize!(copy(A), convert(T,0.5), a) convert(T,0.5)*(A + a*I)
    @test_approx_eq MOD.regularize!(copy(A), one(T), a) a*eye(T,3)
    @test_throws DimensionMismatch MOD.regularize!(B, one(T), one(T))
end


A  = [1 2 3;
      4 5 6;
      7 8 9]
AL = [1 2 3;
      2 5 6;
      3 6 9]
AU = [1 4 7;
      4 5 8;
      7 8 9]

info("Testing ", MOD.syml)
for T in FloatingPointTypes
    B = MOD.syml(convert(Array{T},A))
    @test eltype(B) == T
    @test_approx_eq B convert(Array{T}, AL)
end

info("Testing ", MOD.symu)
for T in FloatingPointTypes
    B = MOD.symu(convert(Array{T},A))
    @test eltype(B) == T
    @test_approx_eq B convert(Array{T}, AU)
end

info("Testing ", MOD.components_svd!)
for T in FloatingPointTypes
    X = rand(T,20,5)
    V1, D1 = MOD.components_svd!(copy(X))

    D2, V2 = svd(X)[2:3]

    @test_approx_eq V1 V2
    @test_approx_eq D1 (D2 .^ 2)
end

info("Testing ", MOD.components_eig!)
for T in FloatingPointTypes
    X = rand(T,20,5)
    S = X'X

    V1, D1 = MOD.components_eig!(copy(S))

    D2, V2 = eig(S)

    @test_approx_eq abs(V1) abs(V2[:,end:-1:1])  # Signs may differ
    @test_approx_eq abs(D1) abs(D2[end:-1:1])
end

info("Testing ", MOD.components_geig!)
for T in FloatingPointTypes
    X1 = rand(T,20,5)
    X2 = rand(T,20,5)

    S1 = X1'X1
    S2 = X2'X2

    V1, D1 = MOD.components_geig!(copy(S1), copy(S2))

    D2, V2 = eig(S1, S2)

    @test_approx_eq abs(V1) abs(V2[:,end:-1:1])
    @test_approx_eq abs(D1) abs(D2[end:-1:1])
end
