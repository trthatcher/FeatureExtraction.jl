using DataDecomposition;
nc = 3; p = 3; k = 4; y = repeat(Int64[i for i = 1:k],inner=[nc],outer=[1]);  X = vcat([rand(nc,p) .+ 10rand(1,3) for i = 1:k]...); freq = Float64[1/k for i=1:k];
lda!(copy(X), y, k, freq)
