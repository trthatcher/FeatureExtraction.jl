print("- Testing matrix functions ... ")
for T in (Float32,Float64,BigFloat)
    S1 = T[1 2 3;
           4 5 6;
           7 8 9]
    SL = T[1 2 3;
           2 5 6;
           3 6 9]
    SU = T[1 4 7;
           4 5 8;
           7 8 9]
    dr = T[1*1+2*2+3*3, 77, 194]
    dc = T[1*1+4*4+7*7, 93, 126]
    w = T[1;2;3]
    wdr = vec(sum((S1 .* S1) .* w',2))
    wdc = vec(sum((S1 .* S1) .* w,1))
    row = T[11 12 13]
    col = T[11;12;13]
    RA = T[12 14 16;
           15 17 19;
           18 20 22]
    CA = T[12 13 14;
           16 17 18;
           20 21 22]
    RS = T[-10 -10 -10;
           -7 -7 -7;
           -4 -4 -4]
    CS = T[-10 -9 -8;
           -8 -7 -6;
           -6 -5 -4]
    diag = T[1, 2, 3]
    @test DataDecomposition.syml(S1) == SL
    @test DataDecomposition.symu(S1) == SU
    @test DataDecomposition.dot_rows(S1) == dr
    @test DataDecomposition.dot_columns(S1) == dc
    @test DataDecomposition.dot_rows(S1,w) == wdr
    @test DataDecomposition.dot_columns(S1,w) == wdc
    S = T[2 2 2;
          2 2 2;
          2 2 2]
    @test DataDecomposition.matrix_prod!(T[3; 2], T[3; 2]) == T[9; 4]
    @test DataDecomposition.matrix_sum!(T[3; 2], T[3; 2]) == T[6; 4]
    @test DataDecomposition.translate!(T[3; 2], one(T)) == T[4; 3]
    @test DataDecomposition.translate!(one(T), T[3; 2]) == T[4; 3]

end
println("Done")