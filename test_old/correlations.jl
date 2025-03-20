

@testset "Correlations methods" begin
     
    @test ccorr([1, 0], [1, 0], centered = true) == [0.0, 1.0]
    @test ccorr([1, 0], [1, 0]) == [1.0, 0.0]
    
    x = [1,2,3,4,5]
    y = [1,2,3,4,5]
    @test ccorr(x,y) ≈ [55, 45, 40, 40, 45]
    @test ccorr(x,y, centered=true) ≈ [40, 45, 55, 45, 40]

    @test ccorr(x, x .* (1im)) == ComplexF64[0.0 - 55.0im, 0.0 - 45.0im, 0.0 - 40.0im, 0.0 - 40.0im, 0.0 - 45.0im]
end
