@testset "Correlations methods" begin
     
    @test ccorr(opt_cu([1, 0], use_cuda), opt_cu([1, 0], use_cuda), centered = true) == opt_cu([0.0, 1.0], use_cuda)
    @test ccorr(opt_cu([1, 0], use_cuda), opt_cu([1, 0], use_cuda)) == opt_cu([1.0, 0.0], use_cuda)
    
    x = opt_cu([1,2,3,4,5], use_cuda)
    y = opt_cu([1,2,3,4,5], use_cuda)
    @test ccorr(x,y) ≈ opt_cu([55, 45, 40, 40, 45], use_cuda)
    @test ccorr(x,y, centered=true) ≈ opt_cu([40, 45, 55, 45, 40], use_cuda)

    @test ccorr(x, x .* (1im)) ≈ opt_cu(ComplexF64[0.0 - 55.0im, 0.0 - 45.0im, 0.0 - 40.0im, 0.0 - 40.0im, 0.0 - 45.0im], use_cuda)
end
