
@testset "Custom Fourier Types" begin
    N = 5
    x = randn((N, N))
    fs = FourierTools.FourierSplit(x, 2, 2, 4)
    @test FourierTools.parenttype(fs) == typeof(x)
    
    fj = FourierTools.FourierJoin(x, 2, 2, 4)

    @test FourierTools.parenttype(fj) == typeof(x)
    @test FourierTools.parenttype(typeof(fj)) == typeof(x)

    @test FourierTools.IndexStyle(typeof(fj)) == IndexStyle(typeof(fj))
end
