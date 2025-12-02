
@testset "Custom Fourier Types" begin
    N = 5
    x = opt_cu(randn((N, N)), use_cuda)
    fs = FourierTools.FourierSplit(x, Val(2), 2, 4, true)
    @test FourierTools.parenttype(fs) == typeof(x)
    @test Array(fs)[1,1] == Array(fs)[1]
    @test Array(fs)[1,2] == Array(fs)[6]
    fs = FourierTools.FourierSplit(x, Val(2), 2, 4, false)
    @test FourierTools.parenttype(fs) == typeof(x)
    
    fj = FourierTools.FourierJoin(x, Val(2), 2, 4, true)
    @test FourierTools.parenttype(fj) == typeof(x)
    @test Array(fj)[1,1] == Array(fj)[1]
    @test Array(fj)[1,2] == Array(fj)[6]

    fj = FourierTools.FourierJoin(x, Val(2), 2, 4, false)
    @test FourierTools.parenttype(fj) == typeof(x)

    @test FourierTools.parenttype(typeof(fj)) == typeof(x)

    @test FourierTools.IndexStyle(typeof(fj)) == IndexStyle(typeof(fj))

    x = opt_cu(ones((4, 7)), use_cuda)
    fs = FourierTools.FourierSplit(x, Val(2), 2, 4, true)
    fj = FourierTools.FourierJoin(x, Val(2), 2, 4, true)
    @test all(fs[:,2] .== 0.5)
    @test all(fj[:,2] .== 2)

end
