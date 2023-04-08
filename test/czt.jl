using NDTools # this is needed for the select_region! function below.

@testset "chirp z-transformation" begin
    @testset "czt" begin
        x = opt_cu(randn(ComplexF32, (5,6,7)), use_cuda)
        @test eltype(czt(x, (2.0,2.0,2.0))) == ComplexF32
        @test eltype(czt(x, (2f0,2f0,2f0))) == ComplexF32
        y = opt_cu(randn(ComplexF32, (5,6)), use_cuda)
        zoom = (1.0,1.0,1.0)
        @test ≈(czt(x, zoom), copy(ft(x)),rtol=1e-4)
        @test ≈(czt(y, (1.0,1.0)), copy(ft(y)),rtol=1e-5)
        @test ≈(iczt(czt(y, (1.0,1.0)), (1.0,1.0)),  y, rtol=1e-5)
        zoom = (2.0,2.0)
        @test ≈(czt(y,zoom),  NDTools.select_region(upsample2(ft(y), fix_center=true), new_size=size(y)), rtol=1e-5)
        # zoom smaller 1.0 causes wrap around:
        zoom = (0.5,2.0)
        @test abs(Array(czt(y,zoom))[1,1]) > 1e-5
        zoom = (0.5,2.0)
        # check if the remove_wrap works
        @test abs(Array(czt(y,zoom; remove_wrap=true))[1,1]) == 0.0
        @test abs(Array(iczt(y,zoom; remove_wrap=true))[1,1]) == 0.0
    end
end
