using NDTools # this is needed for the select_region! function below.

@testset "chirp z-transformation" begin
    @testset "czt" begin
        x = randn(ComplexF32, (5,6,7))
        y = randn(ComplexF32, (5,6))
        zoom = (1.0,1.0,1.0)
        @test ≈(czt(x, zoom), ft(x),rtol=1e-4)         
        @test ≈(czt(y, zoom), ft(y),rtol=1e-5)
        zoom = (2.0,2.0)
        @test ≈(czt(y,zoom),  select_region!(upsample2(ft(y), fix_center=true),new_size=size(y)), rtol=1e-5)
        # @vt czt(y,zoom)  select_region(upsample2(ft(y), fix_center=true), new_size=size(y))
    end
end
