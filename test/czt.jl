using NDTools # this is needed for the select_region! function below.

@testset "chirp z-transformation" begin
    @testset "czt" begin
        x = opt_cu(randn(ComplexF32, (5,6,7)), use_cuda)
        @test eltype(czt(x, (2.0,2.0,2.0))) == ComplexF32
        @test eltype(czt(x, (2f0,2f0,2f0))) == ComplexF32
        @test ≈(czt(x, (1.0,1.0,1.0), (1,3)), ft(x, (1,3)), rtol=1e-5)
        @test ≈(czt(x, (1.0,1.0,1.0), (1,3), src_center=(1,1,1), dst_center=(1,1,1)), fft(x, (1,3)), rtol=1e-5)
        @test ≈(iczt(x, (1.0,1.0,1.0), (1,3), src_center=(1,1,1), dst_center=(1,1,1)), ifft(x, (1,3)), rtol=1e-5)

        y = randn(ComplexF32, (5,6))
        zoom = (1.0,1.0,1.0)
        @test ≈(czt(x, zoom), ft(x), rtol=1e-4)         
        @test ≈(czt(y, (1.0,1.0)), ft(y), rtol=1e-5)

        @test ≈(iczt(czt(y, (1.0,1.0)), (1.0,1.0)), y, rtol=1e-5)
        zoom = (2.0,2.0)
        @test sum(abs.(imag(czt(ones(5,6),zoom, src_center=((5,6).+1)./2)))) < 1e-8

        # for even sizes the czt is not the same as the ft and upsample operation. But should it be or not?
        # @test ≈(czt(y,zoom),  select_region(upsample2(ft(y), fix_center=true), new_size=size(y)), rtol=1e-5)
        # @test ≈(czt(y,zoom, src_center=(size(y).+1)./2), select_region(upsample2(ft(y), fix_center=true), new_size=size(y)), rtol=1e-5)

        # for uneven sizes this works:
        @test ≈(czt(y[1:5,1:5],zoom, (1,2), (10,10)),  upsample2(ft(y[1:5,1:5]), fix_center=true), rtol=1e-5)
        p_czt = plan_czt(y, zoom, (1,2), (11,12))
        @test ≈(p_czt * y, czt(y, zoom, (1,2), (11,12)))
        # zoom smaller 1.0 causes wrap around:
        zoom = (0.5,2.0)
        @test abs(czt(y,zoom)[1,1]) > 1e-5
        zoom = (2.0, 0.5)
        # check if the remove_wrap works
        @test abs(czt(y,zoom; remove_wrap=true)[1,1]) == 0.0
        @test abs(iczt(y,zoom; remove_wrap=true)[1,1]) == 0.0
        @test abs(czt(y,zoom; pad_value=0.2, remove_wrap=true)[1,1]) == 0.2f0
        @test abs(iczt(y,zoom; pad_value=0.5f0, remove_wrap=true)[1,1]) == 0.5f0
    end
end
