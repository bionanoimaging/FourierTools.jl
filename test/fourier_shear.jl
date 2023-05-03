@testset "Fourier Shear" begin

  
    @testset "Complex and real shear produce similar results" begin
        function f(a, b, Δ)
            x = opt_cu(randn((30, 24, 13)), use_cuda)
            xc = 0im .+ x
            xc2 = 1im  .* x
            @test shear(x, Δ, a, b) ≈ real(shear(xc, Δ, a, b))
            @test shear(x, Δ, a, b) ≈ imag(shear(xc2, Δ, a, b))
        end

        f(2, 3, 123.1)
        f(3, 2, 13.1)
        f(1, 2, 13.1)
        f(3, 1, 13.1)
    end

    @testset "Test that in-place works in-place" begin
        function f(a, b, Δ)
            x = opt_cu(randn((30, 24, 13)), use_cuda)
            xc = opt_cu(randn(ComplexF32, (30, 24, 13)), use_cuda)
            xc2 = 1im  .* x
            @test shear!(x, Δ, a, b) ≈ x 
            @test shear!(xc, Δ, a, b) ≈ xc 
            @test shear!(xc2, Δ, a, b) ≈ xc2
        end

        f(2, 3, 123.1)
        f(3, 2, 13.1)
        f(1, 2, 13.1)
        f(3, 1, 13.1)
    end


    @testset "Fix Nyquist" begin
        @test shear(shear([1 2; 3 4.0], 0.123), -0.123, fix_nyquist = true) == [1.0 2.0; 3.0 4.0]
        @test shear(shear([1 2; 3 4.0], 0.123), -0.123, fix_nyquist = false) != [1.0 2.0; 3.0 4.0]
    end

    @testset "assign_shear_wrap!" begin
        q = opt_cu(ones((10,11)), use_cuda)
        assign_shear_wrap!(q, 10)
        @test q[:,1] == [0,0,0,0,0,1,1,1,1,1]
    end
end
