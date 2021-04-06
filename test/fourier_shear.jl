@testset "Fourier Shear" begin

  
    @testset "Complex and real shear produce similar results" begin
        function f(a, b, Δ)
            x = randn((30, 24, 13))
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
            x = randn((30, 24, 13))
            xc = randn(ComplexF32, (30, 24, 13)) 
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
end
