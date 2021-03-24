using Random, Test
using FourierTools

@testset "Test resample_by_FFT method" begin
    @testset "Test that upsample and downsample is reversible" begin
        Random.seed!(42)
        for dim = 1:5
            for _ in 1:4
                s_small = ntuple(_ -> rand(1:13), dim)
                s_large = ntuple(i -> max.(s_small[i], rand(10:16)), dim)
                
                x = randn(Float32, (s_small))
                @test x ≈ resample_by_FFT(resample_by_FFT(x, s_large), s_small)
                @test x ≈ resample_by_RFFT(resample_by_RFFT(x, s_large), s_small)
                x = randn(ComplexF32, (s_small))
                @test x ≈ resample_by_FFT(resample_by_FFT(x, s_large), s_small)
                @test x ≈ resample_by_FFT(resample_by_FFT(real(x), s_large), s_small) + 1im .* resample_by_FFT(resample_by_FFT(imag(x), s_large), s_small) 
            end
        end

    end

end
