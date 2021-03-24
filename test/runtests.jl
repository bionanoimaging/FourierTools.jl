using Random, Test
using FourierTools

@testset "Test resampling  methods" begin
    @testset "Test that upsample and downsample is reversible" begin
        Random.seed!(42)
        for dim = 1:4
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


    @testset "Tests that resample_by_FFT is purely real" begin
        function test_real(s_1, s_2)
            x = randn(Float32, (s_1))
            y = resample_by_FFT(x, s_2, take_real=false)
            @test all(( imag.(y) .+ 1 .≈ 1))
        end

        Random.seed!(42)
        for dim = 1:4
            for _ in 1:5
                s_1 = ntuple(_ -> rand(1:13), dim)
                s_2 = ntuple(i -> rand(1:13), dim)
                test_real(s_1, s_2)
            end
        end
            
        test_real((4, 4),(6, 6))
        test_real((4, 4),(6, 7))
        test_real((4, 4),(9, 9))
        test_real((4, 5),(9, 9))
        test_real((4, 5),(9, 8))
        test_real((8, 8),(6, 7))
        test_real((8, 8),(6, 5))
        test_real((8, 8),(4, 5))
        test_real((9, 9),(4, 5))
        test_real((9, 9),(4, 5))
        test_real((9, 9),(7, 8))
        test_real((9, 9),(6, 5))

    end
end
