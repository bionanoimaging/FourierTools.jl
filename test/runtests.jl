using Random, Test

@testset "Test resample method" begin
    @testset "Test that upsample and downsample is reversible" begin
        Random.seed!(42)
        for dim = 1:5
            for _ in 1:4
                s_small = ntuple(_ -> rand(1:13), dim)
                s_large = ntuple(i -> max.(s_small[i], rand(10:16)), dim)
                
                x = randn(Float32, (s_small))
                @test x ≈ resample(resample(x, s_large), s_small)
                # @test x ≈ resample_rfft(resample(x, s_large), s_small)
                x = randn(ComplexF32, (s_small))
                @test x ≈ resample(resample(x, s_large), s_small)
                @test x ≈ resample(resample(real(x), s_large), s_small) + 1im .* resample(resample(imag(x), s_large), s_small) 
            end
        end

    end

end
