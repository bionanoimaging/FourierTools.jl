Random.seed!(42)

@testset "Fourier filtering" begin

    @testset "Gaussian filter complex" begin
        sz = (21, 22)
        x = randn(ComplexF32, sz)
        sigma = (1.1,2.2)
        gf = filter_gaussian(x, sigma, real_space_kernel=false)
        # Note that this is not the same, since one kernel is generated in real space and one in Fourier space!
        # with sizes around 10, the difference is huge!
        k = gaussian(Float32, sz, sigma=sigma)
        k = k./sum(k) # different than "normal".
        gfc = conv_psf(x, k)
        @test ≈(gf,gfc, rtol=1e-2) # it is realatively inaccurate due to the kernel being generated in different places
        gfr = filter_gaussian(x, sigma, real_space_kernel=true)
        @test ≈(gfr, gfc) # it can be debated how to best normalize a Gaussian filter
        gfr = filter_gaussian(zeros(5).+1im, (1.0,), real_space_kernel=true)
        @test ≈(gfr, zeros(5).+1im) # it can be debated how to best normalize a Gaussian filter
    end

    @testset "Gaussian filter real" begin
        sz = (21, 22)
        x = randn(Float32, sz)
        sigma = (1.1, 2.2)
        gf = filter_gaussian(x, sigma, real_space_kernel=true)
        # Note that this is not the same, since one kernel is generated in real space and one in Fourier space!
        # with sizes around 10, the difference is huge!
        k = gaussian(sz, sigma=sigma)
        k = k./sum(k) # different than "normal".
        gf2 = conv_psf(x, k)
        @test ≈(gf, gf2, rtol=1e-2) # it is realatively inaccurate due to the kernel being generated in different places
        gf2 = filter_gaussian(zeros(sz), sigma, real_space_kernel=true)
        @test ≈(gf2, zeros(sz)) # it can be debated how to best normalize a Gaussian filter
    end
    @testset "Other filters" begin
        @test filter_hamming(FourierTools.delta(Float32, (3,)), border_in=0.0, border_out=1.0) ≈ [0.23,0.54, 0.23]
        @test filter_hann(FourierTools.delta(Float32, (3,)), border_in=0.0, border_out=1.0) ≈ [0.25,0.5, 0.25]
        @test FourierTools.fourier_filter_by_1D_FT!(ones(ComplexF64, 6), [ones(ComplexF64, 6)]; transform_win=true, normalize_win=false) ≈ 6 .* ones(ComplexF64, 6)
        @test FourierTools.fourier_filter_by_1D_FT!(ones(ComplexF64, 6), [ones(ComplexF64, 6)]; transform_win=true, normalize_win=true) ≈ ones(ComplexF64, 6)
        @test FourierTools.fourier_filter_by_1D_RFT!(ones(6), [ones(6)]; transform_win=true, normalize_win=false) ≈ 6 .* ones(6)
        @test FourierTools.fourier_filter_by_1D_RFT!(ones(6), [ones(6)]; transform_win=true, normalize_win=true) ≈ ones(6)
    end
end
