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
        @test ≈(gfr,gfc) # it can be debated how to best normalize a Gaussian filter
    end

    @testset "Gaussian filter real" begin
        sz = (21, 22)
        x = randn(Float32, sz)
        sigma = (1.1,2.2)
        gf = filter_gaussian(x, sigma, real_space_kernel=true)
        # Note that this is not the same, since one kernel is generated in real space and one in Fourier space!
        # with sizes around 10, the difference is huge!
        k = gaussian(sz, sigma=sigma)
        k = k./sum(k) # different than "normal".
        gf2 = conv_psf(x, k)
        @test ≈(gf,gf2, rtol=1e-2) # it is realatively inaccurate due to the kernel being generated in different places
    end
end
