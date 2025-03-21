@testset "test fft_helpers" begin

    @testset "Optional collect" begin
        y = opt_cu([1,2,3],use_cuda)
        x = fftshift_view(y, (1));
        @test fftshift(y) == FourierTools.optional_collect(x)
    end

    @testset "Test ft and ift wrappers" begin
        Random.seed!(42)
        testft(arr, dims) = @test(ft(arr, dims) ≈ fftshift(fft(ifftshift(arr, dims), dims), dims))
        testift(arr, dims) = @test(ift(arr, dims) ≈ fftshift(ifft(ifftshift(arr, dims), dims), dims))
        testffts(arr, dims) = @test(ffts(arr, dims) ≈ fftshift(fft(arr, dims), dims))
        testiffts(arr, dims) = @test(iffts(arr, dims) ≈ ifft(ifftshift(arr, dims), dims))
        testrft(arr, dims) = @test(rffts(arr, dims) ≈ fftshift(rfft(arr, dims), dims[2:end]))
        testirft(arr, dims, d) = @test(irffts(arr, d, dims) ≈ irfft(ifftshift(arr, dims[2:end]), d, dims))
        maxdim = ifelse(use_cuda, 3, 4)
        for dim = 1:maxdim
            for _ in 1:3
                s = ntuple(_ -> rand(1:13), dim)
                arr = opt_cu(randn(ComplexF32, s), use_cuda)
                dims = 1:dim
                testft(arr, dims)
                testift(arr, dims)
                dims = 1:rand(1:dim)
                testft(arr, dims)
                testift(arr, dims)
                testffts(arr, dims)
                testiffts(arr, dims)
            end
        end
    end

    @testset "Test 2d fft helpers" begin
        arr = opt_cu(randn((6,7,8)), use_cuda)
        dims = [1,2]
        d = 6
        @test(ft2d(arr) == fftshift(fft(ifftshift(arr, (1,2)), (1,2)), dims))
        @test(ift2d(arr) == fftshift(ifft(ifftshift(arr, (1,2)), (1,2)), dims))
        @test(ffts2d(arr) == fftshift(fft(arr, (1,2)), (1,2)))
        @test(iffts2d(arr) == ifft(ifftshift(arr, (1,2)), (1,2)))
        @test(rffts2d(arr) == fftshift(rfft(arr, (1,2)), dims[2:2]))
        @test(rft2d(arr) == fftshift(rfft(ifftshift(arr, (1,2)), (1,2)), dims[2:2]))
        @test(fft2d(arr) == fft(arr, dims))
        @test(ifft2d(arr) == ifft(arr, dims))
        @test(rfft2d(arr) == rfft(arr, (1,2)))
        @test(fftshift2d(arr) == fftshift(arr, (1,2)))
        @test(ifftshift2d(arr) == ifftshift(arr, (1,2)))
        @test(fftshift2d_view(arr) == fftshift_view(arr, (1,2)))
        @test(ifftshift2d_view(arr) == ifftshift_view(arr, (1,2)))

        arr = opt_cu(randn(ComplexF32, (4,7,8)), use_cuda)
        @test(irffts2d(arr, d) == irfft(ifftshift(arr, dims[2:2]), d, (1,2)))
        @test(irft2d(arr, d) == irft(arr, d, (1,2))) 
        @test(irfft2d(arr, d) == irfft(arr, d, (1,2))) 
    end

    @testset "Test ft, ift, rft and irft real space centering" begin
        atol = 1e-6
        szs = ((10,10),(11,10),(100,101),(101,101))
        for sz in szs
            my_ones = opt_cu(ones(sz), use_cuda)
            my_delta = opt_cu(collect(delta(sz)), use_cuda)
            @test isapprox(ft(my_ones), prod(sz) .* my_delta, atol=atol)
            @test isapprox(ft(my_delta), my_ones, atol=atol)
            @test isapprox(rft(my_ones), prod(sz) .* opt_cu(delta(rft_size(sz), offset=CtrRFT), use_cuda), atol=atol)
            @test isapprox(rft(my_delta), opt_cu(ones(rft_size(sz)), use_cuda), atol=atol)
            @test isapprox(ift(my_ones), my_delta, atol=atol)
            @test isapprox(ift(my_delta), my_ones ./ prod(sz), atol=atol)
            # needing to specify Complex datatype. Is a CUDA bug for irfft (!!!)
            @test isapprox(irft(opt_cu(ones(ComplexF64, rft_size(sz)), use_cuda), sz[1]), opt_cu(my_delta, use_cuda), atol=atol)
            @test isapprox(irft(opt_cu(collect(delta(ComplexF64, rft_size(sz), offset=CtrRFT)), use_cuda), sz[1]), opt_cu(my_ones ./ prod(sz), use_cuda), atol=atol)
        end
    end

    @testset "Test in place methods" begin
        atol = 1e-6
        x = opt_cu(randn(ComplexF32, (5,3,10)), use_cuda)
        dims = (1,2)
        @test isapprox(fftshift(fft(x, dims), dims), ffts!(copy(x), dims), atol=atol)
        @test isapprox(ffts2d!(copy(x)), ffts!(copy(x), (1,2)), atol=atol)
    end

end
