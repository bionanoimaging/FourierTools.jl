@testset "test fft_helpers" begin

    @testset "Optional collect" begin
        y = [1,2,3]
        x = fftshift_view(y, (1))
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
        for dim = 1:4
            for _ in 1:3
                s = ntuple(_ -> rand(1:13), dim)
                arr = randn(ComplexF32, s)
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
        arr = randn((6,7,8))
        dims = [1,2]
        d = 6
        @test(ft2d(arr) == fftshift(fft(ifftshift(arr, (1,2)), (1,2)), dims))
        @test(ift2d(arr) == fftshift(ifft(ifftshift(arr, (1,2)), (1,2)), dims))
        @test(ffts2d(arr) == fftshift(fft(arr, (1,2)), (1,2)))
        @test(iffts2d(arr) == ifft(ifftshift(arr, (1,2)), (1,2)))
        @test(rffts2d(arr) == fftshift(rfft(arr, (1,2)), dims[2:2]))
        @test(rft2d(arr) == fftshift(rfft(ifftshift(arr, (1,2)), (1,2)), dims[2:2]))
        arr = randn(ComplexF32, (4,7,8))
        @test(irffts2d(arr, d) == irfft(ifftshift(arr, dims[2:2]), d, (1,2)))
        @test(irft2d(arr, d) == irft(arr, d, (1,2))) 
        

    end


    @testset "Test ft, ift, rft and irft real space centering" begin
        szs = ((10,10),(11,10),(100,101),(101,101))
        for sz in szs
            @test ft(ones(sz)) ≈ prod(sz) .* delta(sz)
            @test ft(delta(sz)) ≈ ones(sz)
            @test rft(ones(sz)) ≈ prod(sz) .* delta(rft_size(sz), offset=CtrRFT)
            @test rft(delta(sz)) ≈ ones(rft_size(sz))
            @test ift(ones(sz)) ≈ delta(sz)
            @test ift(delta(sz)) ≈ ones(sz) ./ prod(sz)
            @test irft(ones(rft_size(sz)),sz[1]) ≈ delta(sz)
            @test irft(delta(rft_size(sz),offset=CtrRFT),sz[1]) ≈ ones(sz) ./ prod(sz)
        end
    end
end
