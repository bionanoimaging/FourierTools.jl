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
            for _ in 1:4
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
                
                if dim > 1
                    dims = 1:dim
                    arr = randn(Float32, s)
                    testrft(arr, dims)
                    testirft(rfft(arr, dims), dims, size(arr)[1])
                end
            end
        end
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
