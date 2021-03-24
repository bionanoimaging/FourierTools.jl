@testset "Test util functions" begin

    @testset "Test ft and ift wrappers" begin
        Random.seed!(42)
        testft(arr) = @test(ft(arr) ≈ fftshift(fft(arr)))
        testift(arr) = @test(ift(arr) ≈ ifft(ifftshift(arr)))
        # testrft(arr) = @test(rft(arr) ≈ fftshift(rft(arr)))
        # testirft(arr) = @test( irft(arr, size(arr)[1]) ≈ irfft(ifftshift(arr)), size(arr)[1])
        for dim = 1:4
            for _ in 1:4
                s = ntuple(_ -> rand(1:13), dim)
                arr = randn(ComplexF32, s)
                testft(arr)
                testift(arr)
                arr = randn(Float32, s)
                # testrft(arr)
                # testirft(arr)
            end
        end
    end


    @testset "Test fftshift_view and ifftshift_view" begin
        Random.seed!(42)
        x = randn((2,3,4,5,6,7,8,9))
        dims = (4,6,7)
        @test fftshift(x,dims) == FourierTools.fftshift_view(x, dims)
        @test ifftshift(x,dims) == FourierTools.ifftshift_view(x, dims)
        
        x = randn((33, 33, 34))
        @test fftshift(x) == FourierTools.fftshift_view(x)
        @test ifftshift(x) == FourierTools.ifftshift_view(x)

    end


    @testset "Test fft center and rfft_center0" begin
        Random.seed!(42)
        @test 2 == FourierTools.fft_center(3)
        @test 3 == FourierTools.fft_center(4)
        @test 3 == FourierTools.fft_center(5)
        @test (2,3,4) == FourierTools.fft_center.((3,4,6))
    end

end
