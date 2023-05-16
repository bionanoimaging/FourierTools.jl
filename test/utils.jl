@testset "Test util functions" begin

    @testset "Test fft center and rfft_center_diff" begin
        Random.seed!(42)
        @test 2 == FourierTools.fft_center(3)
        @test 3 == FourierTools.fft_center(4)
        @test 3 == FourierTools.fft_center(5)
        @test (2,3,4) == FourierTools.fft_center.((3,4,6))


        @test (0, 1, 2, 3) == FourierTools.ft_center_diff((12, 3, 5,6), (2,3,4))
        @test (6, 1, 2, 3) == FourierTools.ft_center_diff((12, 3, 5,6))


        @test (0, 0, 2, 3) == FourierTools.rft_center_diff((12, 3, 5,6), (2,3,4))
        @test (0, 0, 0, 3) == FourierTools.rft_center_diff((12, 3, 5,6), (3,4))
        @test (0, 0, 2, 3) == FourierTools.rft_center_diff((13, 3, 5,6), (1,3,4))
        @test (0, 1, 2, 3) == FourierTools.rft_center_diff((13, 3, 5,6))

    end



    @testset "Test rfft_size" begin
        s = (11, 20, 10)
        dat = opt_cu(randn(s), use_cuda);
        if !use_cuda
            @test FourierTools.rfft_size(s, 2) == size(rfft(dat,2))
            @test FourierTools.rft_size(randn(s), 2) == size(rfft(dat,2))
            s = (11, 21, 10)
            @test FourierTools.rfft_size(s, 2) == size(rfft(dat,2))
        end
        
        s = (11, 21, 10)
        dat = opt_cu(randn(s), use_cuda);
        @test FourierTools.rfft_size(s, 1) == size(rfft(dat,(1,2,3)))
    end



    function center_test(x1, x2, x3, y1, y2, y3)
        arr1 = opt_cu(randn((x1, x2, x3)), use_cuda);
        arr2 = opt_cu(zeros((y1, y2, y3)), use_cuda);
    
        FourierTools.center_set!(arr2, arr1)
        arr3 = FourierTools.center_extract(arr2, (x1, x2, x3))
        @test arr1 == arr3
    end
    
     # test center set and center extract methods
    @testset "center methods" begin
        center_test(4, 4, 4, 6,7,4)
        center_test(5, 4, 4, 7, 8, 4)
        center_test(5, 4, 4, 8, 8, 8)
        center_test(6, 4, 4, 7, 8, 8)
    
    
        @test 1 == FourierTools.center_pos(1)
        @test 2 == FourierTools.center_pos(2)
        @test 2 == FourierTools.center_pos(3)
        @test 3 == FourierTools.center_pos(4)
        @test 3 == FourierTools.center_pos(5)
        @test 513 == FourierTools.center_pos(1024)
    
        @test FourierTools.get_indices_around_center((5), (2)) == (2, 3)
        @test FourierTools.get_indices_around_center((5), (3)) == (2, 4)
        @test FourierTools.get_indices_around_center((4), (3)) == (2, 4)
        @test FourierTools.get_indices_around_center((4), (2)) == (2, 3)
    end


    @testset "Test fftpos" begin

        @test fftpos(1, 4, CenterFT) ≈  -0.5:0.25:0.25 
        @test fftpos(1, 4, CenterLast) ≈ -0.75:0.25:0.0 
        @test fftpos(1, 4, CenterMiddle) ≈ -0.375:0.25:0.375 
        @test fftpos(1, 4, CenterFirst) ≈ 0.0:0.25:0.75 
        @test fftpos(1, 4) ≈ 0.0:0.25:0.75 
        @test fftpos(1f0, 4, 2) ≈ -0.25f0:0.25f0:0.5f0


        function f(l, N)
            a = fftpos(l, N, CenterFT)
            b = fftpos(l, N, CenterFirst)
            c = fftpos(l, N, CenterLast)
            d = fftpos(l, N, CenterMiddle)
            @test (a[end] - a[begin] ≈ b[end] - b[begin] ≈ c[end] - c[begin] ≈ d[end] -d[begin])
        end

        f(1, 2)
        f(1, 3)
        f(42, 4)
        f(42, 5)
    end


    @testset "Test δ" begin 
        @test δ((3, 3)) == [0 0 0; 0 1 0; 0 0 0]
        @test δ((4, 3)) == [0 0 0; 0 0 0; 0 1 0; 0 0 0]
        @test δ(Float32, (4, 3)) == Float32[0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 0.0]
        @test δ(Float32, (4, 3)) |> eltype == Float32
        @test δ(Float32, (4, 3)) |> eltype == Float32
        @test δ(Float32, (4, 3)) == Float32[0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 0.0]
    end



    @testset "Pixel size conversion" begin
        @test fourierspace_pixelsize(1, 512) ≈ 1 / 512 
        @test all(fourierspace_pixelsize(1, (512,256)) .≈ 1 ./ (512, 256))
        @test realspace_pixelsize(1, 512) ≈ 1 / 512 
        @test all(realspace_pixelsize(1, (512,256)) .≈ 1 ./ (512, 256))
    end


    @testset "Check eltype error" begin
        @test_throws ArgumentError FourierTools.eltype_error(Float32, Float64)
        @test isnothing(FourierTools.eltype_error(Int, Int))
    end

    @testset "odd_view, fourier_reverse!" begin
        a = opt_cu([1 2 3;4 5 6;7 8 9;10 11 12], use_cuda)
        @test FourierTools.odd_view(a) == opt_cu([4 5 6;7 8 9; 10 11 12], use_cuda)
        fourier_reverse!(a)
        @test a == opt_cu([3 2 1;12 11 10;9 8 7;6 5 4], use_cuda)
        a = opt_cu([1 2 3;4 5 6;7 8 9;10 11 12], use_cuda)
        b = copy(a);
        fourier_reverse!(a,dims=1);
        @test a[2:end,:] == b[end:-1:2,:]
        a = opt_cu([1 2 3 4;5 6 7 8;9 10 11 12 ;13 14 15 16], use_cuda)
        b = copy(a);
        fourier_reverse!(a);
        # the ranges are used to avoid error in single element acces with CuArray
        @test a[2:2,2:2] == b[4:4,4:4]
        @test a[2:2,3:3] == b[4:4,3:3]
        fourier_reverse!(a);
        @test a == b
        fourier_reverse!(a;dims=1);
        @test a[2:end,:] == b[end:-1:2,:]
        rd = opt_cu(rand(5,6,7), use_cuda)
        if !use_cuda
            @test sum(abs.(imag.(ift(fourier_reverse!(ft(rd)))))) < 1e-10
        end
        @test sum(abs.(imag.(ift(fourier_reverse(ft(rd)))))) < 1e-10
        sz = (10,9,6)
        bb = opt_cu(box((sz)), use_cuda)
        if !use_cuda
            @test sum(abs.(real.(ift(fourier_reverse!(ft(bb)))) .- bb)) < 1e-10
        end
        @test sum(abs.(real.(ift(fourier_reverse(ft(bb)))) .- bb)) < 1e-10
    end
end
