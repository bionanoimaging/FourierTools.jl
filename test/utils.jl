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


    @testset "selectsizes" begin
        @test (1, 3, 2) == FourierTools.selectsizes(randn((4,3,2)), (2,3))
        @test (3, 2) == FourierTools.selectsizes(randn((4,3,2)), (2,3), keep_dims=false)
        @test (1, ) == FourierTools.selectsizes(randn((1,)), (1,), keep_dims=false)

    end


    @testset "slice" begin
    
        x = randn((1,2,3,4))
        y = FourierTools.slice(x, 2, 2)
        @test x[:, 2:2, :, :] == y

        x = randn((5,2,3,4))
        y = FourierTools.slice(x, 1, 4)
        @test x[4:4, :, :, :] == y

        x = randn((5))
        y = FourierTools.slice(x, 1, 5)
        @test x[5:5] == y

    end
    
    
    @testset "slice indices" begin
        x = randn((1,2,3))
        y = FourierTools.slice_indices(axes(x), 1, 1)
        @test y == (1:1, 1:2, 1:3)
    
    
        x = randn((20,4,20, 1, 2))
        y = FourierTools.slice_indices(axes(x), 2, 3)
        @test y == (1:20, 3:3, 1:20, 1:1, 1:2)
    end


    @testset "test expanddims" begin
        function f(s, N)
            @test FourierTools.expanddims(randn(s), N + length(s))|> size == (s..., ones(Int,N)...)
        end
        f((1,2,3), 2)
        f((1,2,3,4,5), 8)
        f((1), 5)
    end


    @testset "Test rfft_size" begin
        s = (11, 20, 10)
        @test FourierTools.rfft_size(s, 2) == size(rfft(randn(s),2))
        
        s = (11, 21, 10)
        @test FourierTools.rfft_size(s, 2) == size(rfft(randn(s),2))
        
        s = (11, 21, 10)
        @test FourierTools.rfft_size(s, 1) == size(rfft(randn(s),(1,2,3)))
    end



    function center_test(x1, x2, x3, y1, y2, y3)
        arr1 = randn((x1, x2, x3))
        arr2 = zeros((y1, y2, y3))
    
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
        @test fftpos(1, 10) == -0.5:0.1:0.4

        @test collect(fftpos(0.01, 3)) ≈ [-0.005, 0.0, 0.005]

        @test fftpos(0.1, 20) ≈ -0.05:0.005:0.045

        @test fftpos(0.1, 21) == -0.05:0.005:0.05

        for N = 2:30
            @test length(fftpos(123, N)) == N
        end
    end
end
