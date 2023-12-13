@testset "fftshift alternatives" begin
    @testset "Test fftshift_view and ifftshift_view" begin
        Random.seed!(42)
        x = opt_cu(randn((2,1,4,1,6,7,4,7)), use_cuda);
        dims = (4,6,7)
        @test fftshift(x,dims) == FourierTools.fftshift_view(x, dims)
        @test ifftshift(x,dims) == FourierTools.ifftshift_view(x, dims)
        @test x === FourierTools.optional_collect(ifftshift_view(fftshift_view(x)))
        @test x === FourierTools.optional_collect(fftshift_view(ifftshift_view(x)))
        @test x === FourierTools.optional_collect(ifftshift_view(fftshift_view(x, dims), dims))
        @test x === FourierTools.optional_collect(fftshift_view(ifftshift_view(x, dims), dims))

        x = opt_cu(randn((13, 13, 14)), use_cuda);
        @test fftshift(x) == FourierTools.fftshift_view(x)
        @test ifftshift(x) == FourierTools.ifftshift_view(x)
        @test fftshift(x, (2,3)) == FourierTools.fftshift_view(x, (2,3))
        @test ifftshift(x, (2,3) ) == FourierTools.ifftshift_view(x, (2,3))
    end
end


@testset "fftshift and ifftshift in-place" begin
    function f(arr, dims)
        arr = opt_cu(arr, use_cuda)
        arr3 = copy(arr)
        @test fftshift(arr, dims) == FourierTools._fftshift!(copy(arr), arr, dims)
        @test arr3 == arr
        @test ifftshift(arr, dims) == FourierTools._ifftshift!(copy(arr), arr, dims)
        @test arr3 == arr
        @test FourierTools._fftshift!(copy(arr), arr, dims) != arr
    end

    f(randn((8,)), 1)
    f(randn((2,)), 1)
    f(randn((3,)), 1)
    f(randn((3,4)), 1)
    f(randn((3,4)), 2)
    f(randn((4,4)), (1,2))
    f(randn((5,5)), (1, 2))
    f(randn((5,5)), (1,))
    f(randn((8, 7, 6,4,1)), (1,2))
    f(randn((8, 7, 6,4,1)), (2,3))
    f(randn((8, 7, 6,4,1)), 3)
    f(randn((8, 7, 6,4,1)), (1,2,3,4,5))
end
