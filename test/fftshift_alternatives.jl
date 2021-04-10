@testset "fftshift alternatives" begin

    @testset "Test fftshift_view and ifftshift_view" begin
        Random.seed!(42)
        x = randn((2,1,4,1,6,7,4,7))
        dims = (4,6,7)
        @test fftshift(x,dims) == FourierTools.fftshift_view(x, dims)
        @test ifftshift(x,dims) == FourierTools.ifftshift_view(x, dims)
        @test x === FourierTools.optional_collect(ifftshift_view(fftshift_view(x)))
        @test x === FourierTools.optional_collect(fftshift_view(ifftshift_view(x)))
        @test x === FourierTools.optional_collect(ifftshift_view(fftshift_view(x, dims), dims))
        @test x === FourierTools.optional_collect(fftshift_view(ifftshift_view(x, dims), dims))

        x = randn((13, 13, 14))
        @test fftshift(x) == FourierTools.fftshift_view(x)
        @test ifftshift(x) == FourierTools.ifftshift_view(x)
        @test fftshift(x, (2,3)) == FourierTools.fftshift_view(x, (2,3))
        @test ifftshift(x, (2,3) ) == FourierTools.ifftshift_view(x, (2,3))

    end
end
