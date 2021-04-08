@testset "fftshift alternatives" begin

    @testset "Test fftshift_view and ifftshift_view" begin
        Random.seed!(42)
        x = randn((2,3,4,5,6,7,8,9))
        dims = (4,6,7)
        @test fftshift(x,dims) == FourierTools.fftshift_view(x, dims)
        @test ifftshift(x,dims) == FourierTools.ifftshift_view(x, dims)
        @test x === ifftshift_view(fftshift_view(x))

        x = randn((33, 33, 34))
        @test fftshift(x) == FourierTools.fftshift_view(x)
        @test ifftshift(x) == FourierTools.ifftshift_view(x)

    end


end
