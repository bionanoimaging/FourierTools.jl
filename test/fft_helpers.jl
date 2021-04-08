@testset "test fft_helpers" begin
    @testset "Optional collect" begin
        y = [1,2,3]
        x = fftshift_view(y, (1))
        @test fftshift(y) == FourierTools.optional_collect(x)
    end



end
