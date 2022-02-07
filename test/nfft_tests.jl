@testset "Test nfft_nd  methods" begin
    @testset "nfft_nd" begin
        sz = (6,8, 10)
        dat = rand(sz...)
        nft = fftshift(fft(ifftshift(dat)))
        @test isapprox(nfft_nd(dat, t->(0.0,0.0,0.0), pixel_coords=true, is_deformation=true), nft, rtol=1e-6)
        @test isapprox(nfft_nd(dat, t->(0.0,0.0,0.0), pixel_coords=false, is_deformation=true), nft, rtol=1e-6)
        @test isapprox(nfft_nd(dat, t->t, pad_value=nothing), nft, rtol=1e-6)
        p =plan_nfft_nd(dat, t->t, pad_value=0.0)
        @test isapprox(p*dat, nft, rtol=1e-6)
        @test isapprox(nfft_nd(dat, t->(10.0,10.0,10.0), pad_value=0.0), zeros(sz), rtol=1e-6)
        p = plan_nfft_nd(dat, t->t)
        @test isapprox(p*dat, nft, rtol=1e-6)
        res = zeros(complex(eltype(dat)), sz)
        LinearAlgebra.mul!(res, p, dat)
        @test isapprox(res, nft, rtol=1e-6)
    end
end
