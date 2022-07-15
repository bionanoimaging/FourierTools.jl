@testset "Fractional Fast Fourier Transform" begin 

    box1d = collect(box(Float32, (100,)))
    box1d_ = collect(box(Float32, (101,)))
    

    # consistency with fft
    @test abs.(ft(box1d)[30:70]) ./ sqrt(length(box1d)) ≈ abs.(frfft(box1d, 1.0, shift=true)[30:70])
    @test all(.≈(1 .+ abs.(ft(box1d)[30:70]) ./ sqrt(length(box1d)), 
              1 .+ abs.(frfft(frfft(box1d, 0.5, shift=true), 0.5, shift=true)[30:70]), rtol=5e-3))
    @test eltype(frfft(box1d, 1.0)) === ComplexF32
    
    @test all(.≈(1 .+ abs.(ft(box1d_)[30:70]) ./ sqrt(length(box1d_)), 1 .+ abs.(frfft(box1d_, 1.0, shift=true)[30:70]), rtol=5e-2))
    @test all(.≈(1 .+ abs.(ft(box1d_)[30:70]) ./ sqrt(length(box1d_)), 
              1 .+ abs.(frfft(frfft(box1d_, 0.5, shift=true), 0.5, shift=true)[30:70]), rtol=7e-3))


    for frac in [0, -1, 1, 2,-3, -4,4,-2, 1.1, 2.2, 3.3, 4.4, 5.5, -1.1, -2.2, -3.3, -4.4]
        @test all(.≈(10 .+ abs.(FractionalTransforms.frft(collect(box1d_), frac))[30:70], 
                 10 .+ abs.(frfft(box1d_, frac, shift=true))[30:70], rtol=9e-3))

        @test all(.≈(10 .+ real.(FractionalTransforms.frft(collect(box1d_), frac))[30:70], 
                     10 .+ real.(frfft(box1d_, frac, shift=true))[30:70], rtol=9e-3))

        @test all(.≈(10 .+ imag.(FractionalTransforms.frft(collect(box1d_), frac))[30:70], 
                 10 .+ imag.(frfft(box1d_, frac, shift=true))[30:70], rtol=9e-3))
    end
    # reversibility

    @test all(.≈(real(frfft(frfft(box1d, 0.5, shift=true), -0.5, shift=true))[30:70] , real(box1d)[30:70], rtol=1e-1))
    @test all(.≈(real(frfft(frfft(box1d_, 0.5, shift=true), -0.5, shift=true))[30:70] , real(box1d_)[30:70], rtol=1e-1))



    img = Float32.(testimage("resolution_test"))

    @test abs.(ft(img)) ./ sqrt(length(img)) .+ 10 ≈ 10 .+ abs.(frfft(img, 0.999999)) rtol=1e-4
    @test (real.(ft(img)) ./ sqrt(length(img)))[200:300] ≈ (real.(frfft(img, 0.999999)))[200:300] rtol=0.8

    
    x = randn((12,))
    x2 = randn((13,))
    @test frft(x, 0.5) ≈ frft(reshape(x, 12,1,1,1,1), 0.5)
    @test frft(x, 0.5) ≈ reshape(frft(reshape(x, 1,12,1,1), 0.5), 12)
end
