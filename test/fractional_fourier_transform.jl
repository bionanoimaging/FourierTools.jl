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


    @test all(.≈(1 .+ abs.(FractionalTransforms.frft(collect(box1d_), 0.8))[30:70], 
             1 .+ abs.(frfft(box1d_, 0.8, shift=true))[30:70], rtol=1e-1))
    # reversibility

    @test all(.≈(real(frfft(frfft(box1d, 0.5, shift=true), -0.5, shift=true))[30:70] , real(box1d)[30:70], rtol=1e-6))
    @test all(.≈(real(frfft(frfft(box1d_, 0.5, shift=true), -0.5, shift=true))[30:70] , real(box1d_)[30:70], rtol=1e-6))
end
