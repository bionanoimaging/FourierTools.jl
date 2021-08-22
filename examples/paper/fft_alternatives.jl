using FourierTools, BenchmarkTools, FFTW, IndexFunArrays
FFTW.set_num_threads(6)

N = (2000, 3001)
arr = randn(ComplexF32, N)
mask = IndexFunArrays.rr(Float32, N) .< 50

fft_filter(arr, mask) = begin
    arr_ft = fftshift(fft(arr))
    arr_ft .*= mask
    ifft(ifftshift(arr_ft))
end

ffts_filter(arr, mask) = begin
    arr_ft = ffts(arr) 
    arr_ft .*= mask
    iffts(arr_ft)
end

@show ffts_filter(arr, mask) == fft_filter(arr, mask)

a = @benchmark $fft_filter($arr, $mask)
b = @benchmark $ffts_filter($arr, $mask)
println("FFT: ", minimum(a),"\n", "ffts: ", minimum(b))
