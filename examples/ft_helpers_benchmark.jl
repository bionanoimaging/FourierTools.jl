using FourierTools, BenchmarkTools

function main()
    x = randn((133, 513, 33))
    y = copy(x)
    @btime $y .= real.(ifft(fft($x)));
    @btime $y .= real.(ifft(ifftshift(fftshift(fft($x)))));
    @btime $y .= real.(iffts(ffts($x)));
    @btime $y .= real.(ift(ft($x)));

    return
end

main()
