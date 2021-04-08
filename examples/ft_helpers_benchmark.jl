using FourierTools, BenchmarkTools

x = randn((133, 513, 33))

@btime ifft(fft($x));
@btime ifft(ifftshift(fftshift(fft($x))));
@btime iffts(ffts($x));
@btime ift(ft($x));

return
