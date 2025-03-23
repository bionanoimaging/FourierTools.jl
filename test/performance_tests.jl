using BenchmarkTools
using CUDA
using Test

function test_fft(img)
    img = fft(img)
    img = ifft(img)
    return img
end

function test_ft(img)
    img = ft(img)
    img = ift(img)
    return img
end

use_cuda = false
sz = (1024, 1024)
dat = rand(ComplexF32, sz...)
img = opt_cu(dat, use_cuda)
display(@benchmark test_fft($img)) # 33 ms
display(@benchmark test_ft($img)) # 38 ms

use_cuda = true
img = opt_cu(dat, use_cuda)
display(@benchmark CUDA.@sync test_fft($img)) # 834 µs
display(@benchmark CUDA.@sync test_ft($img)) # 1086 ms

function test_nfft()
    J, N = 8, 16
    k = range(-0.4, stop=0.4, length=J)  # nodes at which the NFFT is evaluated
    f = cu(randn(ComplexF64, J))         # data to be transformed
    p = plan_nfft(k, N, reltol=1e-9)     # create plan
    fHat = adjoint(p) * f                # calculate adjoint NFFT
    y = p * fHat                         # calculate forward NFFT
end

# using CUDA, NFFT, CuNFFT
# Ny, Nx = 1024, 2048
# x = CUDA.randn(Ny, Nx);
# knots = CUDA.rand(2, Ny*Nx) .- 0.5f0;
# plan = NFFT.plan_nfft(CuArray{Float32}, knots, size(x));
# CUDA.@allowscalar [(adjoint(plan) * complex(x[:]))[1] for i=1:10]
