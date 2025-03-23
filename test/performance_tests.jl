using BenchmarkTools
using CUDA
using Test

function test_fft()
    img = rand(ComplexF32, 512, 512)
    img = opt_cu(img, use_cuda)
    img = fft(img)
    img = ifft(img)
    return img
end

function test_ft()
    img = rand(ComplexF32, 512, 512)
    img = opt_cu(img, use_cuda)
    img = ft(img)
    img = ift(img)
    return img
end

diplay(@benchmark test_fft())
diplay(@benchmark test_ft())

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
