using FourierTools
using ImageTransformations
using IndexFunArrays
using Zygote
using NDTools
using LinearAlgebra # for the assigned nfft function LinearAlgebra.mul!
using FractionalTransforms
using TestImages
using CUDA
using Random, Test, FFTW

Random.seed!(42)

use_cuda = true
if use_cuda
    CUDA.allowscalar(false);
end
opt_cu(img, use_cuda) = ifelse(use_cuda, CuArray(img), img)

include("fft_helpers.jl")
include("fftshift_alternatives.jl")
include("utils.jl")
include("fourier_shifting.jl")
include("fourier_shear.jl")
include("fourier_rotate.jl")
include("resampling_tests.jl")
include("convolutions.jl")
include("correlations.jl")
include("custom_fourier_types.jl")
include("damping.jl")
include("czt.jl") #
include("nfft_tests.jl")
include("fractional_fourier_transform.jl")
include("fourier_filtering.jl")

return
