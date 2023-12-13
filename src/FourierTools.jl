module FourierTools

using Reexport
using PaddedViews
# using CircShiftedArrays
using ShiftedArrays # optionally replaced by CircShiftedArrays
@reexport using FFTW
using LinearAlgebra
using IndexFunArrays
using ChainRulesCore
using NDTools
# to have the CuArray type accesible
using CUDA
@reexport using NFFT
FFTW.set_num_threads(4)


include("utils.jl")
# include("fix_cufft.jl")

include("nfft_nd.jl")
include("resampling.jl")
include("custom_fourier_types.jl")
include("fourier_resizing.jl")
include("fourier_shifting.jl")
include("fourier_filtering.jl")
include("fourier_resample_1D_based.jl")
include("fourier_rotate.jl")
include("fourier_shear.jl")
include("fftshift_alternatives.jl")
include("fft_helpers.jl")
include("convolutions.jl")
include("correlations.jl")
include("damping.jl")
include("czt.jl")
include("fractional_fourier_transform.jl")

end # module
