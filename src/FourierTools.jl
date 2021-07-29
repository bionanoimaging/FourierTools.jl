module FourierTools

using PaddedViews, ShiftedArrays
using FFTW
FFTW.set_num_threads(4)
using LinearAlgebra
using IndexFunArrays
using LazyArrays
using ChainRulesCore

include("utils.jl")
include("resampling.jl")
include("custom_fourier_types.jl")
include("fourier_resizing.jl")
include("fourier_shifting.jl")
include("fourier_resample_1D_based.jl")
include("fourier_rotate.jl")
include("fourier_shear.jl")
include("fftshift_alternatives.jl")
include("fft_helpers.jl")
include("convolutions.jl")
include("damping.jl")
include("czt.jl")

end # module
