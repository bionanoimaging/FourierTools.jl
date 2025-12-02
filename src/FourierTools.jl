module FourierTools


using Reexport
# using PaddedViews
using MutableShiftedArrays # for circshift
@reexport using FFTW
using LinearAlgebra
using IndexFunArrays
using ChainRulesCore
using NDTools
import Base: checkbounds, getindex, setindex!, parent, size, axes, copy, collect

@reexport using NFFT
FFTW.set_num_threads(4)

include("utils.jl")
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
include("sdft_interface.jl")
include("sdft_implementations.jl")

end # module
