module FourierTools


using PaddedViews, ShiftedArrays
using FFTW

export ft,ift, rft, irft
export extract, extract_ft, extract_rft

include("utils.jl")
include("resampling.jl")
include("custom_fourier_types.jl")
include("fourier_resizing.jl")


end # module
