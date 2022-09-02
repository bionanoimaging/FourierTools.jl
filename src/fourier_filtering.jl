export fourier_filter!, fourier_filter
export filter_gaussian # , filter_gaussian2

"""
"""
function fourier_filter!(arr::AbstractArray{<:Complex, N}, fct=window_gaussian; border_in=0.0, border_out=1.0) where {N}
    return fourier_filter_by_1D_FT!(arr, fct; border_in=border_in, border_out=border_out)
end

function fourier_filter!(arr::AbstractArray{<:Int, N}, fct=window_gaussian; kwargs...) where {N}
    throw(ArgumentError("FFTW.jl does not accept AbstractArrays{<:Int, N}. Convert array to a Real/Complex."))
end

function fourier_filter!(arr::AbstractArray{<:Real, N}, fct=window_gaussian; border_in=0.0, border_out=1.0) where {N}
    return fourier_filter_by_1D_RFT!(arr, fct;  border_in=border_in, border_out=border_out)
end

"""
    fourier_filter(arr, limits)

Returning a fourier_filtered array.
See [`fourier_filter!`](@ref fourier_filter!) for more details
"""
function fourier_filter(arr, fct=window_gaussian;  border_in=0.0, border_out=1.0)
    return fourier_filter!(copy(arr), fct; border_in=border_in, border_out=border_out)
end

function fourier_filter_by_1D_FT!(arr::TA, fct=window_gaussian; border_in=0.0, border_out=1.0, dims=(1:ndims(arr))) where {N, TA <: AbstractArray{<:Complex, N}}
    # iterates of the dimension d using the corresponding shift
    sz = size(arr)
    for d in dims #(d, limit) in pairs(limits)
        w = TA(ifftshift(fct(real(eltype(arr)), select_sizes(arr,d), border_in=border_in, border_out=border_out)))
        #w = TA(collect(ifftshift(fct(real(eltype(arr)), select_sizes(arr,d), border_in=border_in, border_out=border_out))))
        # go to fourier space and apply w
        fft!(arr, d)
        arr .*= w
    end

    ifft!(arr, dims)
    return arr
end

# the idea is the following:
# rfft(x, 1) -> exp shift -> fft(x, 2) -> exp shift ->  fft(x, 3) -> exp shift -> ifft(x, [2,3]) -> irfft(x, 1)
# So once we did a rft to shift something we can call the routine for complex arrays to shift
function fourier_filter_by_1D_RFT!(arr::TA, fct=window_gaussian;  border_in=0.0, border_out=1.0) where {T<:Real, N, TA<:AbstractArray{T, N}}
    sz = size(arr)
    for d = 1:length(sz) #(d, limit) in pairs(limits)
        p = plan_rfft(arr, d)

        arr_ft = p * arr
        w = TA(fct(real(eltype(arr)), select_sizes(arr_ft,d), offset=CtrRFFT, scale=2 ./size(arr,d), border_in=border_in, border_out=border_out))
        # w = TA(collect(fct(real(eltype(arr)), select_sizes(arr_ft,d), offset=CtrRFFT, scale=2 ./size(arr,d), border_in=border_in, border_out=border_out)))
        arr_ft .*= w
        # since we now did a single rfft dim, we can switch to the complex routine
        # new_limits = ntuple(i -> i ≤ d ? 0 : limits[i], N)
         # workaround to mimic in-place rfft
        fourier_filter_by_1D_FT!(arr_ft;  border_in=border_in, border_out=border_out, dims=(2:ndims(arr_ft)))
        # go back to real space now and return because shift_by_1D_FT processed
        # the other dimensions already
        mul!(arr, inv(p), arr_ft)
        return arr # this breaks the for loop and finishes the algorithm
    end
    return arr
end

function filter_gaussian(arr, sigma=eltype(arr)(1))
    sigma_k = 2 ./ (pi .* sigma) 
    fourier_filter(arr, border_out=sigma_k)
end

# Suggested code by Felix. But it is not as fast as the code above:
#
# function filter_gaussian2(img::AbstractArray{T};
#     sigma=one(T), dims=1:ndims(img)) where (T <: Real)
# shiftdims = dims[2:end]
# f = rfft(img, dims)
# return irfft(f .* ifftshift(gaussian(T, size(f), 
#                            offset=CtrRFT, 
#                            sigma=size(img) ./ (T(2π) .*sigma)),
#             shiftdims), 
# size(img, dims[1]), dims)
# end