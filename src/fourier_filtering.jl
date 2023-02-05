export fourier_filter!, fourier_filter
export filter_gaussian, filter_gaussian!, filter_hann, filter_hann!, filter_hamming, filter_hamming!

"""
    fourier_filter!(arr::AbstractArray, fct=window_gaussian; kwargs...)
    
filters an array by multiplication in Fourierspace. This version uses in-place Fourier-transforms and multiplication whereever possible.
The filter window function is assumed to be separable. Depending on the array type either full-complex or real-to-complex FFTs will be used.
Note that some array types cannot directly be processed due to the type inference mechanism, in which case you should transform such types to
a standard array type (usually via calling "collect").
High-pass filters can be used by exchanging the border_in and border_out argument values.

#Arguments
+`arr`:     the array to filter
+`fct`:     the separable window function to use. The function has an ND-array size as the first parameter but wil be called with only one non-singleton dimension at a time.
+`kwargs...`:   keyword arguments which will be passed to fct. These typically are `border_in=0.8` and `border_out=1.0` for the window-functions as defined in the `IndexFunArrays.jl` toolbox.

#Example
```jdoctest
julia> arr = zeros(10,10); arr[3,4]=1.0;

julia> fourier_filter!(arr)
10×10 Matrix{Float64}:
 -0.00747645   0.00747645  -0.00747645  -0.07899   -0.00747645  …  -0.00747645   0.00747645  -0.00747645   0.00747645
  0.00747645  -0.00747645   0.00747645   0.07899    0.00747645      0.00747645  -0.00747645   0.00747645  -0.00747645
  0.07899     -0.07899      0.07899      0.834544   0.07899         0.07899     -0.07899      0.07899     -0.07899
  0.00747645  -0.00747645   0.00747645   0.07899    0.00747645      0.00747645  -0.00747645   0.00747645  -0.00747645
 -0.00747645   0.00747645  -0.00747645  -0.07899   -0.00747645     -0.00747645   0.00747645  -0.00747645   0.00747645
  0.00747645  -0.00747645   0.00747645   0.07899    0.00747645  …   0.00747645  -0.00747645   0.00747645  -0.00747645
 -0.00747645   0.00747645  -0.00747645  -0.07899   -0.00747645     -0.00747645   0.00747645  -0.00747645   0.00747645
  0.00747645  -0.00747645   0.00747645   0.07899    0.00747645      0.00747645  -0.00747645   0.00747645  -0.00747645
 -0.00747645   0.00747645  -0.00747645  -0.07899   -0.00747645     -0.00747645   0.00747645  -0.00747645   0.00747645
  0.00747645  -0.00747645   0.00747645   0.07899    0.00747645      0.00747645  -0.00747645   0.00747645  -0.00747645
```
"""
function fourier_filter!(arr::AbstractArray{<:Complex, N}, fct=window_gaussian; kwargs...) where {N}
    return fourier_filter_by_1D_FT!(arr, fct; kwargs...)
end

function fourier_filter!(arr::AbstractArray{<:Int, N}, fct=window_gaussian; kwargs...) where {N}
    throw(ArgumentError("FFTW.jl does not accept AbstractArrays{<:Int, N}. Convert array to a Real/Complex."))
end

function fourier_filter!(arr::AbstractArray{<:Real, N}, fct=window_gaussian; kwargs...) where {N}
    return fourier_filter_by_1D_RFT!(arr, fct;  kwargs...)
end

function fourier_filter!(arr::AbstractArray{<:Complex, N}, wins::AbstractVector; transform_win=true, kwargs...) where {N}
    return fourier_filter_by_1D_FT!(arr, wins; transform_win=transform_win, kwargs...)
end

function fourier_filter!(arr::AbstractArray{<:Real, N}, wins::AbstractVector; transform_win=true, kwargs...) where {N}
    return fourier_filter_by_1D_RFT!(arr, fct; transform_win=transform_win, kwargs...)
end

"""
    fourier_filter(arr::AbstractArray, fct=window_gaussian; kwargs...)
    
filters an array by multiplication in Fourierspace. This version uses in-place Fourier-transforms and multiplication whereever possible.
The filter window function is assumed to be separable. Depending on the array type either full-complex or real-to-complex FFTs will be used.
Note that some array types cannot directly be processed due to the type inference mechanism, in which case you should transform such types to
a standard array type (usually via calling "collect").
High-pass filters can be used by exchanging the border_in and border_out argument values.

#Arguments
+`arr`:     the array to filter
+`fct`:     the separable window function to use. The function has an ND-array size as the first parameter but wil be called with only one non-singleton dimension at a time.
+`kwargs...`:   keyword arguments which will be passed to fct. These typically are `border_in=0.8` and `border_out=1.0` for the window-functions as defined in the `IndexFunArrays.jl` toolbox.
#Example
```jdoctest
julia> using IndexFunArrays

julia> arr = zeros(10,10); arr[3,4]=1.0;

julia> fourier_filter(arr, window_hanning; border_in=0.5, border_out=1.0)
10×10 Matrix{Float64}:
 -0.00528825    0.0132184    -0.0217829    -0.0994384    …  -0.00528825    0.00102161    0.000100793   0.00102161
  0.00896055   -0.0223976     0.0369096     0.168491         0.00896055   -0.00173105   -0.000170787  -0.00173105
  0.0317295    -0.0793102     0.130698      0.59663          0.0317295    -0.00612969   -0.00060476   -0.00612969
  0.00896055   -0.0223976     0.0369096     0.168491         0.00896055   -0.00173105   -0.000170787  -0.00173105
 -0.00528825    0.0132184    -0.0217829    -0.0994384       -0.00528825    0.00102161    0.000100793   0.00102161
  0.00186562   -0.00466326    0.00768472    0.0350805    …   0.00186562   -0.000360412  -3.55585e-5   -0.000360412
 -1.38778e-18   1.11022e-17   5.55112e-18  -2.22045e-17      0.0           1.73472e-19  -4.29344e-18  -6.41848e-18
 -0.000499354   0.00124817   -0.0020569    -0.00938969      -0.000499354   9.64682e-5    9.51763e-6    9.64682e-5
 -5.55112e-18   0.0           1.11022e-17   8.88178e-17     -5.55112e-18  -1.21431e-17  -2.47198e-18  -2.08167e-18
  0.00186562   -0.00466326    0.00768472    0.0350805        0.00186562   -0.000360412  -3.55585e-5   -0.000360412```

"""
function fourier_filter(arr, fct=window_gaussian;  kwargs...)
    return fourier_filter!(copy(arr), fct; kwargs...)
end

function fourier_filter_by_1D_FT!(arr::TA, wins::AbstractVector; transform_win=false, dims=(1:ndims(arr))) where {N, TA <: AbstractArray{<:Complex, N}}
    if isempty(dims)
        return arr
    end
    # iterates of the dimension d using the corresponding shift
    for d in dims #(d, limit) in pairs(limits)
        # go to fourier space and apply window
        fft!(arr, d)
        arr .*= let 
            if transform_win
                fft(wins[d], d)
            else
                wins[d]
            end        
        end
    end
    ifft!(arr, dims)
    return arr
end

function fourier_filter_by_1D_FT!(arr::TA, fct=window_gaussian; dims=(1:ndims(arr)), transform_win=false, kwargs...) where {N, TA <: AbstractArray{<:Complex, N}}
    if isempty(dims)
        return arr
    end
    # iterates of the dimension d using the corresponding shift
    TR = real_arr_type(TA)
    wins = Vector{TR}(undef, N)
    # only calculate the necessary windows
    for d in dims 
        # these will possibly be transformed later
        win = similar(arr, real(eltype(arr)), select_sizes(arr, d))
        win .= fct(real(eltype(arr)), select_sizes(arr, d); kwargs...)
        wins[d] = ifftshift(win)
    end
    return fourier_filter_by_1D_FT!(arr, wins; transform_win=transform_win, dims=dims)
end

function fourier_filter_by_1D_RFT!(arr::TA, wins::AbstractVector; dims=(1:ndims(arr)), transform_win=false, kwargs...) where {T<:Real, N, TA<:AbstractArray{T, N}}
    if isempty(dims)
        return arr
    end
    d = dims[1] 
    p = plan_rfft(arr, d)

    arr_ft = p * arr
    arr_ft .*= let 
        if transform_win
            pw = plan_rfft(wins[d], d)
            pw * wins[d]
        else
            wins[d]
        end
    end
    # since we now did a single rfft dim, we can switch to the complex routine
    # new_limits = ntuple(i -> i ≤ d ? 0 : limits[i], N)
    # workaround to mimic in-place rfft
    fourier_filter_by_1D_FT!(arr_ft, wins; dims=dims[2:end], transform_win=transform_win, kwargs...)
    # go back to real space now and return because shift_by_1D_FT processed
    # the other dimensions already
    mul!(arr, inv(p), arr_ft)
    return arr
end

# transforms the first dim as rft and then hands over to the fft-based routines.
function fourier_filter_by_1D_RFT!(arr::TA, fct=window_gaussian; dims=(1:ndims(arr)), transform_win=false, kwargs...) where {T<:Real, N, TA<:AbstractArray{T, N}}
    if isempty(dims)
        return arr
    end
    # TR = real_arr_type(TA)
    d = dims[1]
    p = plan_rfft(arr, d)
    arr_ft = p * arr
    win = let
        if transform_win
            win = similar(arr, real(eltype(arr)), select_sizes(arr,d))
            win .= fct(real(eltype(arr)), select_sizes(arr,d); kwargs...)
            win = ifftshift(win) # for CuArray compatibility this has to be done sequentially. InPlace is not supported.
            pw = plan_rfft(win, d)
            pw*win
        else
            win = similar(arr, real(eltype(arr_ft)), select_sizes(arr_ft,d))
            # win = TR(fct(real(eltype(arr)), select_sizes(arr_ft,d), offset=CtrRFFT, scale=2 ./size(arr,d); kwargs...))
            win .= fct(real(eltype(arr)), select_sizes(arr_ft,d), offset=CtrRFFT, scale=2 ./size(arr,d); kwargs...)
        end
    end
    arr_ft .*= win
    fourier_filter_by_1D_FT!(arr_ft, fct;  dims=dims[2:end], transform_win=transform_win, kwargs...)
    # go back to real space now and return because shift_by_1D_FT processed
    # the other dimensions already
    mul!(arr, inv(p), arr_ft)
    return arr # this breaks the for loop and finishes the algorithm
end

"""
    filter_hann(arr; border_in=(real(eltype(arr)))(0.8), border_out=(real(eltype(arr)))(1), kwargs...)

performs Hann filtering by multiplying a Hann function in Fourier space.
Note that this filter is separable but not circularly symmetric.
See also `fourier_filter()`.

#Arguments
+`arr`:     the array to filter
+`sigma`:     the real-space standard deviation to filter with. From this the Fourier-space standard deviation will be calculated. 
+`kwargs...`:   additional arguments to be passed to `window_gaussian`, which is the underlying function from `IndexFunArray.jl`.
                Of particular importance are `border_in` and `border_out` defining the inner and outer border of the window relative to the Nyquist frequency.
#Example
```jdoctest
julia> res = filter_hann(FourierTools.delta((7,6)), border_in=0.3, border_out=0.4)
7×6 Matrix{Float64}:
  0.00954688  -0.00477344  -0.0334141  -0.0477344  -0.0334141  -0.00477344
 -0.00660664   0.00330332   0.0231233   0.0330332   0.0231233   0.00330332
 -0.0267498    0.0133749    0.0936242   0.133749    0.0936242   0.0133749
 -0.0357143    0.0178571    0.125       0.178571    0.125       0.0178571
 -0.0267498    0.0133749    0.0936242   0.133749    0.0936242   0.0133749
 -0.00660664   0.00330332   0.0231233   0.0330332   0.0231233   0.00330332
  0.00954688  -0.00477344  -0.0334141  -0.0477344  -0.0334141  -0.00477344
```
"""
function filter_hann(arr; border_in=(real(eltype(arr)))(0.8), border_out=(real(eltype(arr)))(1), kwargs...)
    filter_hann!(copy(arr); border_in=border_in, border_out=border_out, kwargs...)
end

"""
    filter_hann!(arr; border_in=(real(eltype(arr)))(0.8), border_out=(real(eltype(arr)))(1), kwargs...)

performs in-place Hann filtering by multiplying a Hann function in Fourier space.
Note that this filter is separable but not circularly symmetric.
See also `fourier_filter!()`.

#Arguments
+`arr`:     the array to replace by filtered version
+`sigma`:     the real-space standard deviation to filter with. From this the Fourier-space standard deviation will be calculated. 
+`kwargs...`:   additional arguments to be passed to `window_gaussian`, which is the underlying function from `IndexFunArray.jl`.
                Of particular importance are `border_in` and `border_out` defining the inner and outer border of the window relative to the Nyquist frequency.

See filter_hann() for an example.
"""
function filter_hann!(arr; border_in=(real(eltype(arr)))(0.8), border_out=(real(eltype(arr)))(1), kwargs...)
    return fourier_filter!(arr, window_hanning; border_in=border_in, border_out=border_out, kwargs...)
end

"""
    filter_hamming(arr; border_in=(real(eltype(arr)))(0.8), border_out=(real(eltype(arr)))(1), kwargs...)

performs Hamming filtering by multiplying a Hamming function in Fourier space.
Note that this filter is separable but not circularly symmetric.
See also `fourier_filter()`.

#Arguments
+`arr`:     the array to filter
+`sigma`:     the real-space standard deviation to filter with. From this the Fourier-space standard deviation will be calculated. 
+`kwargs...`:   additional arguments to be passed to `window_gaussian`, which is the underlying function from `IndexFunArray.jl`.
                Of particular importance are `border_in` and `border_out` defining the inner and outer border of the window relative to the Nyquist frequency.
#Example
```jdoctest
julia> res = filter_hamming(FourierTools.delta((7,6)), border_in=0.3, border_out=0.4)
7×6 Matrix{Float64}:
  0.00808048  -0.00404024  -0.0282817  -0.0488342  -0.0282817  -0.00404024
 -0.00559186   0.00279593   0.0195715   0.0337943   0.0195715   0.00279593
 -0.022641     0.0113205    0.0792435   0.13683     0.0792435   0.0113205
 -0.0363619    0.018181     0.127267    0.219752    0.127267    0.018181
 -0.022641     0.0113205    0.0792435   0.13683     0.0792435   0.0113205
 -0.00559186   0.00279593   0.0195715   0.0337943   0.0195715   0.00279593
  0.00808048  -0.00404024  -0.0282817  -0.0488342  -0.0282817  -0.00404024
```
"""
function filter_hamming(arr; border_in=(real(eltype(arr)))(0.8), border_out=(real(eltype(arr)))(1), kwargs...)
    filter_hamming!(copy(arr); border_in=border_in, border_out=border_out, kwargs...)
end

"""
    filter_hamming!(arr; border_in=(real(eltype(arr)))(0.8), border_out=(real(eltype(arr)))(1), kwargs...)

performs in-place Hamming filtering by multiplying a Hann function in Fourier space.
Note that this filter is separable but not circularly symmetric.
See also `fourier_filter!()`.

#Arguments
+`arr`:     the array to replace by filtered version
+`sigma`:     the real-space standard deviation to filter with. From this the Fourier-space standard deviation will be calculated. 
+`kwargs...`:   additional arguments to be passed to `window_gaussian`, which is the underlying function from `IndexFunArray.jl`.
                Of particular importance are `border_in` and `border_out` defining the inner and outer border of the window relative to the Nyquist frequency.

See filter_hamming() for an example.
"""
function filter_hamming!(arr; border_in=(real(eltype(arr)))(0.8), border_out=(real(eltype(arr)))(1), kwargs...)
    return fourier_filter!(arr, window_hamming; border_in=border_in, border_out=border_out, kwargs...)
end

"""
    filter_gaussian(arr, sigma=eltype(arr)(1); real_space_kernel=true, border_in=(real(eltype(arr)))(0), border_out=(real(eltype(arr))).(2 ./ (pi .* sigma)), kwargs...)

performs Gaussian filtering via Fourier filtering. Note that the argument `real_space_kernel` defines whether the Gaussian is computed in real or Fourier-space. Especially for small array sizes and small kernelsizes, the real-space version is preferred.
See also `filter_gaussian!()` and `fourier_filter()`.

#Arguments
+`arr`:     the array to filter
+`sigma`:     the real-space standard deviation to filter with. From this the Fourier-space standard deviation will be calculated. 
+ `real_space_kernel`: if `true`, the separable Gaussians are computed in real space and then Fourier-transformed. The overhead is relatively small, but the result does not create fringes.
+`kwargs...`:   additional arguments to be passed to `window_gaussian`, which is the underlying function from `IndexFunArray.jl`. This can be useful to create Fourier-shifted (Gabor-) filtering.
"""
function filter_gaussian(arr, sigma=eltype(arr)(1); real_space_kernel=true, border_in=(real(eltype(arr)))(0), border_out=(real(eltype(arr))).(2 ./ (pi .* sigma)), kwargs...)
    filter_gaussian!(copy(arr), sigma; real_space_kernel=real_space_kernel, border_in=border_in, border_out=border_out, kwargs...)
end

"""
    filter_gaussian!(arr, sigma=eltype(arr)(1); real_space_kernel=true, border_in=(real(eltype(arr)))(0), border_out=(real(eltype(arr))).(2 ./ (pi .* sigma)), kwargs...)

performs in-place Gaussian filtering by mulitplication in Fourier space.
Note that the argument `real_space_kernel` defines whether the Gaussian is computed in real or Fourier-space. Especially for small array sizes and small kernelsizes, the real-space version is preferred.
See also `filter_gaussian()` and `fourier_filter!()`.

#Arguments
+`arr`:     the array to replace by filtered version
+`sigma`:     the real-space standard deviation to filter with. From this the Fourier-space standard deviation will be calculated. 
+ `real_space_kernel`: if `true`, the separable Gaussians are computed in real space and then Fourier-transformed. The overhead is relatively small, but the result does not create fringes.
+`kwargs...`:   additional arguments to be passed to `window_gaussian`, which is the underlying function from `IndexFunArray.jl`. This can be useful to create Fourier-shifted (Gabor-) filtering.
"""
function filter_gaussian!(arr, sigma=eltype(arr)(1); real_space_kernel=true, border_in=(real(eltype(arr)))(0), border_out=(real(eltype(arr))).(2 ./ (pi .* sigma)), kwargs...)
    if real_space_kernel
        mysum = sum(arr)
        fourier_filter!(arr, gaussian; transform_win=true, sigma=sigma, kwargs...)
        arr .*= (mysum/sum(arr))
        return arr
    else
        return fourier_filter!(arr; border_in=border_in, border_out=border_out, kwargs...)
    end
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