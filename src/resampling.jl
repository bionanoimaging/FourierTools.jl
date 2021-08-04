export resample, resample_by_FFT, resample_by_RFFT, upsample2_abs2, upsample2, upsample2_1D

"""
    resample(arr, new_size [, normalize])

Calculates the `sinc` interpolation of an `arr` on a new array size
`new_size`.
It is a re-evaluation of the Fourier series at new grid points.
`new_size` can be arbitrary. Each dimension is then independently either up or downsampled.

This method is based on FFTs and therefore implicitly assumes periodic
boundaries and a finite frequency support.

`normalize=true` by default multiplies by an appropriate factor so that
the array size is included in the scaling. This results in an array having roughly
the same mean intensity.


## Basic Principle
If `size(new_size)[i] > size(arr)[i]`, we apply zero padding in Fourier space.

If `size(new_size)[i] < size(arr)[i]`, we cut out a centered part of the
Fourier spectrum.

We apply some tricks at the boundary to increase accuracy of highest frequencies. 

For real arrays we use `rfft` based operations, for complex one we use `fft` based ones.


# Examples

sinc interpolation of 2 datapoints result in an approximation of cosine.

```jldoctest
julia> resample([2.0, 0.0], (6,))
6-element Vector{Float64}:
 2.0
 1.5
 0.5
 0.0
 0.5
 1.5

julia> resample([2.0, 0.0], (6,)) ≈ 1 .+ cos.(2π .* (0:5)./6)
true
```
"""
function resample(arr::AbstractArray{T, N}, new_size; normalize=true) where {T, N}
    if new_size == size(arr)
        return copy(arr)
    end
    # for complex arrays we need a full FFT
    if T <: Complex
        arr_out = resample_by_FFT(arr, Tuple(new_size))
    else 
        arr_out = resample_by_RFFT(arr, Tuple(new_size))
    end
    # normalize that values scale accordingly
    # this violates energy!
    if normalize
        arr_out .*= length(arr_out) ./ length(arr)
    end
    return arr_out
end


"""
    resample_by_RFFT(mat, new_size)

Does a resampling based on `rfft`. This function is called by `resampling`.
"""
function resample_by_RFFT(mat, new_size) where {T}
    old_size=size(mat)
    rf = rffts(mat)
    irffts(select_region_rft(rf,old_size,new_size), new_size[1])
end


"""
    resample_by_FFT(mat, new_size)

Does a resampling based on `fft`. This function is called by `resampling`.
"""
function resample_by_FFT(mat, new_size)
    old_size = size(mat)
    # for real arrays we apply an operation so that mat_fixed_before is hermitian
    mat_fixed_before = ft_fix_before(ffts(mat),old_size,new_size)
    mat_pad = ft_pad(mat_fixed_before,new_size)
    # afterwards we add the highest pos. frequency to the highest lowest one 
    res_f = ft_fix_after(mat_pad, old_size,new_size)
    res = iffts(res_f)
    return res    
end

function upsample2_1D(mat::AbstractArray{T, N}, dim=1, fix_center=false) where {T,N}
    newsize = Tuple((d==dim) ? 2*size(mat,d) : size(mat,d) for d in 1:N)
    res = zeros(eltype(mat), newsize)
    if fix_center && isodd(size(mat,dim))
        selectdim(res,dim,2:2:size(res,dim)) .= mat  
        shifts = Tuple((d==dim) ? 0.5 : 0.0 for d in 1:N)
        selectdim(res,dim,1:2:size(res,dim)) .= shift(mat, shifts) # this is highly optimized and all fft of zero-shift directions are automatically avoided
    else
        selectdim(res,dim,1:2:size(res,dim)) .= mat  
        shifts = Tuple((d==dim) ? -0.5 : 0.0 for d in 1:N)
        selectdim(res,dim,2:2:size(res,dim)) .= shift(mat, shifts) # this is highly optimized and all fft of zero-shift directions are automatically avoided
    end
    return res
end

"""
    upsample2(mat; dims=1:N)

Upsamples by a factor of two in all dimensions. 
The code is optimized for speed by using subpixelshifts rather than Fourier resizing.
"""
function upsample2(mat::AbstractArray{T, N}; dims=1:N, fix_center=false) where {T,N}
    res = mat
    for d in dims
        res = upsample2_1D(res,d, fix_center)
    end
    return res
end

"""
    upsample2_abs2(mat::AbstractArray{T, N}; dims=1:N)

Upsamples by a factor of two and applies the abs2 operation. The code is optimized for speed.
"""
function upsample2_abs2(mat::AbstractArray{T, N}; dims=1:N) where {T,N}
    return abs2.(upsample2(mat, dims=dims))
end
