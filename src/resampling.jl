export resample, resample_by_FFT, resample_by_RFFT

"""
    resample(arr, new_size [, normalize])
Calculates the `sinc` interpolation of an `arr` on a new array size
`new_size`.
It is a re-evaluation of the Fourier series at new grid points.
`new_size` can be arbitrary. Each dimension is then independently either up or downsampled.
This method is based on FFTs and therefore implicitly assumes periodic
boundaries and a finite frequency support.
`normalize=true` by default multiplies by an appropriate factor so that 
the average intensity stays the same.
If `size(new_size)[i] > size(arr)[i]`, we apply zero padding in Fourier space.
If `size(new_size)[i] < size(arr)[i]`, we cut out a centered part of the
Fourier spectrum.
We apply some tricks at the boundary to increase accuracy of highest frequencies. 

 # Examples
```jldoctest
resample([1.0, 2.0, 3.0, 4.0], (8,) )
```
"""
function resample(arr::AbstractArray{T, N}, new_size, normalize=true) where {T, N}
    # for complex arrays we need a full FFT
    if T <: Complex
        arr_out = resample_by_FFT(arr, new_size)
    else 
        arr_out = resample_by_RFFT(arr, new_size)
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
    rf = rft(mat)
    irft(select_region_rft(rf,new_size), new_size[1])
end


"""
    resample_by_FFT(mat, new_size)

Does a resampling based on `fft`. This function is called by `resampling`.
"""
function resample_by_FFT(mat, new_size)
    old_size = size(mat)
    # for real arrays we apply an operation so that mat_fixed_before is hermitian
    mat_fixed_before = ft_fix_before(ft(mat),old_size,new_size)
    mat_pad = ft_pad(mat_fixed_before,new_size)
    # afterwards we add the highest pos. frequency to the highest lowest one 
    res_f = ft_fix_after(mat_pad, old_size,new_size)
    res = ift(res_f)
    return res    
end

