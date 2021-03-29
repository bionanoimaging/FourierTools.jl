export shift
using IndexFunArrays

"""
    shift(arr, shift_vector)
Shifts an array by the shift_vector using sinc-interpolation via FFTs
`shift_vector`.
The vector by which to shift
```
"""
function shift(arr::AbstractArray{T, N}, shift_vector) where {T, N}
    # for complex arrays we need a full FFT
    if T <: Complex
        arr_out = shift_by_FFT(arr, Tuple(new_size))
    else 
        arr_out = shift_by_RFFT(arr, Tuple(new_size))
    end
    return arr_out
end


"""
    shift_by_RFFT(mat, new_size)

Does shifting based on `rfft`. This function is called by `shift`.
"""
function shift_by_RFFT(mat, shift_vec) where {T}
    old_size=size(mat)
    rf = rft(mat)
    shifted_rft = rf.*exp(idx(size(rf),ScaFT,offset=CtrFFT).*shift_vec)
    irft(, old_size[1])
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

