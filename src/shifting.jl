export shift, shift_by_FT, shift_by_RFT
using IndexFunArrays
using LinearAlgebra

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
        arr_out = shift_by_FFT(arr, Tuple(shift_vector))
    else 
        arr_out = shift_by_RFFT(arr, Tuple(shift_vector))
    end
    return arr_out
end


"""
    shift_by_RFFT(mat, new_size)

Does shifting based on `rfft`. This function is called by `shift`.
"""
function shift_by_RFT(mat, shift_vec) where {T}
    old_size=size(mat)
    irft(rft(mat).*exp.(
        dot.([shift_vec], idx(rft_size(mat),scale=ScaRFT,offset=CtrRFT)
        ).*(2im*pi)), size(mat,1))
end


"""
    resample_by_FFT(mat, new_size)

Does a resampling based on `fft`. This function is called by `resampling`.
"""
function shift_by_FT(mat, shift_vec) where {T}
    old_size=size(mat)
    ift(ft(mat).*exp.(
        dot.([shift_vec], idx(size(mat),scale=ScaFT,offset=CtrFT)
        ).*(2im*pi)))
end

