export shift!



"""
    shift!(arr, shifts)

Shifts an array in-place. For real arrays it is based on `rfft`.
For complex arrays based on `fft`.
`shifts` can be non-integer, for integer shifts one should prefer
`circshift` or `ShiftedArrays.circshift` because a FFT-based methods
introduces numerical errors.

### Memory Usage
Note that for complex arrays we can avoid any large memory allocations
because of `fft!`. For `rfft` there does not exist a usable implementation
yet, so for real arrays there might be a temporary larger memory usage.

# Examples
```jldoctest
julia> x = [1.0 2.0 3.0; 4.0 5.0 6.0]
2×3 Matrix{Float64}:
 1.0  2.0  3.0
 4.0  5.0  6.0

julia> shift!(x, (1, 2))
2×3 Matrix{Float64}:
 5.0  6.0  4.0
 2.0  3.0  1.0

julia> x = [0, 1.0, 0.0, 1.0]
4-element Vector{Float64}:
 0.0
 1.0
 0.0
 1.0

julia> shift!(x, 0.5)
4-element Vector{Float64}:
 0.49999999999999994
 0.5
 0.49999999999999994
 0.5
```
"""
function shift!(arr::AbstractArray{<:Complex, N}, shifts) where {N}
    return shift_by_1D_FT!(arr, shifts)
end

function shift!(arr::AbstractArray{<:Real, N}, shifts) where {N}
    return shift_by_1D_RFT!(arr, shifts)
end

"""
    shift(arr, shifts)

Out of place shift.
See `shift!` for more details
"""
function shift(arr, shifts)
    return shift!(copy(arr), shifts)
end

function shift_by_1D_FT!(arr::AbstractArray{<:Complex, N}, shifts) where {N}
    for (d, shift) in pairs(shifts)
        freqs = reshape(fftfreq(size(arr, d)), ntuple(i -> 1, Val(d-1))..., size(arr,d)) 
        # allocates a 1D slice of exp values 
        ϕ = exp.(-1im .* freqs .* 2pi .* shift)
        # in even case, set one value to real
        if iseven(size(arr, d))
            s = size(arr, d) ÷ 2 + 1
            ϕ[s] = real(ϕ[s])
        end
        # go to fourier space and apply ϕ
        fft!(arr, d)
        arr .*= ϕ
        ifft!(arr, d)
    end

    return arr
end


function shift_by_1D_RFT!(arr::AbstractArray{<:Real, N}, shifts) where {T, N}
    for (d, shift) in pairs(shifts)
        s = size(arr, d) ÷ 2 + 1
        freqs = reshape(fftfreq(size(arr, d))[1:s], ntuple(i -> 1, d-1)..., s) 
        ϕ = exp.(-1im .* freqs .* 2pi .* shift)
        if iseven(size(arr, d))
            ϕ[s] = real(ϕ[s])
        end
        p = plan_rfft(arr, d)

        arr_ft = p * arr
        arr_ft .*= ϕ
        mul!(arr, inv(p), arr_ft)
    end

    return arr
end





"""
    shift_by_ND(arr, shift_vector)

Shifts an array by the shift_vector using sinc-interpolation via FFTs
`shift_vector`.
The vector by which to shift
```
"""
function shift_by_ND(arr::AbstractArray{T, N}, shift_vector) where {T, N}
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
function shift_by_ND_RFT(mat, shift_vec) where {T}
    old_size=size(mat)
    irft(rft(mat).*exp.(
        dot.([shift_vec], idx(rft_size(mat),scale=ScaRFT,offset=CtrRFT)
        ).*(2im*pi)), size(mat,1))
end


function shift_by_ND_FT(mat, shift_vec) where {T}
    old_size=size(mat)
    ift(ft(mat).*exp.(
        dot.([shift_vec], idx(size(mat),scale=ScaFT,offset=CtrFT)
        ).*(2im*pi)))
end

