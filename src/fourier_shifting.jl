export shift, shift!


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
function shift!(arr::AbstractArray{<:Complex, N}, shifts; soft_fraction=0.0) where {N}
    return shift_by_1D_FT!(arr, shifts; soft_fraction=soft_fraction)
end

function shift!(arr::AbstractArray{<:Real, N}, shifts; soft_fraction=0.0) where {N}
    return shift_by_1D_RFT!(arr, shifts; soft_fraction=soft_fraction)
end

"""
    shift(arr, shifts)

Returning a shifted array.
See [`shift!`](@ref shift!) for more details
"""
function shift(arr, shifts; soft_fraction=0.0)
    return shift!(copy(arr), shifts; soft_fraction=soft_fraction)
end

function soft_shift(freqs, shift, fraction=0.1; corner=false)
    rounded_shift = round.(shift);
    if corner
        w = window_half_cos(size(freqs),border_in=2.0-2*fraction, border_out=2.0, offset=CtrCorner)
    else
        w = ifftshift_view(window_half_cos(size(freqs),border_in=1.0-fraction, border_out=1.0))
    end
    return exp.(-1im .* freqs .* 2pi .* (w .* shift + (1.0 .-w).* rounded_shift))
end

function shift_by_1D_FT!(arr::AbstractArray{<:Complex, N}, shifts; soft_fraction=0.0) where {N}
    for (d, shift) in pairs(shifts) # iterates of the dimension d using the corresponding shift
        if iszero(shift)
            continue
        end
        freqs = reshape(fftfreq(size(arr, d)), ntuple(i -> 1, Val(d-1))..., size(arr,d)) # better use reorient from NDTools here?
        # allocates a 1D slice of exp values 
        if soft_fraction == 0
            ϕ = exp.(-1im .* freqs .* 2pi .* shift) # use cispi ?
        else
            ϕ = soft_shift(freqs, shift, soft_fraction)
        end
        # in even case, set one value to real
        if iseven(size(arr, d))
            s = size(arr, d) ÷ 2 + 1
            ϕ[s] = real(ϕ[s])
        end
        # go to fourier space and apply ϕ
        fft!(arr, d)
        arr .*= ϕ
    end

    # do one single multi dimensional fft to go back to real space
    # but only over those dimensions where shift != 0
    dims = Int[]
    for (i, s) in pairs(shifts)
        if !iszero(s)
            push!(dims, i)
        end
    end
    ifft!(arr, dims)

    return arr
end

# the idea is the following:
# rfft(x, 1) -> exp shift -> fft(x, 2) -> exp shift ->  fft(x, 3) -> exp shift -> ifft(x, [2,3]) -> irfft(x, 1)
# So once we did a rft to shift something we can call the routine for complex arrays to shift
function shift_by_1D_RFT!(arr::AbstractArray{<:Real, N}, shifts; soft_fraction=0.0) where {T, N}
    for (d, shift) in pairs(shifts)
        if iszero(shift)
            continue
        end
        
        s = size(arr, d) ÷ 2 + 1
        freqs = reshape(fftfreq(size(arr, d))[1:s], ntuple(i -> 1, d-1)..., s) 
        if soft_fraction == 0
            ϕ = exp.(-1im .* freqs .* 2pi .* shift)
        else
            ϕ = soft_shift(freqs, shift, soft_fraction, corner=true)
        end
        if iseven(size(arr, d))
            ϕ[s] = real(ϕ[s])
        end
        p = plan_rfft(arr, d)

        arr_ft = p * arr
        arr_ft .*= ϕ
        # since we now did a single rfft dim, we can switch to the complex routine
        new_shifts = ntuple(i -> i ≤ d ? 0 : shifts[i], N)
        shift_by_1D_FT!(arr_ft, new_shifts; soft_fraction=soft_fraction) # workaround to mimic in-place rfft
        # go back to real space now and return because shift_by_1D_FT processed
        # the other dimensions already
        mul!(arr, inv(p), arr_ft)
        return arr # this breaks the for loop and finishes the algorithm
    end
    return arr
end





 #"""
 #    shift_by_ND(arr, shift_vector)
 #
 #Shifts an array by the shift_vector using sinc-interpolation via FFTs
 #`shift_vector`.
 #The vector by which to shift
 #```
 #"""
 #function shift_by_ND(arr::AbstractArray{T, N}, shift_vector) where {T, N}
 #    # for complex arrays we need a full FFT
 #    if T <: Complex
 #        arr_out = shift_by_FFT(arr, Tuple(shift_vector))
 #    else 
 #        arr_out = shift_by_RFFT(arr, Tuple(shift_vector))
 #    end
 #    return arr_out
 #end
 #
 #
 #"""
 #    shift_by_RFFT(mat, new_size)
 #
 #Does shifting based on `rfft`. This function is called by `shift`.
 #"""
 #function shift_by_ND_RFT(mat, shift_vec) where {T}
 #    old_size=size(mat)
 #    irft(rft(mat).*exp.(
 #        dot.([shift_vec], idx(rft_size(mat),scale=ScaRFT,offset=CtrRFT)
 #        ).*(2im*pi)), size(mat,1))
 #end
 #
 #
 #function shift_by_ND_FT(mat, shift_vec) where {T}
 #    old_size=size(mat)
 #    ift(ft(mat).*exp.(
 #        dot.([shift_vec], idx(size(mat),scale=ScaFT,offset=CtrFT)
 #        ).*(2im*pi)))
 #end
 #
