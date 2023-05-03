export shift, shift!


"""
    shift!(arr, shifts; kwargs...)

Shifts an array in-place. For real arrays it is based on `rfft`.
For complex arrays based on `fft`.
`shifts` can be non-integer, for integer shifts one should prefer
`circshift` or `ShiftedArrays.circshift` because a FFT-based methods
introduces numerical errors.

## kwargs...
* `fix_nyquist_frequency=false`: Fourier shifting of even-sized arrays is not revertible. However, if you did 
    `shift(x, δ)` you can it revert by `shift(x, δ, fix_nyquist_frequency=true)`. This only works if `δ` is the same.
* `take_real=true`: For even-sized arrays we take by default the `real` part of the exponential phase at the Nyquist frequency.
    This satisfies the property of real valuedness and the aliasing of the Nyquist term.

## Memory Usage
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
function shift!(arr::AbstractArray{<:Complex, N}, shifts; soft_fraction=0, fix_nyquist_frequency=false, take_real=true) where {N}
    return shift_by_1D_FT!(arr, shifts; soft_fraction=soft_fraction, fix_nyquist_frequency=fix_nyquist_frequency, take_real=take_real)
end

function shift!(arr::AbstractArray{<:Int, N}, shifts; kwargs...) where {N}
    throw(ArgumentError("FFTW.jl does not accept AbstractArrays{<:Int, N}. Convert array to a Real/Complex."))
end

function shift!(arr::AbstractArray{<:Real, N}, shifts; soft_fraction=0, fix_nyquist_frequency=false, take_real=true) where {N}
    return shift_by_1D_RFT!(arr, shifts; soft_fraction=soft_fraction, fix_nyquist_frequency=fix_nyquist_frequency, take_real=take_real)
end

"""
    shift(arr, shifts)

Returning a shifted array.
See [`shift!`](@ref shift!) for more details
"""
function shift(arr, shifts; soft_fraction=0, fix_nyquist_frequency=false, take_real=true)
    return shift!(copy(arr), shifts; soft_fraction=soft_fraction, fix_nyquist_frequency=fix_nyquist_frequency, take_real=take_real)
end

function soft_shift(freqs, shift, fraction=eltype(freqs)(0.1); corner=false)
    rounded_shift = round.(shift);
    if corner
        w = window_half_cos(size(freqs), border_in=2.0-2*fraction, border_out=2.0, offset=CtrCorner)
    else
        w = ifftshift_view(window_half_cos(size(freqs), border_in=1.0-fraction, border_out=1.0))
    end
    w = cond_instantiate(freqs, w)
    return cispi.(-freqs .* 2 .* (w .* shift + (1.0 .-w).* rounded_shift))
end

function shift_by_1D_FT!(arr::TA, shifts; soft_fraction=0, take_real=false, fix_nyquist_frequency=false) where {N, TA<:AbstractArray{<:Complex, N}}
    # iterates of the dimension d using the corresponding shift
    for (d, shift) in pairs(shifts)
        if iszero(shift)
            continue
        end
        # better use reorient from NDTools here?
        # TR = real_arr_type(TA)

        freqs = similar(arr, real(eltype(arr)), select_sizes(arr, d))
        # freqs = TR(reorient(fftfreq(size(arr, d)),d, Val(N)))
        freqs .= reorient(fftfreq(size(arr, d)),d, Val(N))
        # @show size(freqs)
        # allocates a 1D slice of exp values 
        if iszero(soft_fraction)
            ϕ = cispi.(- freqs .* 2 .* shift)
        else
            ϕ = soft_shift(freqs, shift, soft_fraction)
        end
        # ϕ = exp_ikx_sep(complex_arr_type(TA), size(arr), dims=(d,), shift_by = shift)[1]
        # in even case, set one value to real
        if iseven(size(arr, d))
            s = size(arr, d) ÷ 2 + 1
            CUDA.@allowscalar ϕ[s] = take_real ? real(ϕ[s]) : ϕ[s]
            CUDA.@allowscalar invr = 1 / ϕ[s]
            invr = isinf(invr) ? 0 : invr
            CUDA.@allowscalar ϕ[s] = fix_nyquist_frequency ? invr : ϕ[s]
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

    if !isempty(dims)
        ifft!(arr, dims)
    end
    
    return arr
end


# the idea is the following:
# rfft(x, 1) -> exp shift -> fft(x, 2) -> exp shift ->  fft(x, 3) -> exp shift -> ifft(x, [2,3]) -> irfft(x, 1)
# So once we did a rft to shift something we can call the routine for complex arrays to shift
function shift_by_1D_RFT!(arr::TA, shifts; soft_fraction=0, fix_nyquist_frequency=false, take_real=true) where {N, TA<:AbstractArray{<:Real, N}}
    for (d, shift) in pairs(shifts)
        if iszero(shift)
            continue
        end
        
        p = plan_rfft(arr, d)
        arr_ft = p * arr

        #s1 = select_sizes(arr_ft, d)
        s = size(arr_ft,d); # s1[d]
        # @show size(arr, d) ÷ 2 + 1
        # freqs = TR(reshape(fftfreq(size(arr, d))[1:s], ntuple(i -> 1, d-1)..., s1))

        # TR = real_arr_type(TA)
        freqs = similar(arr, real(eltype(arr_ft)), select_sizes(arr_ft, d))
        # freqs = TR(reorient(fftfreq(size(arr, d))[1:s], d, Val(N)))
        freqs .= reorient(rfftfreq(size(arr, d)), d, Val(N))
        if iszero(soft_fraction)
            ϕ = cispi.(-freqs .* 2 .* shift)
        else
            ϕ = soft_shift(freqs, shift, soft_fraction, corner=true)
        end
        if iseven(size(arr, d))
            # take real and maybe fix nyquist frequency
            CUDA.@allowscalar ϕ[s] = take_real ? real(ϕ[s]) : ϕ[s]
            CUDA.@allowscalar invr = 1 / ϕ[s]
            invr = isinf(invr) ? 0 : invr
            CUDA.@allowscalar ϕ[s] = fix_nyquist_frequency ? invr : ϕ[s]
        end
        arr_ft .*= ϕ
        # since we now did a single rfft dim, we can switch to the complex routine
        new_shifts = ntuple(i -> i ≤ d ? 0 : shifts[i], N)
         # workaround to mimic in-place rfft
        shift_by_1D_FT!(arr_ft, new_shifts; soft_fraction=soft_fraction, take_real=take_real, fix_nyquist_frequency=fix_nyquist_frequency)
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
