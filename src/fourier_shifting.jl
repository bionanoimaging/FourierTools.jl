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


function shift!(arr::AbstractArray{<:Real, N}, shifts) where {T, N}
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

        # in principle mul! could improve performance since in-place
        # however, does not work currently
        #inds = top_left_indices(rft_size(size(arr), d))
        #mul!(arr_ft2, p, arr)
        #arr_ft2 .= p * arr
        #arr_ft2 .= rfft(arr, d)
        #arr_ft2 .*= ϕ
        #p_inv = plan_irfft(arr_ft2, size(arr, d), d,  flags=FFTW.UNALIGNED)
        
        #arr_ft2 = @view arr_ft[inds...]
        #mul!(arr, inv(p), arr_ft2)
        #arr .= irfft(arr_ft2, size(arr, d), d)
    end

    return arr
end
