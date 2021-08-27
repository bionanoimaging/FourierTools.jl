export shear, shear!

"""
    shear(arr, Δ, shear_dir_dim=1, shear_dim=2)

Shears an array by the amount of `Δ` pixels via an FFT approach. `Δ` is the relative shift between the
top and bottom row shifted with respect to each other.
`shear_dir_dim` decides the direction of the shear and `shear_dim` is the 
second dimension where the shear happens.
There is also `shear!` available.

For complex arrays we use `fft`, for real array we use `rfft`.
"""
function shear(arr::AbstractArray, Δ, shear_dir_dim=1, shear_dim=2)
    return shear!(copy(arr), Δ, shear_dir_dim, shear_dim)
end

"""
    shear!(arr, Δ, shear_dir_dim=1, shear_dim=2)

For more details see `shear.`

For complex arrays we can completely avoid large memory allocations.
For real arrays, we need at least allocate on array in the fourier space.
"""
function shear!(arr::AbstractArray{<:Complex, N}, Δ, shear_dir_dim=1, shear_dim=2) where N
    fft!(arr, shear_dir_dim)

    # stores the maximum amount of shift
    shift = reshape(fftfreq(size(arr, shear_dir_dim)), IndexFunArrays.selectsizes(arr, shear_dir_dim))
    
    apply_shift_strength!(arr, arr, shift, shear_dir_dim, shear_dim, Δ)

    # go back to real space
    ifft!(arr, shear_dir_dim)
    return arr
end

function shear!(arr::AbstractArray{<:Real, N}, Δ, shear_dir_dim=1, shear_dim=2) where N
    p = plan_rfft(arr, shear_dir_dim)
    arr_ft = p * arr 

    # stores the maximum amount of shift
    shift = reshape(rfftfreq(size(arr, shear_dir_dim)), IndexFunArrays.selectsizes(arr_ft, shear_dir_dim))
    
    apply_shift_strength!(arr_ft, arr, shift, shear_dir_dim, shear_dim, Δ)
    # go back to real space
 
    
    # overwrites arr in-place
    ldiv!(arr, p, arr_ft)
    return arr
end


function apply_shift_strength!(arr, arr_orig, shift, shear_dir_dim, shear_dim, Δ)
    #applies the strength to each slice
    shift_strength = reshape(fftpos(1, size(arr, shear_dim)), IndexFunArrays.selectsizes(arr, shear_dim))

    # do the exp multiplication in place
    e = cispi.(2 .* Δ .* shift .* shift_strength)
    # for even arrays we need to fix real property of highest frequency
    if iseven(size(e, shear_dir_dim))
        inds = slice_indices(axes(e), shear_dir_dim, fft_center(size(arr_orig, shear_dir_dim))) 
        e[inds...] .= real.(view(e, inds...))
    end
    arr .*= e 
    return arr
end
