export shear, shear!, assign_shear_wrap!

"""
    shear(arr, Δ, shear_dir_dim=1, shear_dim=2; fix_nyquist=false, adapt_size=false::Bool, pad_value=zero(eltype(arr)))

Shears an array by the amount of `Δ` pixels via an FFT approach. `Δ` is the relative shift between the
top and bottom row shifted with respect to each other.
`shear_dir_dim` decides the direction of the shear and `shear_dim` is the 
second dimension where the shear happens.
There is also `shear!` available.

#Arguments
+ `arr`: array to shear 
+ `shear_dir_dim`: dimension of the shift during shear
+ `shear_dim`: dimension along which to progress and apply variing shears along `shear_dir_dim`
+ `fix_nyquist`: apply a fix to the highest frequency during the Fourier-space application of the exponential factor
+ `adapt_size`: if true, pad the data prior to the shear. The result array will be larger
+ `pad_value`: the value to pad with (only applies if `adapt_size=true`)
+ `assign_wrap=assign_wrap`: replaces wrap-around areas by `pad_value` (only of `adapt_size` is `false`)

For complex arrays we use `fft`, for real array we use `rfft`.
"""
function shear(arr::AbstractArray, Δ, shear_dir_dim=1, shear_dim=2; fix_nyquist=false, assign_wrap=false, adapt_size=false::Bool, pad_value=zero(eltype(arr)))
    if adapt_size
        ns = Tuple(d == shear_dir_dim ? size(arr,d)+ceil(Int,abs.(Δ)) : size(arr,d) for d in 1:ndims(arr))
        arr2 = collect(select_region(arr, new_size=ns, pad_value=pad_value))
        return shear!(arr2, Δ, shear_dir_dim, shear_dim, fix_nyquist=fix_nyquist)
    else
        return shear!(copy(arr), Δ, shear_dir_dim, shear_dim, fix_nyquist=fix_nyquist, assign_wrap=assign_wrap, pad_value=pad_value)
    end
end

"""
    shear!(arr, Δ, shear_dir_dim=1, shear_dim=2; fix_nyquist=false, assign_wrap=false, pad_value=zero(eltype(arr)))

For more details see `shear.`
# Extra Arguments 
`assign_wrap`: if `true` wrap-around areas are replaced by `pad_value`
`pad_value`:   the value to replace wrap-around areas with

For complex arrays we can completely avoid large memory allocations.
For real arrays, we need at least allocate on array in the fourier space.
"""
function shear!(arr::TA, Δ, shear_dir_dim=1, shear_dim=2; fix_nyquist=false, assign_wrap=false, pad_value=zero(eltype(arr))) where {N, TA<:AbstractArray{<:Complex, N}}
    fft!(arr, shear_dir_dim)

    # stores the maximum amount of shift
    # TR = real_arr_type(TA)
    shift = similar(arr, real(eltype(arr)), select_sizes(arr, shear_dir_dim))
    shift .= reshape(fftfreq(size(arr, shear_dir_dim)), NDTools.select_sizes(arr, shear_dir_dim))
    # shift = TR(reorient(fftfreq(size(arr, shear_dir_dim)), shear_dir_dim, Val(N)))
    
    apply_shift_strength!(arr, arr, shift, shear_dir_dim, shear_dim, Δ, fix_nyquist)

    # go back to real space
    ifft!(arr, shear_dir_dim)
    if assign_wrap
        assign_shear_wrap!(arr, Δ, shear_dir_dim=shear_dir_dim, shear_dim=shear_dim, pad_value=pad_value)
    end
    return arr
end

function shear!(arr::TA, Δ, shear_dir_dim=1, shear_dim=2; fix_nyquist=false, assign_wrap=false, pad_value=zero(eltype(arr))) where {N, TA<:AbstractArray{<:Real, N}}
    p = plan_rfft(arr, shear_dir_dim)
    arr_ft = p * arr 

    # stores the maximum amount of shift
    # TR = real_arr_type(TA)
    shift = similar(arr, real(eltype(arr_ft)), select_sizes(arr_ft, shear_dir_dim))
    shift .= reshape(rfftfreq(size(arr, shear_dir_dim)), NDTools.select_sizes(arr_ft, shear_dir_dim))
    # shift = TR(reorient(rfftfreq(size(arr, shear_dir_dim)),shear_dir_dim, Val(N)))
    
    apply_shift_strength!(arr_ft, arr, shift, shear_dir_dim, shear_dim, Δ, fix_nyquist)
    # go back to real space

    # overwrites arr in-place
    ldiv!(arr, p, arr_ft)
    if assign_wrap
        assign_shear_wrap!(arr, Δ, shear_dir_dim, shear_dim, pad_value)
    end
    return arr
end

"""
    assign_shear_wrap!(arr, Δ, shear_dir_dim=1, shear_dim=2, pad_value=zero(eltype(arr)))

Assign a `pad_value` to the places that may contain wrapped information using the `shear!` function.
Note that this only accounts for the geometrical wrap and not for possible fringes caused by sub-pixel effects.

# Arguments:
+ `arr`: the array to replace values
+ `Δ`: the amount of shear between both sides of the `arr`
+ `shear_dir_dim`: the dimension of the direction of the shear 
+ `shear_dim`: the shear dimension along which the amount of shift changes to create the shear.
+ `pad_value`: the value to replace the array values by.
"""
function assign_shear_wrap!(arr, Δ, shear_dir_dim=1, shear_dim=2, pad_value=zero(eltype(arr)))
    sd_size = size(arr, shear_dim)
    mid_sd = sd_size.÷2 .+1
    for sd = 1:size(arr, shear_dim)
        myshear = Δ*(sd .- mid_sd)/sd_size
        myabsshear = ceil(Int, abs(myshear))
        if myshear < 0
            selectdim(selectdim(arr, shear_dim, sd:sd), shear_dir_dim, 1:myabsshear) .= pad_value
        elseif myshear > 0
            sdd_size = size(arr, shear_dir_dim)
            from = min(sdd_size - myabsshear, sdd_size)
            selectdim(selectdim(arr, shear_dim, sd:sd), shear_dir_dim, from:sdd_size) .= pad_value
        end
    end
end


function apply_shift_strength!(arr::TA, arr_orig, shift, shear_dir_dim, shear_dim, Δ, fix_nyquist=false) where {T, N, TA<:AbstractArray{T, N}}
    #applies the strength to each slice
    # The TR trick does not seem to work for the code below due to a call with a PaddedArray.
    shift_strength = similar(arr, real(eltype(arr)), select_sizes(arr, shear_dim))
    shift_strength .= reorient(fftpos(1, size(arr, shear_dim), CenterFT), shear_dim, Val(N)) # (real(eltype(TA))).
 
    # do the exp multiplication in place
    e = cispi.(2 .* Δ .* shift .* shift_strength)
    # for even arrays we need to fix real property of highest frequency
    if iseven(size(arr_orig, shear_dir_dim))
        inds = NDTools.slice_indices(axes(e), shear_dir_dim, fft_center(size(arr_orig, shear_dir_dim))) 
        r = real.(view(e, inds...))
        if fix_nyquist
            inv_r = 1 ./ r
            e[inds...] .= map(x -> (isinf(x) ? zero(eltype(inv_r)) : x), inv_r)
        else
            e[inds...] .= r 
        end
    end
    arr .*= e 
    return arr
end
