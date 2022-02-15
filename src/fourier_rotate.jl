export rotate, rotate!


"""
    rotate(arr, θ, rotation_plane=(1,2), adapt_size=true, keep_new_size=true)

Rotate an `arr` in the plane `rotation_plane` with an angle `θ` in degree
around the center pixel.

# Arguments:
+ `arr`: the array to rotate
+ `Θ`: the angle (in deg) to rotate by
+ `rotation_plane`: two dimensions selecting the 2D plane in which a multidimensional dataset is rotated 
+ `adapt_size`: if true (default), the three shears, which make up the rotation, will be allowed to enlarge the size of the array. This is slower but avoids wrap-around artefacts
+ `keep_new_size`: if true, the enlarged sizes (only for `adapt_size=true`) will also be returned. Otherwise the resulting data will be cut down to the original size
+ `pad_value`: specifies the value that areas outside the visible range (in the source) should be assigend to. A smart choice can reduce edge artefacts.

`rotate!` is also available.
"""
function rotate(arr, θ, rotation_plane=(1, 2); adapt_size=true, keep_new_size=false, pad_value=zero(eltype(arr)))
    if adapt_size
        θ = deg2rad(θ)
        a,b = rotation_plane
        old_size = size(arr)

        α = -tan(θ/2) * size(arr, b)
        β = sin(θ) * size(arr, a)
        extra_size = zeros(Int, ndims(arr))
        # to account for the two shears of α
        extra_size[a] = 2*ceil(Int,abs(α))
        # to account for the one shear of β
        extra_size[b] = ceil(Int,abs(β))
        arr = select_region(arr, new_size=old_size .+ Tuple(extra_size), pad_value=pad_value)
        # convert to radiants

        # parameters for shearing
        α = -tan(θ/2) * size(arr, b)
        β = sin(θ) * size(arr, a)

        # do the three step shearing
        arr = shear(arr, α, a, b, adapt_size=false)
        arr = shear(arr, β, b, a, adapt_size=false)
        arr = shear(arr, α, a, b, adapt_size=false)
        if keep_new_size || size(arr) == old_size
            return arr
        else
            return select_region(arr, new_size=old_size)
        end
    else
        return rotate!(copy(arr), θ, rotation_plane) 
    end
end


"""
    rotate!(arr, θ, rotation_plane=(1,2))

Rotate an `arr` in the plane `rotation_plane` with an angle `θ` in degree
around the center pixel.

# Arguments:
+ `arr`: the array to rotate
+ `Θ`: the angle (in deg) to rotate by
+ `rotation_plane`: two dimensions selecting the 2D plane in which a multidimensional dataset is rotated 
"""
function rotate!(arr, θ, rotation_plane=(1, 2))
    # convert to radiants
    θ = deg2rad(θ)
    a,b = rotation_plane

    # parameters for shearing
    α = -tan(θ/2) * size(arr, b)
	β = sin(θ) * size(arr, a)

    # do the three step shearing
    shear!(arr, α, a, b)
    shear!(arr, β, b, a)
    shear!(arr, α, a, b)

    return arr
end
