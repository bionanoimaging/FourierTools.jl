export rotate, rotate!

"""
    rotate(arr, θ, rotation_plane=(1,2), adapt_size=true, keep_new_size=false)

Rotate an `arr` in the plane `rotation_plane` with an angle `θ` in degree around the center pixel. Note that, in contrast to `ImageTransformations.imrotate`, the rotation is done around the Fourier-center pixel (size()÷2+1) and not the geometric  mid point.

# Arguments:
+ `arr`: the array to rotate
+ `Θ`: the angle (in rad) to rotate by
+ `rotation_plane`: two dimensions selecting the 2D plane in which a multidimensional dataset is rotated 
+ `adapt_size`: if true (default), the three shears, which make up the rotation, will be allowed to enlarge the size of the array. This is slower but avoids wrap-around artefacts
    If false, the in-place version of `rotate` is used with all its problems. Only recommended for very small angles!
+ `keep_new_size`: if true, the enlarged sizes (only for `adapt_size=true`) will also be returned. Otherwise the resulting data will be cut down to the original size
+ `pad_value`: specifies the value that areas outside the visible range (in the source) should be assigend to. A smart choice can reduce edge artefacts.

`rotate!` is also available.
"""
function rotate(arr, θ, rotation_plane=(1, 2); adapt_size=true, keep_new_size=false, pad_value=zero(eltype(arr)))
    θ = mod(θ + π, 2π) - π
    if adapt_size
        a,b = rotation_plane
        old_size = size(arr)

        # enforce an odd size along these dimensions, to simplify the potential flips below.
        arr = let
            if iseven(size(arr,a)) || iseven(size(arr,b))
                new_size = size(arr) .+ ntuple(i-> (i==a || i==b) ? iseven(size(arr,i)) : 0, ndims(arr))
                select_region(arr, new_size=new_size, pad_value=pad_value)
            else
                arr
            end
        end
        
        # exploit symmetries
        if abs(θ) > π/2  # 90 deg
            # reversing both coordinates is a rotation by 180 deg
            arr = FourierTools.reverse_view(arr, dims=rotation_plane)
            θ = θ - π
        end

        α = -tan(θ/2) * size(arr, b)
        β = sin(θ) * size(arr, a)

        extra_size = ntuple(ndims(arr)) do i
            if (i==a)
                # to account for the two shears of α
                2*ceil(Int,abs(α))
            elseif (i==b)
                # to account for the one shear of β
                ceil(Int,abs(β))
            else
                0
            end
        end
        arr = select_region(arr, new_size=old_size .+ extra_size, pad_value=pad_value)
        # convert to radiants

        # parameters for shearing
        α = -tan(θ/2) * size(arr, b)
        β = sin(θ) * size(arr, a)

        # do the three step shearing
        arr = shear(arr, α, a, b, adapt_size=false)
        shear!(arr, β, b, a)
        shear!(arr, α, a, b)
        if keep_new_size || size(arr) == old_size
            return arr
        else
            return select_region(arr, new_size=old_size, pad_value=pad_value)
        end
    else
        return rotate!(copy(arr), θ, rotation_plane) 
    end
end


"""
    rotate!(arr, θ, rotation_plane=(1,2))

In-place rotate an `arr` in the plane spanned by the two dimensions in the tuple `rotation_plane` with an angle `θ` in degree
around the center pixel. Note that, in contrast to `ImageTransformations.imrotate`, the rotation is done around the Fourier-center pixel (size()÷2+1) and not the geometric  mid point.
Note also that due to the operation being performed by successive cyclic shear operations in-place, pixels near the corner will be experiencing a massive wrap-around problem.
Use the out-of-place version `rotate` to avoid this.
Note also that this version generates very bad results with the angle approaching π. To fix this, use the out-of-place version of `rotate`.

# Arguments:
+ `arr`: the array to rotate
+ `Θ`: the angle (in rad) to rotate by
+ `rotation_plane`: two dimensions selecting the 2D plane in which a multidimensional dataset is rotated 
"""
function rotate!(arr, θ, rotation_plane=(1, 2); assign_wrap=false, pad_value=zero(eltype(arr)))
    θ = mod(θ + π, 2π) - π
    # convert to radiants
    a,b = rotation_plane

    # parameters for shearing
    α = -tan(θ/2) * size(arr, b)
	β = sin(θ) * size(arr, a)

    # do the three step shearing
    shear!(arr, α, a, b, assign_wrap=assign_wrap, pad_value=pad_value)
    shear!(arr, β, b, a, assign_wrap=assign_wrap, pad_value=pad_value)
    shear!(arr, α, a, b, assign_wrap=assign_wrap, pad_value=pad_value)

    return arr
end
