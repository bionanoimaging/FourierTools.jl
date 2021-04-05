export rotate, rotate!


"""
    rotate(arr, θ, rotation_place=(1,2))

Rotate an `arr` in the plane `rotation_plane` with an angle `θ` in degree
around the center pixel.

`rotate!` is also available.
"""
function rotate(arr, θ, rotation_plane=(1, 2))
    return rotate!(copy(arr), θ, rotation_plane=(1, 2)) 
end


"""
    rotate!(arr, θ, rotation_place=(1,2))

Rotate an `arr` in the plane `rotation_plane` with an angle `θ` in degree
around the center pixel.
"""
function rotate!(arr, θ, rotation_plane=(1, 2))
    # convert to radiants
    θ = deg2rad(θ)
    a,b = rotation_plane

    # parameters for shearing
    α = -tan(θ/2) * size(arr, a)
	β = sin(θ) * size(arr, b)

    # do the three step shearing
    shear!(arr, α, a, b)
    shear!(arr, β, b, a)
    shear!(arr, α, a, b)

    return arr
end
