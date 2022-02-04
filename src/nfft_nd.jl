export *, Mul!
export plan_nfft_nd
export nfft_nd

struct NFFTPlan_ND
    p::NFFTPlan
    # destination size
    dsz::NTuple
end

"""
    plan_nfft_nd(src, dst_coords; pixel_coords=false, is_deformation=false)

Plans an n-dimensional non-uniform FFT on grids with a regular topology. In comparison to the `nfft()` routine, which this computed
is based on, this version does not require any reshape operations.

# Arguments
+ `src`: source array
+ `dst_coords`: array of destination coordinates. This can be either an `Array` of `Tuple` or an `Array with the last dimension of `size` `length(size(dst_coords))-1` 
  referring to the destination coordinates where the FFT needs to be computed. 
  Alternatively also a function mapping a tuple (of source index positions) to a tuple (of destination index positions).
  In the recommended mode, the indices are normalized to to Fouier frequency range (roughly speaking from -0.5:0.5).
+ `pixels_coords`: A `Boolean` flag indicating whether dst_coords refers to coordinates in pixels or (default) in relative frequency positions.
  If `pixel_coords=True` is selected, destination coordinates (1-based indexing) as typical for array indexing is assumed and internally converted to relative positions.
  - is_deformation: A `Boolean` controlling wether `dst_coords` refers to the destination coordinates or the relative distance from standard grid coordinates (size determind from `dst_coordinates`).  

```julia-repl
# Lets try a 2D rotation:
julia> using TestImages, NDTools, View5D, IndexFunArrays, FourierTools

julia> img = Float64.(testimage("resolution"));

# define a rotation operation
julia> rot_alpha(a, t) = (cosd(a)*t[1] - sind(a)*t[2], sind(a)*t[1]+cosd(a)*t[2])

julia> new_pos = rot_alpha.(10.0, idx(img, scale=ScaFT))

julia> f = ift(img)

julia> p = plan_nfft_nd(f, new_pos; is_deformation=false, pixel_coords=false)

julia> g = real.(p * f)

#display the result
julia> @ve img, g
```
"""
function plan_nfft_nd(src, dst_coords; pixel_coords=false, is_deformation=false, reltol=1e-9)
    # convert 2d coordinates to matrix form
    src = let
        if eltype(src) <: Complex
            src
        else
            Complex.(src)
        end
    end

    x, dsz = let 
        if eltype(dst_coords) <: Tuple
            x = reshape(reinterpret(reshape,eltype(dst_coords[1]), dst_coords), (ndims(dst_coords), prod(size(dst_coords))))
            x, size(dst_coords)
        else
            x = reshape(dst_coords, (size(dst_coords)[end], prod(size(dst_coords)[1:end-1])))
            x, size(dst_coords)[1:end-1]
        end
    end
    x = let
            if pixel_coords
                if is_deformation
                    x./ dsz 
                else
                    (x.-1) ./ dsz .- 0.5
                end
            else
                x
            end
        end

    x = let
        if is_deformation
            xy = Tuple.(CartesianIndices(dsz))  
            ((reshape(reinterpret(reshape,eltype(xy[1]), xy), (length(dsz), prod(dsz))) .-1) ./ dsz .- 0.5) .+ x
        else
            x
        end
    end
    return NFFTPlan_ND(plan_nfft(x, dsz; reltol=reltol), dsz)
end

"""
    nfft_nd(src, dst_coords; pixel_coords=false, is_deformation=false)

performs an n-dimensional non-uniform FFT on grids with a regular topology. In comparison to the `nfft()` routine, which this computed
is based on, this version does not require any reshape operations.
See `plan_nfft_nd` for details on the arguments and usage examples.
"""
function nfft_nd(src, dst_coords; pixel_coords=false, is_deformation=false, reltol=1e-9)
    p = plan_nfft_nd(src, dst_coords; pixel_coords=pixel_coords, is_deformation=is_deformation, reltol=reltol)
    return p * src
end

function LinearAlgebra.mul!(fHat::StridedArray, p::NFFTPlan_ND, f::AbstractArray; verbose=false, timing::Union{Nothing,TimingStats} = nothing)
    mul!(fHat, p.p, f; verbose=verbose, timing=timing)
    return fHat
end

function LinearAlgebra.mul!(fHat::AbstractArray{Tg}, p::NFFTPlan_ND, f::AbstractArray{T}) where {Tg, T}
    mul!(fHat, p.p, f)
    return fHat
end

function Base.:*(p::NFFTPlan_ND, f::AbstractArray{Complex{U},D}; kargs...) where {U,D}
    fHat = similar(f, eltype(f), size_out(p.p))
    mul!(fHat, p, f; kargs...)
    fHat = reshape(fHat, p.dsz)
    return fHat
end
