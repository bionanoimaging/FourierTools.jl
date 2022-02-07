export *, Mul!
export plan_nfft_nd
export nfft_nd

struct NFFTPlan_ND{CT, D, NT}  <: Any where {CT, D, NT <: Union{Array{Bool,1}, Nothing}}
    p::NFFTPlan
    # destination size
    dsz::NTuple{D, Int}
    pad_value::CT
    # could be ::Array{Bool, D}, but is not specified to allow nothing as an entry
    pad_mask::NT
end

"""
    plan_nfft_nd(src, dst_coords; pixel_coords=false, is_deformation=false, pad_value=nothing, reltol=1e-9)

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
+ is_deformation: A `Boolean` controlling wether `dst_coords` refers to the destination coordinates or the relative distance from standard grid coordinates (size determind from `dst_coordinates`).  
+ `pad_value`: if supplied, values outside the valid pixel range (roughly -0.5:0.5) are replaced by this complex-valued pad value.
+ `reltol`: The numerical precision to which the results are computed. This is passed to the `nfft` routine. Worse precision is faster.

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
function plan_nfft_nd(src::AbstractArray{T,D}, dst_coords; pixel_coords=false, is_deformation=false, pad_value=nothing, reltol=1e-9) where {T,D}
    RT = real(T)
    CT = complex(T)

    dst_coords = let
        if isa(dst_coords, Function)
            # evaluate the function to get the numerical destination coordinate positions
            dst_coords.(idx(RT, size(src), scale=ScaFT))
        else
            dst_coords
        end
    end

    # convert ND coordinates to 2D matrix form

    x, dsz = let 
        if eltype(dst_coords) <: Tuple
            x = reshape(reinterpret(reshape,eltype(dst_coords[1]), dst_coords), (ndims(dst_coords), prod(size(dst_coords))))
            x, size(dst_coords)
        else
            sz = size(dst_coords)
            dst_coords = PermutedDimsArray(dst_coords, (length(sz), (1:length(sz)-1)...))
            x = reshape(dst_coords, (sz[end], prod(sz[1:end-1])))
            x, sz[1:end-1]
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
    # deal with the out-of-range positions

    maxfreq = floor.((dsz.-1)./2) ./ dsz

    pad_mask, pad_value = let
        if isnothing(pad_value)
            nothing, zero(CT)
        else
            any((x .< -0.5) .|| (x .> maxfreq), dims=1)[:], zero(CT)
        end
    end

    x = clamp.(x, .-0.5, maxfreq)

    return NFFTPlan_ND(plan_nfft(x, dsz; reltol=reltol), dsz, pad_value, pad_mask)
end

"""
    nfft_nd(src, dst_coords; pixel_coords=false, is_deformation=false)

performs an n-dimensional non-uniform FFT on grids with a regular topology. In comparison to the `nfft()` routine, which this computed
is based on, this version does not require any reshape operations.
See `plan_nfft_nd` for details on the arguments and usage examples.
Note that the input can be `Real` valued and will be automatically converted to `Complex`.

```julia-repl
# A Zoomed transform in 3D
julia> nfft_nd(rand(10,12,12), (t)-> (0.8*t[1], 0.7*t[2], 0.6*t[3]))
```
"""
function nfft_nd(src, dst_coords; pixel_coords=false, is_deformation=false, pad_value=nothing, reltol=1e-9)
    p = plan_nfft_nd(src, dst_coords; pixel_coords=pixel_coords, is_deformation=is_deformation, pad_value=pad_value, reltol=reltol)
    return p * src
end

# out of place multiplication to the fHat result. fHat can have ND-shape and will be reshaped as a view internally
function LinearAlgebra.mul!(fHat::StridedArray, p::NFFTPlan_ND, f::AbstractArray; verbose=false, timing::Union{Nothing,TimingStats} = nothing)
    # not that the reshape is just a different view, not copying the data
    rHat = reshape(fHat, size_out(p.p)[1])
    mul!(rHat, p.p, f; verbose=verbose, timing=timing)
    if !isnothing(p.pad_mask)
        rHat[p.pad_mask] .= p.pad_value
    end
    return fHat
end

# out of place multiplication to the fHat result. fHat can have ND-shape and will be reshaped as a view internally
function LinearAlgebra.mul!(fHat::AbstractArray{Tg}, p::NFFTPlan_ND, f::AbstractArray{T}) where {Tg, T}
    # not that the reshape is just a different view, not copying the data
    rHat = reshape(fHat, size_out(p.p)[1])
    mul!(rHat, p.p, f)
    if !isnothing(p.pad_mask)
        rHat[p.pad_mask] .= p.pad_value
    end
    return fHat
end

function Base.:*(p::NFFTPlan_ND, f::AbstractArray{Complex{U},D}; kargs...) where {U,D}
    fHat = similar(f, eltype(f), p.dsz) # size_out(p.p)
    mul!(fHat, p, f; kargs...)
    return fHat
end

# for being called with the less stringent (real) datatype
function Base.:*(p::NFFTPlan_ND, f::AbstractArray{RT,D}; kargs...) where {RT <: Real, D}
        return p * complex.(f)
end