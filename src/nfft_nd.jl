export *, Mul!
export plan_nfft_nd
export nfft_nd

# PT <: Union{NFFTPlan, LinearAlgebra.Adjoint{CT, NFFT.NFFTPlan{Real, D, R}}
struct NFFTPlan_ND{CT, D, NT, PT}  <: Any where {CT, D, NT <: Union{Array{Bool,1}, Nothing}, PT}
    p :: PT 
    # destination size
    dsz::NTuple{D, Int}
    pad_value::CT
    # could be ::Array{Bool, D}, but is not specified to allow nothing as an entry
    pad_mask::NT
    is_adjoint::Bool
end

# Any_NFFTPlan_ND = Union{
#     NFFTPlan_ND, 
#     LinearAlgebra.Adjoint{CT, NFFTPlan_ND{CT, D, NT, PT}}  where {CT, D, NT, PT}
#     }

"""
    plan_nfft_nd(src, dst_coords; dst_size=nothing, is_in_pixels=false, is_local_shift=false, pad_value=nothing, reltol=1e-9)

Plans an n-dimensional non-uniform FFT on grids with a regular topology. In comparison to the `nfft()` routine, which this computed
is based on, this version does not require any reshape operations.

# Arguments
+ `src`: source array
+ `dst_coords`: array of destination coordinates. This can be either an `Array` of `Tuple` or an `Array with the last dimension of `size` `length(size(dst_coords))-1` 
  referring to the destination coordinates where the FFT needs to be computed. 
  Alternatively also a function mapping a tuple (of source index positions) to a tuple (of destination index positions).
  In the recommended mode, the indices are normalized to to Fouier frequency range (roughly speaking from -0.5:0.5).
+ `dst_size`: this argument is only used for functions. If you require a different result size for `dst_coords` being a function, state it here.By defaul (`dst_size=nothing`) the 
              destination size will be inferred form the argument `new_pos` or assumed to be `size(src)`.

+ `pixels_coords`: A `Boolean` flag indicating whether dst_coords refers to coordinates in pixels or (default) in relative frequency positions.
  If `is_in_pixels=true` is selected, destination coordinates (1-based indexing) as typical for array indexing is assumed and internally converted to relative positions.
+ `is_local_shift`: A `Boolean` controlling wether `dst_coords` refers to the destination coordinates or the relative distance from standard grid coordinates (size determind from `dst_coordinates`).  
+ `pad_value`: if supplied, values outside the valid pixel range (roughly -0.5:0.5) are replaced by this complex-valued pad value.
+  `is_adjoint`: if `true` this plan is based on the adjoint rather than the ordinary plan
+ `reltol`: The numerical precision to which the results are computed. This is passed to the `nfft` routine. Worse precision is faster.

```julia-repl
# Lets try a 2D rotation:
julia> using TestImages, NDTools, View5D, IndexFunArrays, FourierTools

julia> img = Float64.(testimage("resolution"));

# define a rotation operation
julia> rot_alpha(a, t) = (cosd(a)*t[1] - sind(a)*t[2], sind(a)*t[1]+cosd(a)*t[2])

julia> new_pos = rot_alpha.(10.0, idx(img, scale=ScaFT))

julia> f = ift(img)

julia> p = plan_nfft_nd(f, new_pos; is_local_shift=false, is_in_pixels=false)

julia> g = real.(p * f)

#display the result
julia> @ve img, g
```
"""
function plan_nfft_nd(src::AbstractArray{T,D}, dst_coords; dst_size=nothing, is_in_pixels=false, is_local_shift=false, pad_value=nothing, is_adjoint=false, reltol=1e-9) where {T,D}
    RT = real(T)
    CT = complex(T)

    dst_size = let
        if isnothing(dst_size)
            if isa(dst_coords, Function)
                size(src)
            else
                if eltype(dst_coords) <: Tuple
                    size(dst_coords)
                else
                    size(dst_coords)[1:end-1]
                end
            end
        else
            dst_size
        end
    end

    coord_sz = let 
        if is_adjoint
            # has to match the src size for the adjoint
            size(src)
        else
            # has to match the dst size for the adjoint
            dst_size
        end
    end
    

    dst_coords = let
        if isa(dst_coords, Function)
            # evaluate the function to get the numerical destination coordinate positions
            if is_in_pixels
                dst_coords.(idx(RT, coord_sz, offset=Tuple(zeros(Int, length(coord_sz)))))
            else
                dst_coords.(idx(RT, coord_sz, scale=ScaFT))
            end
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

    if is_adjoint && size(src) != dsz
        error("For the adjoint plan_nfft_nd, the source size $(size(src)) has to agree to the given coordinates referring to $(dsz).")
    end

    x = let
            if is_in_pixels
                if is_local_shift
                    x./ coord_sz
                else
                    (x .- one(RT)) ./ coord_sz .- RT(0.5)
                end
            else
                x
            end
        end

    x = let
        if is_local_shift
            xy = Tuple.(CartesianIndices(dsz))  
            ((reshape(reinterpret(reshape,eltype(xy[1]), xy), (length(dsz), prod(dsz))) .- one(RT)) ./ dsz .- RT(0.5)) .+ x
        else
            x
        end
    end
    # deal with the out-of-range positions

    maxfreq = RT.(floor.((dsz.-1)./2) ./ dsz)

    pad_mask, pad_value = let
        if isnothing(pad_value)
            nothing, zero(CT)
        else
            any((x .< -RT(0.5)) .|| (x .> maxfreq), dims=1)[:], zero(CT)
        end
    end

    x = clamp.(x, .- RT(0.5), maxfreq)
    src_sz = let 
        if is_adjoint
            dst_size
        else
            size(src)
        end
    end

    p = plan_nfft(x, src_sz; reltol=reltol)
    return NFFTPlan_ND((p), dst_size, pad_value, pad_mask, is_adjoint)
end

"""
    nfft_nd(src, dst_coords; is_in_pixels=false, is_local_shift=false)

performs an n-dimensional non-uniform FFT on grids with a regular topology. In comparison to the `nfft()` routine, which this computed
is based on, this version does not require any reshape operations.
See `plan_nfft_nd` for details on the arguments and usage examples.
Note that the input can be `Real` valued and will be automatically converted to `Complex`.

```julia-repl
# A Zoomed transform in 3D
julia> nfft_nd(rand(10,12,12), (t)-> (0.8*t[1], 0.7*t[2], 0.6*t[3]))
```
"""
function nfft_nd(src, dst_coords; dst_size=nothing, is_in_pixels=false, is_local_shift=false, pad_value=nothing, is_adjoint=false, reltol=1e-9)
    p = plan_nfft_nd(src, dst_coords; dst_size=dst_size, is_in_pixels=is_in_pixels, is_local_shift=is_local_shift, pad_value=pad_value, is_adjoint=is_adjoint, reltol=reltol)
    return p * src
end

# out of place multiplication to the fHat result. fHat can have ND-shape and will be reshaped as a view internally
function LinearAlgebra.mul!(fHat::StridedArray, p::NFFTPlan_ND, f::AbstractArray; verbose=false, timing::Union{Nothing,TimingStats} = nothing)
    # not that the reshape is just a different view, not copying the data
    if p.is_adjoint
        # rHat = reshape(fHat, size_in(p.p))
        pa = LinearAlgebra.adjoint(p.p)
        f = reshape(f, size_in(pa))
        # rHat = reshape(fHat, size_out(p.p)) # need to match to f in size
        mul!(fHat, pa, f; verbose=verbose, timing=timing)
    else
        rHat = reshape(fHat, size_out(p.p)) # need to match to f in size
        # f = reshape(f, size_in(p.p))
        mul!(rHat, p.p, f; verbose=verbose, timing=timing)
    end
    if !isnothing(p.pad_mask)
        rHat[p.pad_mask] .= p.pad_value
    end
    return fHat
end

# out of place multiplication to the fHat result. fHat can have ND-shape and will be reshaped as a view internally
function LinearAlgebra.mul!(fHat::AbstractArray{Tg}, p::NFFTPlan_ND, f::AbstractArray{T}) where {Tg, T}
    # not that the reshape is just a different view, not copying the data
    if p.is_adjoint
        # rHat = reshape(fHat, size_in(p.p))
        pa = LinearAlgebra.adjoint(p.p)
        f = reshape(f, size_in(pa))
        # rHat = reshape(fHat, size_out(p.p)) # need to match to f in size
        mul!(fHat, pa, f; verbose=verbose, timing=timing)
    else
        rHat = reshape(fHat, size_out(p.p)) # need to match to f in size
        # f = reshape(f, size_in(p.p))
        mul!(rHat, p.p, f; verbose=verbose, timing=timing)
    end
    if !isnothing(p.pad_mask)
        rHat[p.pad_mask] .= p.pad_value
    end
    return fHat
end

# function adjoint(p::NFFTPlan_ND{T, D, T3, P}) where {T, D, T3, P} 
#     return LinearAlgebra.Adjoint{T, typeof(p)}(p)
# end

# function Base.show(io::IO, p::LinearAlgebra.Adjoint{CT, NFFTPlan_ND{CT, D, NT, PT}}  where {CT, D, NT, PT})  
#     print(io, "Adjoint NFFTPlan_ND with ", p.parent.p.M, " sampling points for an input array of size", 
#            p.parent.p.N, " and an output array of size", p.parent.p.NOut, " with dims ", p.parent.p.dims)
# end

# function LinearAlgebra.mul!(g::StridedArray, p::LinearAlgebra.Adjoint{Complex{Tp},<:NFFTPlan_ND}, fHat::AbstractVector{T}) where {Tp,T}
#     g = reshape(g, size_in(p.p))
#     mul!(g, adjoint(p.p), fHat)
#     if !isnothing(p.pad_mask)
#         g[p.pad_mask] .= p.pad_value
#     end
#     return g
# end

function Base.:*(p::NFFTPlan_ND, f::AbstractArray{Complex{U},D}; kargs...) where {U,D}
    fHat = similar(f, eltype(f), p.dsz) # size_out(p.p)
    mul!(fHat, p, f; kargs...)
    return fHat
end

# for being called with the less stringent (real) datatype
function Base.:*(p::NFFTPlan_ND, f::AbstractArray{RT,D}; kargs...) where {RT <: Real, D}
        return p * complex.(f)
end

