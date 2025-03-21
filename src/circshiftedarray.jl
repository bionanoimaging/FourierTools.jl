"""
    padded_tuple(v::AbstractVector, s)

Internal function used to compute shifts. Return a `Tuple` with as many element
as the dimensions of `v`. The first `length(s)` entries are filled with values
from `s`, the remaining entries are `0`. `s` should be an integer, in which case
`length(s) == 1`, or a container of integers with keys `1:length(s)`.

# Examples

```jldoctest padded_tuple
julia> FourierTools.padded_tuple(rand(10, 10), 3)
(3, 0)

julia> FourierTools.padded_tuple(rand(10, 10), (4,))
(4, 0)

julia> FourierTools.padded_tuple(rand(10, 10), (1, 5))
(1, 5)
```
"""
padded_tuple(v::AbstractArray, s) = ntuple(i -> i ≤ length(s) ? s[i] : 0, ndims(v))

# Computing a shifted index (subtracting the offset)
offset(offsets::NTuple{N,Int}, inds::NTuple{N,Int}) where {N} = map(-, inds, offsets)

"""
    CircShiftedArray(parent::AbstractArray, shifts)

Custom `AbstractArray` object to store an `AbstractArray` `parent` circularly shifted
by `shifts` steps (where `shifts` is a `Tuple` with one `shift` value per dimension of `parent`).
Use `copy` to collect the values of a `CircShiftedArray` into a normal `Array`.

!!! note
    `shift` is modified with a modulo operation and does not store the passed value
    but instead a nonnegative number which leads to an equivalent shift.

!!! note
    If `parent` is itself a `CircShiftedArray`, the constructor does not nest
    `CircShiftedArray` objects but rather combines the shifts additively.

# Examples

```jldoctest circshiftedarray
julia> v = [1, 3, 5, 4];

julia> s = CircShiftedArray(v, (1,))
4-element CircShiftedVector{Int64, Vector{Int64}}:
 4
 1
 3
 5

julia> copy(s)
4-element Vector{Int64}:
 4
 1
 3
 5
```
"""
struct CircShiftedArray{T, N, S<:AbstractArray} <: AbstractArray{T, N}
    parent::S
    # the field `shifts` stores the circular shifts modulo the size of the parent array
    shifts::NTuple{N, Int}
    function CircShiftedArray(p::AbstractArray{T, N}, n = ()) where {T, N}
        myshifts = map(mod, padded_tuple(p, n), size(p))
        return new{T, N, typeof(p)}(p, myshifts)
    end
end

function CircShiftedArray(c::CircShiftedArray, n = ())
    myshifts = map(+, shifts(c), padded_tuple(c, n))
    return CircShiftedArray(parent(c), myshifts)
end

"""
    CircShiftedVector{T, S<:AbstractArray}

Shorthand for `CircShiftedArray{T, 1, S}`.
"""
const CircShiftedVector{T, S<:AbstractArray} = CircShiftedArray{T, 1, S}

CircShiftedVector(v::AbstractVector, n = ()) = CircShiftedArray(v, n)

Base.size(s::CircShiftedArray) = size(parent(s))
Base.axes(s::CircShiftedArray) = axes(parent(s))

@inline function bringwithin(ind_with_offset::Int, ranges::AbstractUnitRange)
    return ifelse(ind_with_offset < first(ranges), ind_with_offset + length(ranges), ind_with_offset)
end

@inline function Base.getindex(s::CircShiftedArray{T, N}, x::Vararg{Int, N}) where {T, N}
    @boundscheck checkbounds(s, x...)
    v, ind = parent(s), offset(shifts(s), x)
    i = map(bringwithin, ind, axes(s))
    return @inbounds v[i...]
end

@inline function Base.setindex!(s::CircShiftedArray{T, N}, el, x::Vararg{Int, N}) where {T, N}
    @boundscheck checkbounds(s, x...)
    v, ind = parent(s), offset(shifts(s), x)
    i = map(bringwithin, ind, axes(s))
    @inbounds v[i...] = el
    return s
end

Base.parent(s::CircShiftedArray) = s.parent

"""
    shifts(s::CircShiftedArray)

Return amount by which `s` is shifted compared to `parent(s)`.
"""
shifts(s::CircShiftedArray) = s.shifts


function copy(s::CircShiftedArray)
    res = similar(parent(s), eltype(s), size(s))
    res .= s
end

# function Base.copyto!(dst::AbstractArray, src::CircShiftedArray)
#     dst[:] .= @view src[:]
# end

# function Base.copyto!(dst::AbstractArray, Rdest::CartesianIndices, src::CircShiftedArray, Rsrc::CartesianIndices)
#     dst[Rdest...] .= @view src[Rsrc...]
# end

function collect(x::T)  where {T<:CircShiftedArray{<:Any,<:Any,<:CircShiftedArray}}
    x = CircShiftedArray(collect(parent(x)), shifts(x))
    return collect(x) # stay on the GPU
end
