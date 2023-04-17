export CircShiftedArray
using Base
# a = reshape(1:1000000,(1000,1000)) .+ 0
# a = reshape(1:(15*15),(15,15)) .+ 0
# c = CircShiftedArray(a,(3,3));
# b = copy(a)
# d = c .+ c;

"""
    CircShiftedArray{T, N, A<:AbstractArray{T,N}, myshift<:NTuple{N,Int}} <: AbstractArray{T,N}

is a type which lazily encampsulates a circular shifted array. If broadcasted with another `CircShiftedArray` it will stay to be a `CircShiftedArray` as long as the shifts are equal.
For unequal shifts, the `circshift` routine will be used. Note that the shift is encoded as an `NTuple{}` into the type definition.
"""
struct CircShiftedArray{T, N, A<:AbstractArray{T,N}, myshift<:Tuple} <: AbstractArray{T,N}
    parent::A    

    function CircShiftedArray(parent::A, myshift::NTuple{N,Int}) where {T,N,A<:AbstractArray{T,N}}
        ws = wrapshift(myshift, size(parent))
        new{T,N,A, Tuple{ws...}}(parent)
    end
    function CircShiftedArray(parent::CircShiftedArray{T,N,A,S}, myshift::NTuple{N,Int}) where {T,N,A,S}
        ws = wrapshift(myshift .+ to_tuple(csa_shift(typeof(parent))), size(parent))
        new{T,N,A, Tuple{ws...}}(parent)
    end
    # function CircShiftedArray(parent::CircShiftedArray{T,N,A,S}, myshift::NTuple{N,Int}) where {T,N,A,S==myshift}
    #     parent
    # end
end
# wraps shifts into the range 0...N-1
wrapshift(shift::NTuple, dims::NTuple) = ntuple(i -> mod(shift[i], dims[i]), length(dims))
# wraps indices into the range 1...N
wrapids(shift::NTuple, dims::NTuple) = ntuple(i -> mod1(shift[i], dims[i]), length(dims))
invert_rng(s, sz) = wrapshift(sz .- s, sz)

# define a new broadcast style
struct CircShiftedArrayStyle{N,S} <: Base.Broadcast.AbstractArrayStyle{N} end
csa_shift(::Type{CircShiftedArray{T,N,A,S}}) where {T,N,A,S} = S
to_tuple(S::Type{T}) where {T<:Tuple}= tuple(S.parameters...)
csa_shift(::CircShiftedArray{T,N,A,S}) where {T,N,A,S} = to_tuple(S)

# convenient constructor
CircShiftedArrayStyle{N,S}(::Val{M}, t::Tuple) where {N,S,M} = CircShiftedArrayStyle{max(N,M), Tuple{t...}}()
# make it known to the system
Base.Broadcast.BroadcastStyle(::Type{T}) where (T<: CircShiftedArray) = CircShiftedArrayStyle{ndims(T), csa_shift(T)}()
# make subarrays (views) of CircShiftedArray also broadcast inthe CircArray style:
Base.Broadcast.BroadcastStyle(::Type{SubArray{T,N,P,I,L}}) where {T,N,P<:CircShiftedArray,I,L} = CircShiftedArrayStyle{ndims(P), csa_shift(P)}()
# Base.Broadcast.BroadcastStyle(::Type{T}) where (T2,N,P,I,L, T <: SubArray{T2,N,P,I,L})= CircShiftedArrayStyle{ndims(P), csa_shift(p)}()
Base.Broadcast.BroadcastStyle(::CircShiftedArrayStyle{N,S}, ::Base.Broadcast.DefaultArrayStyle{M}) where {N,S,M} = CircShiftedArrayStyle{max(N,M),S}() #Broadcast.DefaultArrayStyle{CuArray}()
#Base.Broadcast.BroadcastStyle(::CircShiftedArrayStyle{0,S}, ::Base.Broadcast.DefaultArrayStyle{M}) where {S,M} = CircShiftedArrayStyle{M,S} #Broadcast.DefaultArrayStyle{CuArray}()

Base.size(csa::CircShiftedArray) = size(csa.parent)
Base.size(csa::CircShiftedArray, d::Int) = size(csa.parent, d)
Base.IndexStyle(::Type{<:CircShiftedArray}) = IndexLinear()

# linear indexing ignores the shifts
@inline Base.getindex(csa::CircShiftedArray{T,N,A,S}, i::Int) where {T,N,A,S} = getindex(csa.parent, i)
@inline Base.setindex!(csa::CircShiftedArray{T,N,A,S}, v, i::Int) where {T,N,A,S} = setindex!(csa.parent, v, i)

# ttest(csa::CircShiftedArray{T,N,A,S}) where {T,N,A,S} = println("$S,  $(to_tuple(S))")

# mod1 avoids first subtracting one and then adding one
@inline Base.getindex(csa::CircShiftedArray{T,N,A,S}, i::Vararg{Int,N}) where {T,N,A,S} = 
    getindex(csa.parent, (mod1(i[j]-to_tuple(S)[j], size(csa.parent, j)) for j in 1:N)...)

@inline Base.setindex!(csa::CircShiftedArray{T,N,A,S}, v, i::Vararg{Int,N}) where {T,N,A,S} = 
    (setindex!(csa.parent, v, (mod1(i[j]-to_tuple(S)[j], size(csa.parent, j)) for j in 1:N)...); v)

# if materialize is provided, a broadcasting expression would always collapse to the base type.
# Base.Broadcast.materialize(csa::CircShiftedArray{T,N,A,S}) where {T,N,A,S} = circshift(csa.parent, to_tuple(S))

# These apply for broadcasted assignment operations.
Base.Broadcast.materialize!(dest::CircShiftedArray{T,N,A,S}, csa::CircShiftedArray{T2,N2,A2,S}) where {T,N,A,S,T2,N2,A2} = Base.Broadcast.materialize!(dest.parent, csa.parent)

# function Base.Broadcast.materialize(bc::Base.Broadcast.Broadcasted{CircShiftedArrayStyle{N,S}}) where {T,N,A,S}
#     similar(...size(bz)
#     invoke(Base.Broadcast.materialize!, Tuple{CircShiftedArray{T,N,A,S}, Base.Broadcast.Broadcasted}, dest, bc)
# end

# remove all the circ-shift part if all shifts are the same
function Base.Broadcast.materialize!(dest::CircShiftedArray{T,N,A,S}, bc::Base.Broadcast.Broadcasted{CircShiftedArrayStyle{N,S}}) where {T,N,A,S}
    invoke(Base.Broadcast.materialize!, Tuple{CircShiftedArray{T,N,A,S}, Base.Broadcast.Broadcasted}, dest, bc)
end
# we cannot specialize the Broadcast style here, since the rhs may not contain a CircShiftedArray and still wants to be assigned
function Base.Broadcast.materialize!(dest::CircShiftedArray{T,N,A,S}, bc::Base.Broadcast.Broadcasted) where {T,N,A,S}
    @show "materialize! cs"
    @show typeof(bc)
    @show only_shifted(bc)
    if only_shifted(bc)
        # bcn = Base.Broadcast.Broadcasted{CircShiftedArrayStyle{N,S}}(bc.f, bc.args, bc.axes)
        # fall back to standard assignment
        @show "use raw"
        # to avoid calling the method defined below, we need to use `invoke`:
        invoke(Base.Broadcast.materialize!, Tuple{AbstractArray, Base.Broadcast.Broadcasted}, dest, bc) 
    else
        # get all not-shifted arrays and apply the materialize operations piecewise using array views
        materialize_checkerboard!(dest.parent, bc, Tuple(1:N), wrapshift(size(dest) .- csa_shift(dest), size(dest)), true)
        # materialize_checkerboard!(dest.parent, bc, Tuple(1:N), csa_shift(dest), true)
    end
    return dest
end

# NOT WORKING !
function Base.Broadcast.materialize!(dest::AbstractArray, bc::Base.Broadcast.Broadcasted{CircShiftedArrayStyle{N,S}}) where {N,S}
    # function Base.Broadcast.materialize!(dest::CircShiftedArray{T,N,A,S}, bc::Base.Broadcast.Broadcasted{CircShiftedArrayStyle}) where {T,N,A,S}
    @show "materialize! cs into normal array "
    # @show to_tuple(S)
    # @show typeof(bc)
    materialize_checkerboard!(dest, bc, Tuple(1:N), wrapshift(size(dest) .- to_tuple(S), size(dest)), false)
    return dest
end

function generate_shift_ranges(dest, myshift)
    circshift_rng_1 = ntuple((d)->firstindex(dest,d):firstindex(dest,d)+myshift[d]-1, ndims(dest))
    circshift_rng_2 = ntuple((d)->firstindex(dest,d)+myshift[d]:lastindex(dest,d), ndims(dest))
    noshift_rng_1 = ntuple((d)->lastindex(dest,d)-myshift[d]+1:lastindex(dest,d), ndims(dest))
    noshift_rng_2 = ntuple((d)->firstindex(dest,d):lastindex(dest,d)-myshift[d], ndims(dest))
    return ((circshift_rng_1, circshift_rng_2), (noshift_rng_1, noshift_rng_2))
end

# function generate_shift_ranges(dest::SubArray{T,N,P,I,L}, myshift) where {T,N,P,I,L}
#     v.indices[d]
#     noshift_rng_1 = ntuple((d)->lastindex(dest,d)-myshift[d]+1:lastindex(dest,d), ndims(dest))
#     noshift_rng_2 = ntuple((d)->firstindex(dest,d):lastindex(dest,d)-myshift[d], ndims(dest))
#     return ((circshift_rng_1, circshift_rng_2), (noshift_rng_1, noshift_rng_2))
# end

"""
    materialize_checkerboard!(dest, bc, dims, myshift) 

this function calls itself recursively to subdivide the array into tiles, which each needs to be processed individually via calls to `materialize!`.

|--------|
| a| b   |
|--|-----|---|
| c| dD  | C |
|--+-----|---|
   | B   | A |
   |---------|

"""
function materialize_checkerboard!(dest, bc, dims, myshift, dest_is_cs_array=true) 

    dest = refine_view(dest)
    # gets Tuples of Tuples of 1D ranges (low and high) for each dimension
    cs_rngs, ns_rngs = generate_shift_ranges(dest, myshift)
    @show ns_rngs
    @show size(dest)

    for n in CartesianIndices(ntuple((x)->2, ndims(dest)))
        cs_rng = Tuple(cs_rngs[n[d]][d] for d=1:ndims(dest))
        ns_rng = Tuple(ns_rngs[n[d]][d] for d=1:ndims(dest))
        @show cs_rng
        @show ns_rng
        dst_rng = ifelse(dest_is_cs_array, cs_rng, ns_rng)
        dst_rng = refine_shift_rng(dest, dst_rng)
        dst_view = @view dest[dst_rng...]

        bc1 = split_array_broadcast(bc, ns_rng, cs_rng)
        @show typeof(dest)
        @show size(dst_view)
        @show size(bc1)
        if (prod(size(dst_view)) > 0)
            Base.Broadcast.materialize!(dst_view, bc1)
        end
    end
end

# some code which determines whether all arrays are shifted
only_shifted(bc::Number)  = true
only_shifted(bc::AbstractArray)  = false
only_shifted(bc::CircShiftedArray)  = true
only_shifted(bc::Base.Broadcast.Broadcasted) = all(only_shifted.(bc.args))

# These functions remove the CircShiftArray in a broadcast and replace each by a view into the original array 
split_array_broadcast(bc::Number, noshift_rng, shift_rng) = bc
split_array_broadcast(bc::AbstractArray, noshift_rng, shift_rng) = @view bc[noshift_rng...]
split_array_broadcast(bc::CircShiftedArray, noshift_rng, shift_rng)  = @view bc.parent[shift_rng...]
function split_array_broadcast(v::SubArray{T,N,P,I,L}, noshift_rng, shift_rng) where {T,N,P<:CircShiftedArray,I,L}    
    new_cs = refine_view(v)
    new_shift_rng = refine_shift_rng(v, shift_rng)
    res = split_array_broadcast(new_cs, noshift_rng, new_shift_rng)
    @show res
    return res
end

function refine_shift_rng(v::SubArray{T,N,P,I,L}, shift_rng) where {T,N,P,I,L}    
    new_shift_rng = ntuple((d)-> ifelse(isa(v.indices[d],Base.Slice), shift_rng[d], Base.Colon()), ndims(v.parent))
    return new_shift_rng
end
function refine_shift_rng(v, shift_rng)
    return shift_rng
end

"""
    function refine_view(v::SubArray{T,N,P,I,L}, shift_rng)

returns a refined view of a CircShiftedArray as a CircShiftedArray, if necessary. Otherwise just the original array.
find out, if the range of this view crosses any boundary of the parent CircShiftedArray
by calculating the new indices
if, so though an error. find the full slices, which can stay a circ shifted array withs shifts
"""
function refine_view(v::SubArray{T,N,P,I,L}) where {T,N,P<:CircShiftedArray,I,L}
    myshift = csa_shift(v.parent)
    sz = size(v.parent)
    # find out, if the range of this view crosses any boundary of the parent CircShiftedArray
    # by calculating the new indices
    # if, so though an error.
    # find the full slices, which can stay a circ shifted array withs shifts
    sub_rngs = ntuple((d)-> !isa(v.indices[d], Base.Slice), ndims(v.parent))

    new_ids_begin = wrapids(ntuple((d)-> v.indices[d][begin] .- myshift[d], ndims(v.parent)), sz)
    new_ids_end = wrapids(ntuple((d)-> v.indices[d][end] .- myshift[d], ndims(v.parent)), sz)
    if any(sub_rngs .&& (new_ids_end .< new_ids_begin))
        error("a view of a shifted array is not allowed to cross boarders of the original array. Do not use a view here.")
        # potentially this can be remedied, once there is a decent CatViews implementation
    end
    new_rngs = ntuple((d)-> ifelse(isa(v.indices[d],Base.Slice), v.indices[d], new_ids_begin[d]:new_ids_end[d]), ndims(v.parent))
    new_shift = ntuple((d)-> ifelse(isa(v.indices[d],Base.Slice), 0, myshift[d]), ndims(v.parent))
    new_cs = CircShiftedArray((@view v.parent.parent[new_rngs...]), new_shift)
    @show size(v)
    @show size(new_cs)
    @show typeof(new_cs)
    return new_cs
end

function refine_view(csa::AbstractArray)
    return csa
end 


function split_array_broadcast(bc::Base.Broadcast.Broadcasted, noshift_rng, shift_rng)
    # Ref below protects the argument from broadcasting
    @show typeof(bc)
    bc_modified = split_array_broadcast.(bc.args, Ref(noshift_rng), Ref(shift_rng))
    @show typeof(bc_modified)
    # @show size(bc_modified[1])
    res=Base.Broadcast.broadcasted(bc.f, bc_modified...)
    # @show typeof(res)
    # Base.Broadcast.Broadcasted{Style, Tuple{modified_axes...}, F, Args}()
    return res
end

function Base.Broadcast.materialize!(dest::CircShiftedArray{T,N,A,S}, src::CircShiftedArray) where {T,N,A,S}
    Base.Broadcast.materialize!(dest.parent, src.parent)
end

# function copy(CircShiftedArray)
#     collect(CircShiftedArray)
# end

function Base.collect(csa::CircShiftedArray{T,N,A,S}) where {T,N,A,S} 
    # @show "collect"
    circshift(csa.parent, to_tuple(S))
end

# Base.Broadcast.promote_type(::Type{CircShiftedArray{T,N,A}}, ::Type{<:AbstractArray{T,N}}) where {T,N,A<:AbstractArray} = CircShiftedArray{T,N,A}
# two CSAs of the same shift should stay a CSA
# Base.Broadcast.promote_rule(csa1::Type{CircShiftedArray{T,N,A,S}}, csa2::Type{CircShiftedArray{T,N,A,S}}) = CircShiftedArray{T,N,promote_type(typeof(csa1.parent),typeof(csa2.parent)),T}
# broadcasting with a non-CSA should apply the shift
#Base.Broadcast.promote_rule(csa::Type{CircShiftedArray{T,N,A,S}}, na::Type{<:AbstractArray}) where {T,N,A,S} = CircShiftedArray{T,N, promote_type(typeof(csa), typeof(na)), S}
# interaction with numbers should not still stay a CSA
#Base.Broadcast.promote_rule(csa::Type{CircShiftedArray{T,N,A,S}}, na::Type{Number})  where {T,N,A,S} = CircShiftedArray{T,N,promote_type(typeof(csa.parent),typeof(na)),S}

# Base.Broadcast.promote_rule(csa::Type{CircShiftedArray}, na::Type{<:AbstractArray}) = typeof(csa)
# # interaction with numbers should not still stay a CSA
# Base.Broadcast.promote_rule(csa::Type{CircShiftedArray}, na::Type{Number})  = typeof(csa)
# Base.Broadcast.promote_rule(scsa::Type{SubArray{T,N,P,Rngs,B}}, t::T2) where {T,N,P<:CircShiftedArray,Rngs,B,T2}  = typeof(scsa.parent)


#Base.Broadcast.promote_rule(::Type{CircShiftedArray{T,N}}, ::Type{S}) where {T,N,S} = CircShiftedArray{promote_type(T,S),N}
#Base.Broadcast.promote_rule(::Type{CircShiftedArray{T,N}}, ::Type{<:Tuple}, shp...) where {T,N} = CircShiftedArray{T,length(shp)}

# Base.Broadcast.promote_shape(::Type{CircShiftedArray{T,N,A,S}}, ::Type{<:AbstractArray}, ::Type{<:AbstractArray}) where {T,N,A<:AbstractArray,S} = CircShiftedArray{T,N,A,S}
# Base.Broadcast.promote_shape(::Type{CircShiftedArray{T,N,A,S}}, ::Type{<:AbstractArray}, ::Type{<:Number}) where {T,N,A<:AbstractArray,S} = CircShiftedArray{T,N,A,S}

# in most cases by broadcasting over other arrays, we want to apply the circular shift
# function Base.Broadcast.broadcasted(f::Function, csa::CircShiftedArray, other::Vararg) # AbstractArray...
#     circshifted_parent = Base.circshift(csa.parent, csa.myshift)
#     Base.broadcasted(f, circshifted_parent, other...)
# end

# function Base.Broadcast.broadcasted(f::Function, csa::CircShiftedArray{T,N,A,S}, other) where {T,N,A,S}# AbstractArray...
#     @show "Bad1"
#     circshifted_parent = Base.circshift(csa.parent, to_tuple(S))
#     Base.broadcasted(f, circshifted_parent, other)
# end

# function Base.Broadcast.broadcasted(f::Function, other, csa::CircShiftedArray{T,N,A,S}) where {T,N,A,S}# AbstractArray...
#     @show "Bad2"
#     circshifted_parent = Base.circshift(csa.parent, to_tuple(S))
#     Base.broadcasted(f, other, circshifted_parent)
# end

# function Base.Broadcast.broadcasted(f::Function, other::AbstractArray, csa::CircShiftedArray) where {}
#     @show "Bad2"
#     circshifted_parent = Base.circshift(csa.parent, to_tuple(S))
#     Base.broadcasted(f, other, circshifted_parent)
# end

# two times the same shift
# function Base.Broadcast.broadcasted(f::Function, csa1::CircShiftedArray{T1,N1,A1,S}, csa2::CircShiftedArray{T2,N2,A2,S}) where {T1,N1,A1,S, T2,N2,A2} # AbstractArray...
#     @show "Good1"
#     CircShiftedArray(f(csa1.parent, csa2.parent), to_tuple(S))
# end



function Base.similar(arr::CircShiftedArray)
    @show "Similar"
    similar(arr.parent)
end

# function Base.similar(bc::Base.Broadcast.Broadcasted{CircShiftedArrayStyle{N}}, ::ET, ::Any) where {ET,N}
#     @show "Similar Bc"
#     invoke(Base.Broadcast.similar, Tuple{Base.Broadcast.Broadcasted.DefaultArrayStyle{N}}, bc) 
# end

function Base.show(io::IO, mm::MIME"text/plain", cs::CircShiftedArray) 
    CUDA.@allowscalar invoke(Base.show, Tuple{IO, typeof(mm), AbstractArray}, io, mm, cs) 
end
# CUDA.@allowscalar 
# function Base.show(cs::CircShiftedArray)
#     return show(stdout, cs)
# end

# two similarly shifted arrays should remain a shifted array
# Base.Broadcast.broadcasted(::typeof(Base.circshift), csa::CircShiftedArray{T,N,A}, shift::NTuple) where {T,N,A<:AbstractArray{T,N}} =
#     CircShiftedArray{T,N,A}(Base.circshift(csa.parent, shift), wrapshift(csa.myshift .+ shift, size(csa.parent)))

# Base.Broadcast.broadcasted(f::Function, csa::CircShiftedArray, other::Vararg) =
#     Base.broadcasted(f, circshift(csa.parent, csa.myshift), other...)

# my bad idea...:
# function Base.Broadcast.broadcasted(f::Function, csa1::CircShiftedArray, csa2::CircShiftedArray)
#     if 
#     bc = f(csa1.parent, csa2.parent)
#     return CircShiftedArray(bc, csa1.myshift)
# end