export CircShiftedArray
using Base
# a = reshape(1:100,(10,10)) .+ 0
# c = CircShiftedArray(a,(3,3));
# d = c .+ c;

struct CircShiftedArray{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    parent::A
    myshift::NTuple{N,Int}

    function CircShiftedArray(parent::A, myshift::NTuple{N,Int}) where {T,N,A<:AbstractArray{T,N}}
        new{T,N,A}(parent, wrapshift(myshift, size(parent)))
    end
end

wrapshift(shift::NTuple, dims::NTuple) = ntuple(i -> mod1(shift[i], dims[i]), length(dims))

Base.size(csa::CircShiftedArray) = size(csa.parent)
Base.size(csa::CircShiftedArray, d::Int) = size(csa.parent, d)
Base.IndexStyle(::Type{<:CircShiftedArray}) = IndexLinear()

# mod1 avoids first subtracting one and then adding one
Base.getindex(csa::CircShiftedArray, i::Vararg{Int,N}) where {N} = 
    getindex(csa.parent, (mod1(i[j]-csa.myshift[j], size(csa.parent, j)) for j in 1:N)...)

Base.setindex!(csa::CircShiftedArray, v, i::Vararg{Int,N}) where {N} = 
    (setindex!(csa.parent, v, (mod1(i[j]-csa.myshift[j], size(csa.parent, j)) for j in 1:N)...); v)

Base.Broadcast.materialize(csa::CircShiftedArray) = circshift(csa.parent, csa.myshift)

Base.collect(csa::CircShiftedArray) = circshift(csa.parent, csa.myshift)

# Base.Broadcast.promote_type(::Type{CircShiftedArray{T,N,A}}, ::Type{<:AbstractArray{T,N}}) where {T,N,A<:AbstractArray} = CircShiftedArray{T,N,A}
Base.Broadcast.promote_rule(::Type{CircShiftedArray{T1,N1,A1}}, arg2::Type{<:AbstractArray{T2,N2}}) where {T1,N1,A1<:AbstractArray,T2,N2} = CircShiftedArray{promote_type(T1,T2),max(N1,N2), promote_type(A1,typeof(arg2))}
#Base.Broadcast.promote_rule(::Type{CircShiftedArray{T,N}}, ::Type{S}) where {T,N,S} = CircShiftedArray{promote_type(T,S),N}
#Base.Broadcast.promote_rule(::Type{CircShiftedArray{T,N}}, ::Type{<:Tuple}, shp...) where {T,N} = CircShiftedArray{T,length(shp)}

Base.Broadcast.promote_shape(::Type{CircShiftedArray{T,N,A}}, ::Type{<:AbstractArray}, ::Type{<:AbstractArray}) where {T,N,A<:AbstractArray} = CircShiftedArray{T,N,A}
Base.Broadcast.promote_shape(::Type{CircShiftedArray{T,N,A}}, ::Type{<:AbstractArray}, ::Type{<:Number}) where {T,N,A<:AbstractArray} = CircShiftedArray{T,N,A}

# in most cases by broadcasting over other arrays, we want to apply the circular shift
# function Base.Broadcast.broadcasted(f::Function, csa::CircShiftedArray, other::Vararg) # AbstractArray...
#     circshifted_parent = Base.circshift(csa.parent, csa.myshift)
#     Base.broadcasted(f, circshifted_parent, other...)
# end

function Base.Broadcast.broadcasted(f::Function, csa::CircShiftedArray, other) # AbstractArray...
    circshifted_parent = Base.circshift(csa.parent, csa.myshift)
    Base.broadcasted(f, circshifted_parent, other)
end

function Base.Broadcast.broadcasted(f::Function, other, csa::CircShiftedArray) # AbstractArray...
    circshifted_parent = Base.circshift(csa.parent, csa.myshift)
    Base.broadcasted(f, other, circshifted_parent)
end

function Base.Broadcast.broadcasted(f::Function, csa1::CircShiftedArray, csa2::CircShiftedArray) # AbstractArray...
    circshifted_parent1 = Base.circshift(csa1.parent, csa1.myshift)
    circshifted_parent2 = Base.circshift(csa2.parent, csa2.myshift)
    Base.broadcasted(f, circshifted_parent1, circshifted_parent2)
end

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