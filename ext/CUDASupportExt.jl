module CUDASupportExt
using CUDA 
using Adapt
using ShiftedArrays
using FourierTools
using Base # to allow displaying such arrays without causing the single indexing CUDA error

# define adapt structures for the ShiftedArrays model. This will not be needed if the PR is merged:
Adapt.adapt_structure(to, x::CircShiftedArray{T, D}) where {T, D} = CircShiftedArray(adapt(to, parent(x)), shifts(x));
parent_type(::Type{CircShiftedArray{T, N, S}})  where {T, N, S} = S
Base.Broadcast.BroadcastStyle(::Type{T})  where {T<:CircShiftedArray} = Base.Broadcast.BroadcastStyle(parent_type(T))

# cu_storage_type(::Type{T}) where {CT,CN,CD,T<:CuArray{CT,CN,CD}} = CD
# lets do this for the ShiftedArray type
# Adapt.adapt_structure(to, x::ShiftedArray{T, M, N}) where {T, M, N} = ShiftedArray(adapt(to, parent(x)), shifts(x); default=ShiftedArrays.default(x));

# # function Base.Broadcast.BroadcastStyle(::Type{T})  where (CT,CN,CD,T<: ShiftedArray{<:Any,<:Any,<:Any,<:CuArray})
# function Base.Broadcast.BroadcastStyle(::Type{T})  where {T2, N, CD, T<:ShiftedArray{<:Any,<:Any,<:Any,<:CuArray{T2,N,CD}}}
#     CUDA.CuArrayStyle{N,CD}()
# end

# lets do this for the FourierSplit
Adapt.adapt_structure(to, x::FourierTools.FourierSplit{T, M, AA}) where {T, M, AA} = FourierTools.FourierSplit(adapt(to, parent(x)), ndims(x), x.L1, x.L2, x.do_split);

# function Base.Broadcast.BroadcastStyle(::Type{T})  where (CT,CN,CD,T<: ShiftedArray{<:Any,<:Any,<:Any,<:CuArray})
function Base.Broadcast.BroadcastStyle(::Type{T})  where {T2, N, CD, T<:FourierTools.FourierSplit{<:Any,<:Any,<:CuArray{T2,N,CD}}}
    CUDA.CuArrayStyle{N,CD}()
end

function Base.collect(x::T)  where {CT, N, CD, T<:FourierTools.FourierSplit{<:Any,<:Any,<:CuArray{CT,N,CD}}}
    return copy(x) # stay on the GPU        
end

function Base.Array(x::T)  where {CT, N, CD, T<:FourierTools.FourierSplit{<:Any,<:Any,<:CuArray{CT,N,CD}}}
    return Array(copy(x)) # stay on the GPU        
end

function Base.:(==)(x::T, y::AbstractArray)  where {CT, N, CD, T<:FourierTools.FourierSplit{<:Any,<:Any,<:CuArray{CT,N,CD}}}    
    return all(x .== y)
end

function Base.:(==)(y::AbstractArray, x::T)  where {CT, N, CD, T<:FourierTools.FourierSplit{<:Any,<:Any,<:CuArray{CT,N,CD}}}    
    return all(x .== y)
end

function Base.:(==)(x::T, y::T)  where {CT, N, CD, T<:FourierTools.FourierSplit{<:Any,<:Any,<:CuArray{CT,N,CD}}}    
    return all(x .== y)
end

end