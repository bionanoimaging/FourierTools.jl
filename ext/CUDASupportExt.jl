module CUDASupportExt
using CUDA 
using Adapt
# using ShiftedArrays
using FourierTools
using Base # to allow displaying such arrays without causing the single indexing CUDA error

# define adapt structures for the ShiftedArrays model. This will not be needed if the PR is merged:
# Adapt.adapt_structure(to, x::FourierTools.CircShiftedArray{T, D}) where {T, D} = FourierTools.CircShiftedArray(adapt(to, parent(x)), FourierTools.shifts(x));
# parent_type(::Type{FourierTools.CircShiftedArray{T, N, A, S}})  where {T, N, A, S} = A
# Base.Broadcast.BroadcastStyle(::Type{T})  where {T<:FourierTools.CircShiftedArray} = Base.Broadcast.BroadcastStyle(parent_type(T))

Adapt.adapt_structure(to, x::FourierTools.CircShiftedArray{T, N, S}) where {T, N, S} = FourierTools.CircShiftedArray(adapt(to, parent(x)), FourierTools.shifts(x));
parent_type(::Type{FourierTools.CircShiftedArray{T, N, S}})  where {T, N, S} = S

# Base.Broadcast.BroadcastStyle(::Type{T}) where {T2, N, S, T <:FourierTools.CircShiftedArray{T2, N, S}}  = Base.Broadcast.BroadcastStyle(parent_type(T))
function Base.Broadcast.BroadcastStyle(::Type{T})  where {CT, N, CD, T<:FourierTools.CircShiftedArray{<:Any,<:Any,<:CuArray{CT,N,CD}}}
    CUDA.CuArrayStyle{N,CD}()
end

# Define the BroadcastStyle for SubArray of MutableShiftedArray with CuArray
function Base.Broadcast.BroadcastStyle(::Type{T})  where {CT, N, CD, T<:SubArray{<:Any, <:Any, <:FourierTools.CircShiftedArray{<:Any,<:Any,<:CuArray{CT,N,CD}}}}
    CUDA.CuArrayStyle{N,CD}()
end

# Define the BroadcastStyle for ReshapedArray of MutableShiftedArray with CuArray
function Base.Broadcast.BroadcastStyle(::Type{T})  where {CT, N, CD, T<:Base.ReshapedArray{<:Any, <:Any, <:FourierTools.CircShiftedArray{<:Any,<:Any,<:CuArray{CT,N,CD}}, <:Any}}
    CUDA.CuArrayStyle{N,CD}()
end

function Base.collect(x::T)  where {CT, N, CD, T<:FourierTools.CircShiftedArray{<:Any,<:Any,<:CuArray{CT,N,CD}}}
    return copy(x) # stay on the GPU        
end

function Base.Array(x::T)  where {CT, N, CD, T<:FourierTools.CircShiftedArray{<:Any,<:Any,<:CuArray{CT,N,CD}}}
    return Array(copy(x)) # remove from GPU
end

function Base.:(==)(x::T, y::AbstractArray)  where {CT, N, CD, T<:FourierTools.CircShiftedArray{<:Any,<:Any,<:CuArray{CT,N,CD}}}    
    return all(x .== y)
end

function Base.:(==)(y::AbstractArray, x::T)  where {CT, N, CD, T<:FourierTools.CircShiftedArray{<:Any,<:Any,<:CuArray{CT,N,CD}}}    
    return all(x .== y)
end

function Base.:(==)(x::T, y::T)  where {CT, N, CD, T<:FourierTools.CircShiftedArray{<:Any,<:Any,<:CuArray{CT,N,CD}}}    
    return all(x .== y)
end

function Base.isapprox(x::T, y::AbstractArray; atol=0, rtol=atol>0 ? 0 : sqrt(eps(real(eltype(x)))), va...)  where {CT, N, CD, T<:FourierTools.CircShiftedArray{<:Any,<:Any,<:CuArray{CT,N,CD}}}    
    atol = (atol != 0) ? atol : rtol * maximum(abs.(x))
    return all(abs.(x .- y) .<= atol)
end

function Base.isapprox(y::AbstractArray, x::T; atol=0, rtol=atol>0 ? 0 : sqrt(eps(real(eltype(x)))),  va...)  where {CT, N, CD, T<:FourierTools.CircShiftedArray{<:Any,<:Any,<:CuArray{CT,N,CD}}}    
    atol = (atol != 0) ? atol : rtol * maximum(abs.(x))
    return all(abs.(x .- y) .<= atol)
end

function Base.isapprox(x::T, y::T; atol=0, rtol=atol>0 ? 0 : sqrt(eps(real(eltype(x)))),  va...)  where {CT, N, CD, T<:FourierTools.CircShiftedArray{<:Any,<:Any,<:CuArray{CT,N,CD}}}    
    atol = (atol != 0) ? atol : rtol * maximum(abs.(x))
    return all(abs.(x .- y) .<= atol)
end

function Base.show(io::IO, mm::MIME"text/plain", cs::T) where {CT, N, CD, T<:FourierTools.CircShiftedArray{<:Any,<:Any,<:CuArray{CT,N,CD}}}
    CUDA.@allowscalar invoke(Base.show, Tuple{IO, typeof(mm), AbstractArray}, io, mm, cs) 
end


# cu_storage_type(::Type{T}) where {CT,CN,CD,T<:CuArray{CT,CN,CD}} = CD
# lets do this for the ShiftedArray type
# Adapt.adapt_structure(to, x::ShiftedArray{T, M, N}) where {T, M, N} = ShiftedArray(adapt(to, parent(x)), FourierTools.shifts(x); default=ShiftedArrays.default(x));

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
    return Array(copy(x)) # remove from GPU
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

function FourierTools.optional_collect(a::CuArray)
    a 
end

end