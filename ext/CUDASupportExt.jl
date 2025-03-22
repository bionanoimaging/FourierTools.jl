module CUDASupportExt
using CUDA 
using Adapt
# using ShiftedArrays
using FourierTools
using IndexFunArrays # to prevent a stack overflow in get_base_arr
using Base # to allow displaying such arrays without causing the single indexing CUDA error

get_base_arr(arr::Array) = arr
get_base_arr(arr::CuArray) = arr
get_base_arr(arr::IndexFunArray) = arr
function get_base_arr(arr::AbstractArray) 
    get_base_arr(parent(arr))
end

# define a number of Union types to not repeat all definitions for each type
AllShiftedType = Union{FourierTools.CircShiftedArray{<:Any,<:Any,<:Any},
                            FourierTools.FourierSplit{<:Any,<:Any,<:Any},
                            FourierTools.FourierJoin{<:Any,<:Any,<:Any}}

# these are special only if a CuArray is wrapped

AllSubArrayType = Union{SubArray{<:Any, <:Any, <:AllShiftedType, <:Any, <:Any},
                        Base.ReshapedArray{<:Any, <:Any, <:AllShiftedType, <:Any},
                        SubArray{<:Any, <:Any, <:Base.ReshapedArray{<:Any, <:Any, <:AllShiftedType, <:Any}, <:Any, <:Any}}
AllShiftedAndViews = Union{AllShiftedType, AllSubArrayType}

AllShiftedTypeCu{N, CD} = Union{FourierTools.CircShiftedArray{<:Any,<:Any,<:CuArray{<:Any,N,CD}},
                            FourierTools.FourierSplit{<:Any,<:Any,<:CuArray{<:Any,N,CD}},
                            FourierTools.FourierJoin{<:Any,<:Any,<:CuArray{<:Any,N,CD}}}
AllSubArrayTypeCu{N, CD} = Union{SubArray{<:Any, <:Any, <:AllShiftedTypeCu{N,CD}, <:Any, <:Any},
                                 Base.ReshapedArray{<:Any, <:Any, <:AllShiftedTypeCu{N,CD}, <:Any},
                                 SubArray{<:Any, <:Any, <:Base.ReshapedArray{<:Any, <:Any, <:AllShiftedTypeCu{N,CD}, <:Any}, <:Any, <:Any}}
AllShiftedAndViewsCu{N, CD} = Union{AllShiftedTypeCu{N, CD}, AllSubArrayTypeCu{N, CD}}

# define adapt structures for the ShiftedArrays model. This will not be needed if the PR is merged:
# Adapt.adapt_structure(to, x::FourierTools.CircShiftedArray{T, D}) where {T, D} = FourierTools.CircShiftedArray(adapt(to, parent(x)), FourierTools.shifts(x));
# parent_type(::Type{FourierTools.CircShiftedArray{T, N, A, S}})  where {T, N, A, S} = A
# Base.Broadcast.BroadcastStyle(::Type{T})  where {T<:FourierTools.CircShiftedArray} = Base.Broadcast.BroadcastStyle(parent_type(T))

Adapt.adapt_structure(to, x::FourierTools.CircShiftedArray{T, N, S}) where {T, N, S} = FourierTools.CircShiftedArray(adapt(to, parent(x)), FourierTools.shifts(x));
# parent_type(::Type{FourierTools.CircShiftedArray{T, N, S}})  where {T, N, S} = S

# Base.Broadcast.BroadcastStyle(::Type{T}) where {T2, N, S, T <:FourierTools.CircShiftedArray{T2, N, S}}  = Base.Broadcast.BroadcastStyle(parent_type(T))
# function Base.Broadcast.BroadcastStyle(::Type{T})  where {CT, N, CD, T<:FourierTools.CircShiftedArray{<:Any,<:Any,<:CuArray{CT,N,CD}}}
#     CUDA.CuArrayStyle{N,CD}()
# end
function Base.Broadcast.BroadcastStyle(::Type{T})  where {N, CD, T<:AllShiftedTypeCu{N, CD}}
    CUDA.CuArrayStyle{N,CD}()
end

# Define the BroadcastStyle for SubArray of MutableShiftedArray with CuArray
# function Base.Broadcast.BroadcastStyle(::Type{T})  where {CT, N, CD, T<:SubArray{<:Any, <:Any, <:FourierTools.CircShiftedArray{<:Any,<:Any,<:CuArray{CT,N,CD}}}}
#     CUDA.CuArrayStyle{N,CD}()
# end
function Base.Broadcast.BroadcastStyle(::Type{T})  where {N, CD, T<:AllSubArrayTypeCu{N, CD}}
    CUDA.CuArrayStyle{N,CD}()
end

# Define the BroadcastStyle for ReshapedArray of MutableShiftedArray with CuArray
# function Base.Broadcast.BroadcastStyle(::Type{T})  where {CT, N, CD, T<:Base.ReshapedArray{<:Any, <:Any, <:FourierTools.CircShiftedArray{<:Any,<:Any,<:CuArray{CT,N,CD}}, <:Any}}
#     CUDA.CuArrayStyle{N,CD}()
# end

function Base.copy(s::AllShiftedAndViews) # where {CT, N, CD, T<:FourierTools.CircShiftedArray{<:Any,<:Any,<:CuArray{CT,N,CD}}}
    res = similar(get_base_arr(s), eltype(s), size(s));
    # @show "copy here"
    # @show s.D
    res .= s
    # CUDA.@allowscalar @show res[5]
    return res
end

function Base.collect(x::AllShiftedAndViews)  # where {CT, N, CD, T<:FourierTools.CircShiftedArray{<:Any,<:Any,<:CuArray{CT,N,CD}}}
    return copy(x) # stay on the GPU        
end

function Base.Array(x::AllShiftedAndViews) # where {CT, N, CD, T<:FourierTools.CircShiftedArray{<:Any,<:Any,<:CuArray{CT,N,CD}}}
    return Array(copy(x)) # remove from GPU
end

function Base.:(==)(x::AllShiftedAndViewsCu, y::AbstractArray)  # where {CT, N, CD, T<:FourierTools.CircShiftedArray{<:Any,<:Any,<:CuArray{CT,N,CD}}}    
    return all(x .== y)
end

function Base.:(==)(y::AbstractArray, x::AllShiftedAndViewsCu) # where {CT, N, CD, T<:FourierTools.CircShiftedArray{<:Any,<:Any,<:CuArray{CT,N,CD}}}    
    return all(x .== y)
end

function Base.:(==)(x::AllShiftedAndViewsCu, y::AllShiftedAndViewsCu) # where {CT, N, CD, T<:FourierTools.CircShiftedArray{<:Any,<:Any,<:CuArray{CT,N,CD}}}    
    return all(x .== y)
end

function Base.isapprox(x::AllShiftedAndViewsCu, y::AbstractArray; atol=0, rtol=atol>0 ? 0 : sqrt(eps(real(eltype(x)))), va...) # where {CT, N, CD, T<:FourierTools.CircShiftedArray{<:Any,<:Any,<:CuArray{CT,N,CD}}}    
    atol = (atol != 0) ? atol : rtol * maximum(abs.(x))
    return all(abs.(x .- y) .<= atol)
end

function Base.isapprox(y::AbstractArray, x::AllShiftedAndViewsCu; atol=0, rtol=atol>0 ? 0 : sqrt(eps(real(eltype(x)))),  va...) # where {CT, N, CD, T<:FourierTools.CircShiftedArray{<:Any,<:Any,<:CuArray{CT,N,CD}}}    
    atol = (atol != 0) ? atol : rtol * maximum(abs.(x))
    return all(abs.(x .- y) .<= atol)
end

function Base.isapprox(x::AllShiftedAndViewsCu, y::AllShiftedAndViewsCu; atol=0, rtol=atol>0 ? 0 : sqrt(eps(real(eltype(x)))),  va...) # where {CT, N, CD, T<:FourierTools.CircShiftedArray{<:Any,<:Any,<:CuArray{CT,N,CD}}}    
    atol = (atol != 0) ? atol : rtol * maximum(abs.(x))
    return all(abs.(x .- y) .<= atol)
end

function Base.show(io::IO, mm::MIME"text/plain", cs::AllShiftedAndViews) # where {CT, N, CD, T<:FourierTools.CircShiftedArray{<:Any,<:Any,<:CuArray{CT,N,CD}}}
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
Adapt.adapt_structure(to, x::FourierTools.FourierSplit{T, M, AA, D}) where {T, M, AA, D} = FourierTools.FourierSplit(adapt(to, parent(x)), Val(D), x.L1, x.L2, x.do_split);
# parent_type(::Type{FourierTools.FourierSplit{T, N, S}})  where {T, N, S} = S

# function Base.Broadcast.BroadcastStyle(::Type{T})  where (CT,CN,CD,T<: ShiftedArray{<:Any,<:Any,<:Any,<:CuArray})
# function Base.Broadcast.BroadcastStyle(::Type{T})  where {T2, N, CD, T<:FourierTools.FourierSplit{<:Any,<:Any,<:CuArray{T2,N,CD}}}
#     CUDA.CuArrayStyle{N,CD}()
# end

# function Base.Broadcast.BroadcastStyle(::Type{T})  where {CT, N, CD, T<:SubArray{<:Any, <:Any, <:FourierTools.FourierSplit{<:Any,<:Any,<:CuArray{CT,N,CD}}}}
#     CUDA.CuArrayStyle{N,CD}()
# end

# function Base.copy(s::FourierTools.FourierSplit) #  where {CT, N, CD, T<:FourierTools.FourierSplit{<:Any,<:Any,<:CuArray{CT,N,CD}}}
#     res = similar(get_base_arr(s), eltype(s), size(s));
#     res .= s
# end

# function Base.collect(x::FourierTools.FourierSplit) # where {CT, N, CD, T<:FourierTools.FourierSplit{<:Any,<:Any,<:CuArray{CT,N,CD}}}
#     return copy(x) # stay on the GPU        
# end

# function Base.Array(x::T)  where {CT, N, CD, T<:FourierTools.FourierSplit{<:Any,<:Any,<:CuArray{CT,N,CD}}}
#     return Array(copy(x)) # remove from GPU
# end

# function Base.:(==)(x::T, y::AbstractArray)  where {CT, N, CD, T<:FourierTools.FourierSplit{<:Any,<:Any,<:CuArray{CT,N,CD}}}    
#     return all(x .== y)
# end

# function Base.:(==)(y::AbstractArray, x::T)  where {CT, N, CD, T<:FourierTools.FourierSplit{<:Any,<:Any,<:CuArray{CT,N,CD}}}    
#     return all(x .== y)
# end

# function Base.:(==)(x::T, y::T)  where {CT, N, CD, T<:FourierTools.FourierSplit{<:Any,<:Any,<:CuArray{CT,N,CD}}}    
#     return all(x .== y)
# end

# function Base.show(io::IO, mm::MIME"text/plain", cs::FourierTools.FourierSplit) # where {CT, N, CD, T<:FourierTools.FourierSplit{<:Any,<:Any,<:CuArray{CT,N,CD}}}
#     CUDA.@allowscalar invoke(Base.show, Tuple{IO, typeof(mm), AbstractArray}, io, mm, cs) 
# end

# for FourierJoin
Adapt.adapt_structure(to, x::FourierTools.FourierJoin{T, M, AA, D}) where {T, M, AA, D} = FourierTools.FourierJoin(adapt(to, parent(x)), Val(D), x.L1, x.L2, x.do_join);
# parent_type(::Type{FourierTools.FourierJoin{T, N, S}})  where {T, N, S} = S

# function Base.Broadcast.BroadcastStyle(::Type{T})  where {T2, N, CD, T<:FourierTools.FourierJoin{<:Any,<:Any,<:CuArray{T2,N,CD}}}
#     CUDA.CuArrayStyle{N,CD}()
# end

# function Base.copy(s::FourierTools.FourierJoin)  # where {CT, N, CD, T<:FourierTools.FourierJoin{<:Any,<:Any,<:CuArray{CT,N,CD}}}
#     res = similar(get_base_arr(s), eltype(s), size(s));
#     res .= s
# end

# function Base.collect(x::FourierTools.FourierJoin) # where {CT, N, CD, T<:FourierTools.FourierJoin{<:Any,<:Any,<:CuArray{CT,N,CD}}}
#     return copy(x) # stay on the GPU        
# end

# function Base.Array(x::T)  where {CT, N, CD, T<:FourierTools.FourierJoin{<:Any,<:Any,<:CuArray{CT,N,CD}}}
#     return Array(copy(x)) # remove from GPU
# end

# function Base.show(io::IO, mm::MIME"text/plain", cs::FourierTools.FourierJoin) # where {CT, N, CD, T<:FourierTools.FourierJoin{<:Any,<:Any,<:CuArray{CT,N,CD}}}
#     CUDA.@allowscalar invoke(Base.show, Tuple{IO, typeof(mm), AbstractArray}, io, mm, cs) 
# end

### addition functions specific to CUDA

function FourierTools.optional_collect(a::CuArray)
    a 
end

end