module CUDASupportExt
using CUDA 
using Adapt
# using ShiftedArrays
using FourierTools
using IndexFunArrays # to prevent a recuursive stack overflow in get_base_arr
using Base 

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

Adapt.adapt_structure(to, x::FourierTools.CircShiftedArray{T, N, S}) where {T, N, S} = FourierTools.CircShiftedArray(adapt(to, parent(x)), FourierTools.shifts(x));
Adapt.adapt_structure(to, x::FourierTools.FourierSplit{T, M, AA, D}) where {T, M, AA, D} = FourierTools.FourierSplit(adapt(to, parent(x)), Val(D), x.L1, x.L2, x.do_split);
Adapt.adapt_structure(to, x::FourierTools.FourierJoin{T, M, AA, D}) where {T, M, AA, D} = FourierTools.FourierJoin(adapt(to, parent(x)), Val(D), x.L1, x.L2, x.do_join);

function Base.Broadcast.BroadcastStyle(::Type{T})  where {N, CD, T<:AllShiftedTypeCu{N, CD}}
    CUDA.CuArrayStyle{N,CD}()
end

# Define the BroadcastStyle for SubArray of MutableShiftedArray with CuArray

function Base.Broadcast.BroadcastStyle(::Type{T})  where {N, CD, T<:AllSubArrayTypeCu{N, CD}}
    CUDA.CuArrayStyle{N,CD}()
end

function Base.copy(s::AllShiftedAndViews)
    res = similar(get_base_arr(s), eltype(s), size(s));
    res .= s
    return res
end

function Base.collect(x::AllShiftedAndViews) 
    return copy(x) # stay on the GPU        
end

function Base.Array(x::AllShiftedAndViews) 
    return Array(copy(x)) # remove from GPU
end

function Base.:(==)(x::AllShiftedAndViewsCu, y::AbstractArray) 
    return all(x .== y)
end

function Base.:(==)(y::AbstractArray, x::AllShiftedAndViewsCu) 
    return all(x .== y)
end

function Base.:(==)(x::AllShiftedAndViewsCu, y::AllShiftedAndViewsCu) 
    return all(x .== y)
end

function Base.isapprox(x::AllShiftedAndViewsCu, y::AbstractArray; atol=0, rtol=atol>0 ? 0 : sqrt(eps(real(eltype(x)))), va...) 
    atol = (atol != 0) ? atol : rtol * maximum(abs.(x))
    return all(abs.(x .- y) .<= atol)
end

function Base.isapprox(y::AbstractArray, x::AllShiftedAndViewsCu; atol=0, rtol=atol>0 ? 0 : sqrt(eps(real(eltype(x)))),  va...)     
    atol = (atol != 0) ? atol : rtol * maximum(abs.(x))
    return all(abs.(x .- y) .<= atol)
end

function Base.isapprox(x::AllShiftedAndViewsCu, y::AllShiftedAndViewsCu; atol=0, rtol=atol>0 ? 0 : sqrt(eps(real(eltype(x)))),  va...) # where {CT, N, CD, T<:FourierTools.CircShiftedArray{<:Any,<:Any,<:CuArray{CT,N,CD}}}    
    atol = (atol != 0) ? atol : rtol * maximum(abs.(x))
    return all(abs.(x .- y) .<= atol)
end

function Base.show(io::IO, mm::MIME"text/plain", cs::AllShiftedAndViews) 
    CUDA.@allowscalar invoke(Base.show, Tuple{IO, typeof(mm), AbstractArray}, io, mm, cs) 
end

### addition functions specific to CUDA

function FourierTools.optional_collect(a::CuArray)
    a 
end

end