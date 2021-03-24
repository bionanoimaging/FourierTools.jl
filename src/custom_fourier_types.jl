##  This View checks for the index to be L1 or the mirrored version (L2)
# and then replaces the value by half of the parent at L1
struct FourierDuplicate{T,N, AA<:AbstractArray{T, N}} <: AbstractArray{T,N}
    parent::AA # holds the data (or is another view)
    D::Int # dimension along which to apply to copy
    L1::Int # low index position to copy from (and half)
    L2::Int # high index positon to copy to (and half)

    function FourierDuplicate(parent::AA, D::Int,L1::Int) where {T,N, AA<:AbstractArray{T, N}}
        if ndims(parent) != N
            throw(DimensionMismatch("parent and indices should have the same dimension, instead they're $(ndims(parent)) and $N."))
        end
        mid = fft_center(size(parent)[D])
        L2 = mid + (mid-L1)
        return new{T,N, AA}(parent, D, L1, L2)
    end
end

Base.IndexStyle(::Type{FD}) where {FD<:FourierDuplicate} = IndexStyle(parenttype(FD))
parenttype(::Type{FourierDuplicate{T,N,AA}}) where {T,N,AA} = AA
parenttype(A::FourierDuplicate) = parenttype(typeof(A))
Base.parent(A::FourierDuplicate) = A.parent 
Base.size(A::FourierDuplicate) = size(parent(A))

@inline function Base.getindex(A::FourierDuplicate{T,N, <:AbstractArray{T, N}}, i::Vararg{Int,N}) where {T,N}
    if i[A.D]==A.L2
        @inbounds return parent(A)[Base.setindex(i,A.L1, A.D)...] / 2
    else 
        @inbounds return parent(A)[i...] / (1 + Int(i[A.D]==A.L1))
        # @inbounds return parent(A)[i...]
    end
end

## This View checks for the index to be L1 
# and then replaces the value by add the value at the mirrored position L2
struct FourierSum{T,N, AA<:AbstractArray{T, N}} <: AbstractArray{T, N}
    parent::AA
    D::Int # dimension along which to apply to copy
    L1::Int # low index position to copy from (and half)
    L2::Int # high index positon to copy to (and half)

    function FourierSum(parent::AA, D::Int,L1::Int) where {T, N, AA<:AbstractArray{T, N}}
        if ndims(parent) != N
            throw(DimensionMismatch("parent and indices should have the same dimension, instead they're $(ndims(parent)) and $N."))
        end
        mid = fft_center(size(parent)[D])
        L2 = mid + (mid-L1)
        return new{T, N, AA}(parent, D, L1, L2)
    end
end
Base.IndexStyle(::Type{FS}) where {FS<:FourierSum} = IndexStyle(parenttype(FS))
parenttype(::Type{FourierSum{T,N,AA}}) where {T,N,AA} = AA
parenttype(A::FourierSum) = parenttype(typeof(A))
Base.parent(A::FourierSum) = A.parent
Base.size(A::FourierSum) = size(parent(A))

@inline function Base.getindex(A::FourierSum{T,N, <:AbstractArray{T, N}}, i::Vararg{Int,N}) where {T,N}
    if i[A.D]==A.L1
        @inbounds return parent(A)[i...] + parent(A)[Base.setindex(i, A.L2, A.D)...]
    else 
        @inbounds return parent(A)[i...]
    end
end

