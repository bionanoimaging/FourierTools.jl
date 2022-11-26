##  This View checks for the index to be L1 or the mirrored version (L2)
# and then replaces the value by half of the parent at L1
# do_split is a Bool that indicates whether this mechanism is active. It is needed for type stability reasons of functions returnting this type
struct FourierSplit{T,N, AA<:AbstractArray{T, N}} <: AbstractArray{T,N}
    parent::AA # holds the data (or is another view)
    D::Int # dimension along which to apply to copy
    L1::Int # low index position to copy from (and half)
    L2::Int # high index positon to copy to (and half)
    do_split::Bool

    # This version below is needed to avoid a split for the firs rft dimension but still return half the value
    # FFTs and other RFT dimension should use the version without L2
    function FourierSplit(parent::AA, D::Int,L1::Int,L2::Int, do_split::Bool) where {T,N, AA<:AbstractArray{T, N}}
        return new{T,N, AA}(parent, D, L1, L2, do_split)
    end
    function FourierSplit(parent::AA, D::Int, L1::Int, do_split::Bool) where {T,N, AA<:AbstractArray{T, N}}
        mid = fft_center(size(parent)[D])
        L2 = mid + (mid-L1)
        return FourierSplit(parent, D,L1,L2, do_split)
    end
end

Base.IndexStyle(::Type{FD}) where {FD<:FourierSplit} = IndexStyle(parenttype(FD))
parenttype(::Type{FourierSplit{T,N,AA}}) where {T,N,AA} = AA
parenttype(A::FourierSplit) = parenttype(typeof(A))
Base.parent(A::FourierSplit) = A.parent 
Base.size(A::FourierSplit) = size(parent(A))

@inline function Base.getindex(A::FourierSplit{T,N, <:AbstractArray{T, N}}, i::Vararg{Int,N}) where {T,N}
    if (i[A.D]==A.L2 || i[A.D]==A.L1) && A.do_split # index along this dimension A.D corrsponds to slice L2
        # not that "setindex" in the line below modifies only the index, not the array
        @inbounds return parent(A)[Base.setindex(i,A.L1, A.D)...] / 2
    else i[A.D]==A.L2
        @inbounds return parent(A)[i...]
        # @inbounds return parent(A)[i...]
    end
end

## This View checks for the index to be L1 
# and then replaces the value by add the value at the mirrored position L2
# do_join is a Bool that indicates whether this mechanism is active. It is needed for type stability reasons of functions returnting this type
struct FourierJoin{T,N, AA<:AbstractArray{T, N}} <: AbstractArray{T, N}
    parent::AA
    D::Int # dimension along which to apply to copy
    L1::Int # low index position to copy from (and half)
    L2::Int # high index positon to copy to (and half)
    do_join::Bool

    # This version below is needed to avoid a split for the firs rft dimension but still return half the value
    # FFTs and other RFT dimension should use the version without L2
    function FourierJoin(parent::AA, D::Int, L1::Int, L2::Int, do_join::Bool) where {T, N, AA<:AbstractArray{T, N}}
        return new{T, N, AA}(parent, D, L1, L2, do_join)
    end

    function FourierJoin(parent::AA, D::Int,L1::Int, do_join::Bool) where {T, N, AA<:AbstractArray{T, N}}
        mid = fft_center(size(parent)[D])
        L2 = mid + (mid-L1)
        return FourierJoin(parent, D, L1, L2, do_join)
    end
end
Base.IndexStyle(::Type{FS}) where {FS<:FourierJoin} = IndexStyle(parenttype(FS))
parenttype(::Type{FourierJoin{T,N,AA}}) where {T,N,AA} = AA
parenttype(A::FourierJoin) = parenttype(typeof(A))
Base.parent(A::FourierJoin) = A.parent
Base.size(A::FourierJoin) = size(parent(A))

@inline function Base.getindex(A::FourierJoin{T,N, <:AbstractArray{T, N}}, i::Vararg{Int,N}) where {T,N}
    if i[A.D]==A.L1 && A.do_join
        @inbounds return (parent(A)[i...] + parent(A)[Base.setindex(i, A.L2, A.D)...])
    else 
        @inbounds return (parent(A)[i...])
    end
end

