export ft,ift, rft, irft, rft_size, fft_center


"""
    fft_center(x)

Returns the center of a size in Fourier sense and Julia 
1-based indices.
"""
function fft_center(x)
    return x ÷ 2 + 1
end

"""
    rft_size(x)

Returns the size of an rft or rfft performed on the data x, without performing the rfft.
"""
function rft_size(x; dim=1)
    return Base.setindex(size(x),size(x,dim)÷2+1,dim)
end



"""
    ft_center_diff(s [, dims])

Calculates how much each dimension must be shifted that the
center frequency is at the Fourier center.
This if for a normal `fft`
"""
function ft_center_diff(s::NTuple{N, T}, dims=ntuple(identity, Val(N))) where {N, T}
    ntuple(i -> i ∈ dims ?  s[i] ÷ 2 : 0 , N)
end


"""
    rft_center_diff(s [, dims])

Calculates how much each dimension must be shifted that the
center frequency is at the Fourier center.
This is for `rfft`. The `dims[1]` must be therefore not shifted!
"""
function rft_center_diff(sz::NTuple{N, T}, dims=ntuple(identity, Val(N))::NTuple) where {N,T}
    ntuple(i -> i == first(dims) ? 0 : i ∈ dims ? sz[i] ÷ 2 : 0, N)
    # Tuple(d == 1 ? 0 : sz[d].÷2 for d in 1:length(sz))
end


"""
    ft(A [, dims])

Result is semantically equivalent to `fftshift(fft(A, dims), dims)`
However, the shift is done with `ShiftedArrays` and therefore doesn't allocate memory.
"""
function ft(mat::AbstractArray{T, N}, dims=ntuple(identity, Val(N))) where {T, N}
    return fftshift_view(fft(mat, dims), dims)
end

"""
    ift(A [, dims])

Result is semantically equivalent to `ifft(ifftshift(A), dims), dims)`
"""
function ift(mat::AbstractArray{T, N}, dims=ntuple(identity, Val(N))) where {T, N}
    # remove ift shift
    # return ifft(collect(ifftshift_view(mat, dims)), dims);
    return ifft(ifftshift(mat, dims), dims)
end

"""
    fftshift_view(A [, dims])

Result is semantically equivalent to `fftshift(A, dims)` but returns 
a view instead. 
"""
function fftshift_view(mat::AbstractArray{T, N}, dims=ntuple(identity, Val(N))) where {T, N}
    ShiftedArrays.circshift(mat, ft_center_diff(size(mat), dims))
end

"""
    ifftshift_view(A [, dims])

Result is semantically equivalent to `fftshift(A, dims)` but returns 
a view instead. 
"""
function ifftshift_view(mat::AbstractArray{T, N}, dims=ntuple(identity, Val(N))) where {T, N}
    ShiftedArrays.circshift(mat, .-(ft_center_diff(size(mat), dims)))
end



"""
    rft(A [, dims])

Calculates a `rfft(A, dims)` and then shift the frequencies to the center.
`dims[1]` is not shifted, because there is no negative and positive frequency.
The shift is done with `ShiftedArrays` and therefore doesn't allocate memory.
"""
function rft(mat::AbstractArray{T, N}, dims=ntuple(identity, Val(N))) where {T, N}
    rfftshift_view(rfft(mat, dims), dims);
end

"""
    irft(A, d, [, dims])

Calculates a `irfft(A, d, dims)` and then shift the frequencies back to the corner.
`dims[1]` is not shifted, because there is no negative and positive frequency.
The shift is done with `ShiftedArrays` and therefore doesn't allocate memory.
"""
function irft(mat::AbstractArray{T, N}, d::Int, dims=ntuple(identity, Val(N))) where {T, N}
    irfft(collect(irfftshift_view(mat, dims)), d, dims);
end

"""
    rfftshift_view(A, dims)

Shifts the frequencies to the center expect for `dims[1]` because there os no negative
and positive frequency.
"""
function rfftshift_view(mat::AbstractArray{T, N}, dims=ntuple(identity, Val(N))) where {T, N}
    ShiftedArrays.circshift(mat, rft_center_diff(size(mat), dims))
end

"""
    irfftshift_view(A, dims)

Shifts the frequencies back to the corner except for `dims[1]` because there os no negative
and positive frequency.
"""
function irfftshift_view(mat::AbstractArray{T, N}, dims=ntuple(identity, Val(N))) where {T, N}
    ShiftedArrays.circshift(mat ,.-(rft_center_diff(size(mat), dims)))
end


"""
    selectsizes(x, dism; keep_dims=true)

Select the sizes of `x` for all `dims`
If `keep_dims=true` the non-selected dimensions are
returned as 1.

# Examples
```jldoctest
julia> FourierTools.selectsizes(randn((4,3,2)), (2,3))
(1, 3, 2)

julia> FourierTools.selectsizes(randn((4,3,2)), (2,3), keep_dims=false)
(3, 2)
```

"""
function selectsizes(x::AbstractArray{T},dims::NTuple{N,Int};
                    keep_dims=true) where{T,N}
    if ~keep_dims
        return map(n->size(x,n),dims)
    end
    sz = ones(Int, ndims(x))
    for n in dims
        sz[n] = size(x,n) 
    end
    return Tuple(sz)
end 


#= # This is the setindex function that used to be in PaddedViews
# copied from commit https://github.com/JuliaArrays/PaddedViews.jl/commit/ff689b1f5d41545f3decf1f00b94c5ad7b1d5ac8
Base.@propagate_inbounds function Base.setindex!(A::PaddedView{T, N}, v, i::Vararg{Int, N}) where {T, N}
    @boundscheck begin
        # This gives some performance boost https://github.com/JuliaLang/julia/issues/33273
        _throw_argument_error() = throw(ArgumentError("PaddedViews do not support (re)setting the padding value. Consider making a copy of the array first."))
        _throw_bounds_error(A, i) = throw(BoundsError(A, i))
        if checkbounds(Bool, A, i...)
            # checkbounds(Bool, parent(A), i...) || _throw_argument_error()
            # just ignore assignments in this region
        else
            _throw_bounds_error(A, i)
        end
    end
    setindex!(parent(A), v, i...)
    return A
end
 =#
