export fftshift_view, ifftshift_view
export fftshift2d, ifftshift2d, fftshift2d_view, ifftshift2d_view


"""
    _fftshift!(dst, src, [dims])

Equivalent to `dst = fftshift(src)` but uses internally
`circshift!(dst, src, size(src) .÷ 2)`.

If `dims` is not given then the signal is shifted along each dimension.
"""
function _fftshift!(dst::AbstractArray{T, N}, src::AbstractArray{T, N},
                   dims=ntuple(i -> i, Val(N))) where {T, N}
    Δ = ntuple(i -> i ∈ dims ? size(src, i) ÷ 2 : 0, Val(N))
    circshift!(dst, src, Δ)
end

"""
    _ifftshift!(dst, src, [dims])

Equivalent to `dst = ifftshift(src)` but uses internally
`circshift!(dst, src, .- size(src) .÷ 2)`

If `dims` is not given then the signal is shifted along each dimension.
"""
function _ifftshift!(dst::AbstractArray{T, N}, src::AbstractArray{T, N},
                   dims=ntuple(i -> i, Val(N))) where {T, N}
    Δ = ntuple(i -> i ∈ dims ? - size(src, i) ÷ 2 : 0, Val(N))
    circshift!(dst, src, Δ)
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
    diff = .-(ft_center_diff(size(mat), dims))
    return ShiftedArrays.circshift(mat, diff)
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
    fftshift2d(mat::AbstractArray{T, N}) where {T, N}

Short-hand for `fftshift(mat, (1,2))`.
See `fftshift` for details.
"""
function fftshift2d(mat::AbstractArray{T, N}) where {T, N}
    fftshift(mat,(1,2))
end

"""
    ifftshift2d(mat::AbstractArray{T, N}) where {T, N}

Short-hand for `ifftshift(mat, (1,2))`.
See `ifftshift` for details.
"""
function ifftshift2d(mat::AbstractArray{T, N}) where {T, N}
    ifftshift(mat,(1,2))
end

"""
    fftshift2d_view(mat::AbstractArray{T, N}) where {T, N}

    Short-hand for `fftshift_view(mat, (1,2))`.
    See `fftshift_view` for details.
"""
function fftshift2d_view(mat::AbstractArray{T, N}) where {T, N}
    fftshift_view(mat,(1,2))
end

"""
    ifftshift2d_view(mat::AbstractArray{T, N}) where {T, N}

    Short-hand for `ifftshift_view(mat, (1,2))` performing only a 2D inverse ft.
    See `ifft` for details.
"""
function ifftshift2d_view(mat::AbstractArray{T, N}) where {T, N}
    ifftshift_view(mat,(1,2))
end

