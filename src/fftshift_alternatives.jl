export fftshift_view, ifftshift_view

# add constructor here, might be merged to ShiftedArrays
# this prevents that CircShiftedArrays get nested with twice application
# https://github.com/JuliaArrays/ShiftedArrays.jl/pull/44
function ShiftedArrays.CircShiftedArray(csa::CircShiftedArray{T, N, <:AbstractArray},
                                        n = Tuple(0 for i in 1:N)) where {T, N, }
    CircShiftedArray(parent(csa), n .+ csa.shifts)
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
