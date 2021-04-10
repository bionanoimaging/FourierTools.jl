export fftshift_view, ifftshift_view

# add constructor here, might be merged to ShiftedArrays
# this prevents that CircShiftedArrays get nested with twice application
# https://github.com/JuliaArrays/ShiftedArrays.jl/pull/44
function ShiftedArrays.CircShiftedArray(csa::CircShiftedArray, n = Tuple(0 for i in 1:N))
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
    fftshiftshift_view(A [, dims])

In addition to the fftshift, this version also includes the phase modification to account
for centering the zero coordinate system in real space befor the fft. 
"""
function fftshiftshift_view(mat::AbstractArray{T, N}, dims=ntuple(identity, Val(N))) where {T, N}
    ex = exp_ikx(mat, shift_by=.- ft_center_diff(size(mat), dims))
    sa = ShiftedArrays.circshift(mat, ft_center_diff(size(mat), dims))
    ex .* sa
    # LazyArray( @~ ex .* sa)  # does not work for 1D arrays
end

"""
    ifftshift_view(A [, dims])

Result is semantically equivalent to `fftshift(A, dims)` but returns 
a view instead. 
"""
function ifftshift_view(mat::AbstractArray{T, N}, dims=ntuple(identity, Val(N))) where {T, N}
    diff = .-(ft_center_diff(size(mat), dims))
    x = ShiftedArrays.circshift(mat, diff)
    return x
end

function ifftshift_view(mat::CircShiftedArray{T, N, AA}, dims=ntuple(identity, Val(N))) where {T, N, AA}
    diff = .-(ft_center_diff(size(mat), dims))
    return ShiftedArrays.circshift(mat, diff)
end


"""
    ifftshiftshift_view(A [, dims])

In addition to the ifftshift, this version also includes the phase modification to account
for centering the zero coordinate system in real space after the ifft. 
"""
function ifftshiftshift_view(mat::AbstractArray{T, N}, dims=ntuple(identity, Val(N))) where {T, N}
    ShiftedArrays.circshift(mat .* exp_ikx(mat, shift_by=ft_center_diff(size(mat),dims)), .-(ft_center_diff(size(mat), dims)))
end


"""
    rfftshift_view(A, dims)

Shifts the frequencies to the center expect for `dims[1]` because there os no negative
and positive frequency.
"""
function rfftshift_view(mat::AbstractArray{T, N}, dims=ntuple(identity, Val(N))) where {T, N}
    ShiftedArrays.circshift(mat, rft_center_diff(size(mat), dims))
end

get_RFT_scale(real_size) = 0.5 ./ (max.(real_size ./ 2, 1))  # The same as the FFT scale but for the full array in real space!

"""
    rfftshiftshift_view(A,real_size, dims)

Shifts the frequencies to the center expect for `dims[1]` because there os no negative
and positive frequency. This version also accounts for centering the real space coordinate system.
"""
function rfftshiftshift_view(mat::AbstractArray{T, N}, real_size, dims=ntuple(identity, Val(N))) where {T, N}
    # exp_ikx(mat,shift_by= .-rft_center_diff(size(mat), dims)).*ShiftedArrays.circshift(mat, rft_center_diff(size(mat), dims))
    ex = exp_ikx(mat, shift_by=.- ft_center_diff(real_size, dims), scale=get_RFT_scale(real_size), offset=CtrRFT)
    sa = ShiftedArrays.circshift(mat, rft_center_diff(size(mat), dims))
    LazyArray( @~ ex .* sa)
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
    irfftshiftshift_view(A, real_size, dims)

Shifts the frequencies back to the corner except for `dims[1]` because there os no negative
and positive frequency. This version also accounts for centering the real space coordinate system.
"""
function irfftshiftshift_view(mat::AbstractArray{T, N}, real_size, dims=ntuple(identity, Val(N))) where {T, N}
    ex = exp_ikx(mat, shift_by=ft_center_diff(real_size,dims), scale=get_RFT_scale(real_size), offset=CtrRFT)
    la = LazyArray( @~ mat .* ex)
    ShiftedArrays.circshift(la, .-(rft_center_diff(size(mat), dims)))    
end

