export select_region_ft, select_region_rft
export rft_fix_after, rft_fix_before, ft_fix_after, ft_fix_before

"""
    select_region_ft(mat,new_size)

Performs the necessary Fourier-space operations of resampling
in the space of ft (meaning the already circshifted version of fft).

Note that in dependence of the size, fixes in Fourier-space are applied under the assumption
that the corresponding data in Real space is real-valued only. You should used this function 
if you want to select a region (extract) in Fourier space and know that you are dealing with
a real-valued signal. For complex-valued data, the ordinary `select_region` function should be
used instead.
`new_size`.

The size of the array view after the operation finished. The Fourier-center
is always assumed to align before and after the padding operation.

 # Examples
```jldoctest
julia> x = [1 20 3; 4 500 6; -7 821 923]
3×3 Matrix{Int64}:
  1   20    3
  4  500    6
 -7  821  923

julia> ffts(x)
3×3 MutableShiftedArrays.CircShiftedArray{ComplexF64, 2, Matrix{ComplexF64}}:
   106.5+390.577im  -1099.5-1062.61im   1000.5+700.615im
 -1138.5+354.204im   2271.0+0.0im      -1138.5-354.204im
  1000.5-700.615im  -1099.5+1062.61im    106.5-390.577im

julia> select_region_ft(ffts(x), (4,4))
4×4 PaddedView(0.0 + 0.0im, OffsetArray(::MutableShiftedArrays.CircShiftedArray{ComplexF64, 2, Matrix{ComplexF64}}, 2:4, 2:4), (Base.OneTo(4), Base.OneTo(4))) with eltype ComplexF64:
 0.0+0.0im      0.0+0.0im          0.0+0.0im          0.0+0.0im
 0.0+0.0im    106.5+390.577im  -1099.5-1062.61im   1000.5+700.615im
 0.0+0.0im  -1138.5+354.204im   2271.0+0.0im      -1138.5-354.204im
 0.0+0.0im   1000.5-700.615im  -1099.5+1062.61im    106.5-390.577im

julia> x = [1 20; 4 500; -7 821; -2 2]
4×2 Matrix{Int64}:
  1   20
  4  500
 -7  821
 -2    2

julia> ffts(x)
4×2 MutableShiftedArrays.CircShiftedArray{ComplexF64, 2, Matrix{ComplexF64}}:
  -347.0+0.0im     331.0+0.0im
   809.0-492.0im  -793.0+504.0im
 -1347.0+0.0im    1339.0+0.0im
   809.0+492.0im  -793.0-504.0im

julia> select_region_ft(ffts(x), (5,3))
5×3 FourierTools.FourierSplit{ComplexF64, 2, FourierTools.FourierSplit{ComplexF64, 2, PaddedViews.PaddedView{ComplexF64, 2, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, OffsetArrays.OffsetMatrix{ComplexF64, MutableShiftedArrays.CircShiftedArray{ComplexF64, 2, Matrix{ComplexF64}}}}}}:
 -86.75+0.0im     165.5+0.0im    -86.75+0.0im
  404.5-246.0im  -793.0+504.0im   404.5-246.0im
 -673.5+0.0im    1339.0+0.0im    -673.5+0.0im
  404.5+246.0im  -793.0-504.0im   404.5+246.0im
 -86.75+0.0im     165.5+0.0im    -86.75+0.0im
```
"""
function select_region_ft(mat, new_size)
    old_size = size(mat)
    mat_fixed_before = ft_fix_before(mat, old_size, new_size)
    mat_pad = ft_pad(mat_fixed_before, new_size)
    # afterwards we add the highest pos. frequency to the highest lowest one 
    return ft_fix_after(mat_pad ,old_size, new_size)
end

"""
    select_region_rft(mat,old_size, new_size)

performs the necessary Fourier-space operations of resampling
in the space of rft (meaning the already circshifted version of rfft).
Note that in dependence of the size, fixes in Fourier-space are applied, which is
why you should used this function if you want to select a region (extract) in rft space.
Since rfts always assume the corresponding Real-space data to be real, this rule should
have no exception for rft data.

`old_size`.
The size of the corresponding real-space array before it was rfted. 
    
`new_size`.
The size of the corresponding real-space array view after the operation finished. The Fourier-center
is always assumed to align before and after the padding aperation.
"""
function select_region_rft(mat, old_size, new_size)
    rft_new_size = Base.setindex(new_size,new_size[1] ÷ 2 + 1, 1)
    return rft_fix_after(rft_pad(
        rft_fix_before(mat, old_size, new_size), rft_new_size), old_size, new_size)
end

function ft_pad(mat, new_size)
    return select_region(optional_collect(mat); new_size = new_size)
end

function rft_pad(mat, new_size)
    c2 = rft_center_diff(size(mat))
    c2 = Base.setindex(c2, new_size[1] .÷ 2, 1);
    return select_region(optional_collect(mat); new_size=new_size, center = c2 .+ 1)
end

"""
    ft_fix_before(mat::MT, size_old, size_new, ::Val{N})::FourierJoin{T,N,MT}  where {T,N, MT<:AbstractArray{T,N}} 

implements the specialized (highest dimension) version of a recursive dimension-specific function that returns an array type which knows
how to access (joins) certain elements.
"""
function ft_fix_before(mat::MT, size_old, size_new, ::Val{N})::FourierJoin{T,N,MT}  where {T,N, MT<:AbstractArray{T,N}} 
    sn = size_new[N]
    so = size_old[N]
    do_join = (sn < so && iseven(sn))
    L1 = (size_old[N] - size_new[N] ) ÷ 2 + 1
    return FourierJoin(mat, Val(N), L1, do_join)
end

"""
    ft_fix_before(mat::MT, size_old, size_new, ::Val{D}=Val(1))  where {D, T, N, MT<:AbstractArray{T,N}} 

implements the general version of a recursive dimension-specific function that returns an array type which knows
how to access (joins) certain elements.
"""
function ft_fix_before(mat::MT, size_old, size_new, ::Val{D}=Val(1))  where {D, T, N, MT<:AbstractArray{T,N}} 
    if D <= N
        sn = size_new[D]
        so = size_old[D]
        do_join = (sn < so && iseven(sn))
        L1 = (size_old[D] - size_new[D] )÷2 +1
        mat = FourierJoin(mat, Val(D), L1, do_join)
        return ft_fix_before(mat, size_old, size_new, Val(D + 1))
    else
        L1 = (size_old[N] -size_new[N] )÷2 +1
        return FourierJoin(mat, Val(N), L1, false)
    end
end

# routine only for the last dimensions N == D
function ft_fix_after(mat::MT, size_old, size_new, ::Val{N})::FourierSplit{T,N,MT,N}   where {T, N, MT<:AbstractArray{T,N}}
    sn = size_new[N]
    so = size_old[N]
    do_split = (sn > so && iseven(so))
    L1 = (size_new[N] - size_old[N]) ÷ 2 + 1
    return FourierSplit(mat, Val(N), L1, do_split)
end

function ft_fix_after(mat::MT, size_old, size_new, ::Val{D}=Val(1)) where {D, T, N, MT<:AbstractArray{T,N}}
    if D <= N
        sn = size_new[D]
        so = size_old[D]
        do_split = (sn > so && iseven(so))
        L1 = (size_new[D]-size_old[D])÷2+1
        mat = FourierSplit(mat, Val(D), L1, do_split)
        return ft_fix_after(mat, size_old, size_new, Val(D + 1))
    else
        L1 = (size_new[N]-size_old[N])÷2+1
        return FourierSplit(mat, Val(N), L1, false)
    end
end

function rft_fix_first_dim_before(mat, size_old, size_new; dim::Val{D}=Val(1)) where {D}
    # Note that this dim is the corresponding real-space size
    sn = size_new[D] 
    so = size_old[D]
    # result size is even upon cropping
    do_join = (sn < so && iseven(sn))
    L1 = size_new[D] ÷ 2 + 1
    # a hack to dublicate the value
    mat = FourierJoin(mat, Val(D), L1, L1, do_join)
    return mat
end

function rft_fix_first_dim_after(mat,size_old,size_new; dim::Val{D}=Val(1)) where {D}
    # Note that this dim is the corresponding real-space size
    sn = size_new[D] 
    so = size_old[D] 
    # source size is even upon padding
    do_split = (sn > so && iseven(so)) 
    L1 = size_old[D] ÷ 2 + 1
    # This hack prevents a second position to be affected
    mat = FourierSplit(mat, Val(D), L1, -1, do_split)
    # if equal do nothing
    return mat
end

function rft_fix_before(mat,size_old,size_new)
    # ignore the first dimension
    mat=rft_fix_first_dim_before(mat,size_old,size_new; dim=Val(1)) 
    # ignore the first dimension since it starts at Val(2)
    ft_fix_before(mat, size_old, size_new, Val(2)) 
end

function rft_fix_after(mat, size_old, size_new)
    # ignore the first dimension
    mat = rft_fix_first_dim_after(mat, size_old, size_new; dim=Val(1)) 
    # ignore the first dimension since it starts at Val(2)
    ft_fix_after(mat, size_old, size_new, Val(2))
end
