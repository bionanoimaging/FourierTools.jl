export select_region, select_region_ft, select_region_rft
export rft_fix_after, rft_fix_before, ft_fix_after, ft_fix_before


"""
    select_region_ft(mat,new_size)

performs the necessary Fourier-space operations of resampling
in the space of ft (meaning the already circshifted version of fft).
Note that in dependence of the size, fixes in Fourier-space are applied under the assumption
that the corresponding data in Real space is real-valued only. You should used this function 
if you want to select a region (extract) in Fourier space and know that you are dealing with
a real-valued signal. For complex-valued data, the ordinary select_region function should be
used instead.
`new_size`.
The size of the array view after the operation finished. The Fourier-center
is always assumed to align before and after the padding aperation.
# Examples
```jldoctest
using FFTW, FourierTools
select_region_ft(ft(rand(5,5)),(7,7))
```
"""
function select_region_ft(mat,new_size)
    old_size = size(mat)
    mat_fixed_before = ft_fix_before(mat,old_size,new_size)
    mat_pad = ft_pad(mat_fixed_before,new_size)
    # afterwards we add the highest pos. frequency to the highest lowest one 
    return ft_fix_after(mat_pad ,old_size,new_size)
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
 # Examples
```jldoctest
using FFTW, FourierTools
select_region_rft(rft(rand(5,5)),(5,5),(7,7))
```
"""
function select_region_rft(mat,old_size, new_size)
    rft_old_size = size(mat)
    rft_new_size = Base.setindex(new_size,new_size[1]÷2 +1, 1)
    return rft_fix_after(rft_pad(
        rft_fix_before(mat,old_size,new_size),
        rft_new_size),old_size,new_size)
end

"""
    select_region(mat,new_size)

performs the necessary Fourier-space operations of resampling
in the space of ft (meaning the already circshifted version of fft).

`new_size`.
The size of the array view after the operation finished. 

`center`.
Specifies the center of the new view in coordinates of the old view. By default an alignment of the Fourier-centers is assumed.
# Examples
```jldoctest
julia> using FFTW, FourierTools

julia> select_region(ones(3,3),new_size=(7,7),center=(1,3))
7×7 PaddedView(0.0, OffsetArray(::Matrix{Float64}, 4:6, 2:4), (Base.OneTo(7), Base.OneTo(7))) with eltype Float64:
 0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  1.0  1.0  1.0  0.0  0.0  0.0
 0.0  1.0  1.0  1.0  0.0  0.0  0.0
 0.0  1.0  1.0  1.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0
```
"""
function select_region(mat; new_size=size(mat), center=ft_center_diff(size(mat)).+1)
    oldcenter = ft_center_diff(new_size).+1
    PaddedView(0,mat,new_size, oldcenter .- center.+1);
end

function ft_pad(mat, new_size)
    return select_region(mat;new_size=new_size)
end

function rft_pad(mat, new_size)
    c2 = rft_center_diff(size(mat))
    c2 = Base.setindex(c2, new_size[1].÷2, 1);
    return select_region(mat;new_size=new_size, center=c2.+1)
end

function ft_fix_before(mat, size_old, size_new; start_dim=1)
    for d = start_dim:ndims(mat)
        sn = size_new[d]
        so = size_old[d]
        if sn < so && iseven(sn)
            L1 = (size_old[d] -size_new[d] )÷2 +1
            mat = FourierJoin(mat, d, L1)
        end
    end
    return mat
end

function ft_fix_after(mat,size_old,size_new; start_dim=1)
    start_dim
    ndims(mat)
    for d=start_dim:ndims(mat)
        sn = size_new[d]
        so = size_old[d]
        if sn > so && iseven(so)
            L1 = (size_new[d]-size_old[d])÷2+1
            mat = FourierSplit(mat,d,L1)
        end
        # if equal do nothing
    end
    return mat
end

function rft_fix_first_dim_before(mat,size_old,size_new;dim=1)
    sn = size_new[dim] # Note that this dim is the corresponding real-space size
    so = size_old[dim] # Note that this dim is the corresponding real-space size
    if sn < so && iseven(sn) # result size is even upon cropping
        L1 = size_new[dim] ÷ 2 + 1
        mat = FourierJoin(mat, dim, L1, L1) # a hack to dublicate the value
    end
    return mat
end

function rft_fix_first_dim_after(mat,size_old,size_new;dim=1)
    sn = size_new[dim] # Note that this dim is the corresponding real-space size
    so = size_old[dim] # Note that this dim is the corresponding real-space size
    if sn > so && iseven(so) # source size is even upon padding
        L1 = size_old[dim] ÷ 2 + 1
        mat = FourierSplit(mat,dim,L1,-1) # This hack prevents a second position to be affected
    end
    # if equal do nothing
    return mat
end

function rft_fix_before(mat,size_old,size_new)
    mat=rft_fix_first_dim_before(mat,size_old,size_new;dim=1) # ignore the first dimension
    ft_fix_before(mat,size_old,size_new;start_dim=2) # ignore the first dimension
end

function rft_fix_after(mat,size_old,size_new)
    mat = rft_fix_first_dim_after(mat,size_old,size_new;dim=1) # ignore the first dimension
    ft_fix_after(mat,size_old,size_new;start_dim=2) # ignore the first dimension
end
