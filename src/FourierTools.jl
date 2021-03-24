module FourierTools


using PaddedViews, ShiftedArrays
using FFTW

export ft,ift, rft, irft
export extract, extract_ft, extract_rft

include("utils.jl")
include("resampling.jl")
include("custom_fourier_types.jl")

function extract(mat; new_size=size(mat), center=ft_center_0(mat).+1)
    oldcenter = ft_center_0(new_size).+1
    PaddedView(0,mat,new_size, oldcenter .- center.+1);
end

function ft_pad(mat, new_size)
    return extract(mat;new_size=new_size)
end

function rft_pad(mat, new_size)
    c2 = rft_center_0(mat)
    c2 = replace_dim(c2,1,new_size[1].÷2);
    return extract(mat;new_size=new_size, center=c2.+1)
end

function ft_fix_before(mat, size_old, size_new; start_dim=1)
    for d = start_dim:ndims(mat)
        sn = size_new[d]
        so = size_old[d]
        if sn < so && iseven(sn)
            L1 = (size_old[d] -size_new[d] )÷2 +1
            mat = FourierSum(mat, d, L1)
        end
    end
    return mat
end

function ft_fix_after(mat,size_old,size_new; start_dim=1)
    for d=start_dim:ndims(mat)
        sn = size_new[d]
        so = size_old[d]
        if sn > so && iseven(so)
            L1 = (size_new[d]-size_old[d])÷2+1
            mat = FourierDuplicate(mat,d,L1)
        end
        # if equal do nothing
    end
    return mat
end


function rft_fix_before(mat,size_old,size_new)
    ft_fix_before(mat,size_old,size_new;start_dim=2) # ignore the first dimension
end



function rft_fix_after(mat,size_old,size_new)
    ft_fix_after(mat,size_old,size_new;start_dim=2) # ignore the first dimension
end

"""
performs the necessary Fourier-space operations of resampling
in the space of ft (meaning the already circshifted version of fft).
"""
function extract_ft(mat,new_size)
    old_size = size(mat)
    mat_fixed_before = ft_fix_before(mat,old_size,new_size)
    mat_pad = ft_pad(mat_fixed_before,new_size)
    # afterwards we add the highest pos. frequency to the highest lowest one 
    return ft_fix_after(mat_pad ,old_size,new_size)
end




"""
performs the necessary Fourier-space operations of resampling
in the space of rft (meaning the already circshifted version of rfft).
"""
function extract_rft(mat,new_size)
    rft_old_size = size(mat)
    rft_new_size = replace_dim(new_size,1,new_size[1]÷2 +1)
    return rft_fix_after(rft_pad(
        rft_fix_before(mat,rft_old_size,rft_new_size),
        rft_new_size),rft_old_size,rft_new_size)
end



end # module
