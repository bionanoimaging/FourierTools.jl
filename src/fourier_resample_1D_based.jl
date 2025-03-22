
function resample_by_1D(arr::AbstractArray{<:Complex, N}, new_size; normalize=true) where N
    return resample_by_1D_FT!(copy(arr), new_size, normalize=normalize)
end

function resample_by_1D(arr::AbstractArray{<:Real, N}, new_size; normalize=true) where N
    return real(resample_by_1D_FT!(Complex.(arr), new_size, normalize=normalize))
end


function resample_by_1D_FT!(arr::AbstractArray{<:Complex, N}, new_size; normalize=true) where N
    correction_factor = 1 # √2
    initial_length = length(arr)
    for d = 1:N
        ns = new_size[d]
        s = size(arr, d)
        # nothing to do in that dimension
        if ns == s
            continue
        end
        # go to fourier space
        arr = ffts!(arr, d)
        if ns > s
            # out = zeros(eltype(arr), Base.setindex(size(arr), ns, d))
            out = similar(arr, Base.setindex(size(arr), ns, d)) # to work with CuArary
            out .= 0
            center_set!(out, arr)
            # in the even case we need to fix hermitian property
            if iseven(s)
                l, r = get_indices_around_center(ns, s)
                inds_left = NDTools.slice_indices(axes(out), d, l)
                out[inds_left...] .*= 0.5 * correction_factor
                inds_right = NDTools.slice_indices(axes(out), d, r+1)
                out[inds_right...] .= out[inds_left...]
            end
            arr = iffts(out, d)
        else ns < s
            # extract a new view array with new size in that dimension
            arr_v = center_extract(arr, Base.setindex(size(arr), ns, d))
            # in the even case, we need to add the highest slice
            if iseven(ns)
                l, r = get_indices_around_center(s, ns)
                inds_left = NDTools.slice_indices(axes(arr_v), d, 1)
                inds_right = NDTools.slice_indices(axes(arr), d, r+1)
                arr_v[inds_left...] .+= arr[inds_right...]
                arr_v[inds_left...] ./= correction_factor
            end
            #overwrite old arr handle
            arr = iffts(arr_v, d)
        end
    end
    # normalize that values scale accordingly
    # this violates energy!
    if normalize
        arr .*= length(arr) ./ initial_length 
    end
    return arr
end
