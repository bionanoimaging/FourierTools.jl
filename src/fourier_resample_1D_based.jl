

function resample_by_1D_FT!(arr::AbstractArray{<:Complex, N}, new_size) where N
    correction_factor = √2
    out = similar(arr, new_size)
    for d = 1:N
        ns = new_size[d]
        s = size(arr, d)
        # nothing to do in that dimension
        if ns == s
            continue
        end
        # go to fourier space
        ft!(arr, d)
        if ns > s
            center_set!(out, arr)
            # in the even case we need to fix hermitian property
            if iseven(s)
                l, r = get_indices_around_center(ns, s)
                inds_left = slice_indices(axes(out), d, l)
                out[inds_left...] .*= 0.5
                inds_right = slice_indices(axes(out), d, r+1)
                out[inds_right...] .= out[inds_left...]
            end
            arr = ift(out)
        else ns < s
            # extract a new view array with new size in that dimension
            arr_v = center_extract(arr, Base.setindex(size(arr), ns, d))
            # in the even case, we need to add the highest slice
            if iseven(ns)
                inds_left = slice_indices(axes(out), d, 1)
                inds_right = slice_indices(axes(out), d, ns+1)
                arr_v[inds_left...] .+= arr[inds_right...]
            end
            arr = ift(arr_v)
            #overwrite old arr_v handle
        end
    end
    return arr
end
