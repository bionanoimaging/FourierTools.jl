export resample
export resample_by_FFT
export resample_by_RFFT
export upsample2_abs2
export upsample2
export upsample2_1D
export resample_nfft
export resample_czt
export barrel_pin


"""
    resample(arr, new_size [, normalize=true])

Calculates the `sinc` interpolation of an `arr` on a new array size
`new_size`.
It is a re-evaluation of the Fourier series at new grid points.
`new_size` can be arbitrary. Each dimension is then independently either up or downsampled.

This method is based on FFTs and therefore implicitly assumes periodic
boundaries and a finite frequency support.

`normalize=true` by default multiplies by an appropriate factor so that
the array size is included in the scaling. This results in an array having roughly
the same mean intensity.


## Basic Principle
If `size(new_size)[i] > size(arr)[i]`, we apply zero padding in Fourier space.

If `size(new_size)[i] < size(arr)[i]`, we cut out a centered part of the
Fourier spectrum.

We apply some tricks at the boundary to increase accuracy of highest frequencies. 

For real arrays we use `rfft` based operations, for complex one we use `fft` based ones.


# Examples

sinc interpolation of 2 datapoints result in an approximation of cosine.

```jldoctest
julia> resample([2.0, 0.0], (6,))
6-element Vector{Float64}:
 2.0
 1.5
 0.5
 0.0
 0.5
 1.5

julia> resample([2.0, 0.0], (6,)) ≈ 1 .+ cos.(2π .* (0:5)./6)
true
```
"""
function resample(arr::AbstractArray{T, N}, new_size; normalize=true) where {T, N}
    if new_size == size(arr)
        return copy(arr)
    end
    # for complex arrays we need a full FFT
    if T <: Complex
        arr_out = resample_by_FFT(arr, Tuple(new_size))
    else 
        arr_out = resample_by_RFFT(arr, Tuple(new_size))
    end
    # normalize that values scale accordingly
    # this violates energy!
    if normalize
        arr_out .*= length(arr_out) ./ length(arr)
    end
    return arr_out
end


"""
    resample_by_RFFT(mat, new_size)

Does a resampling based on `rfft`. This function is called by `resampling`.
"""
function resample_by_RFFT(mat, new_size)
    old_size=size(mat)
    rf = rffts(mat)
    irffts(select_region_rft(rf,old_size,new_size), new_size[1])
end


"""
    resample_by_FFT(mat, new_size)

Does a resampling based on `fft`. This function is called by `resampling`.
"""
function resample_by_FFT(mat, new_size)
    old_size = size(mat)
    # for real arrays we apply an operation so that mat_fixed_before is hermitian
    mat_fixed_before = ft_fix_before(ffts(mat),old_size,new_size)
    mat_pad = ft_pad(mat_fixed_before,new_size)
    # afterwards we add the highest pos. frequency to the highest lowest one 
    res_f = ft_fix_after(mat_pad, old_size,new_size)
    res = iffts(res_f)
    return res    
end

"""
    upsample2_1D(mat::AbstractArray{T, N}; dims=1, fix_center=false, keep_singleton=false)

Upsamples by a factor of two along dimension `dim`. 
The code is optimized for speed by using subpixelshifts rather than Fourier resizing.
By default the first pixel maintains its position. However, this leads to a shift of the center (size[d]÷2+1) in the resulting array for  uneven array sizes.
`fix_center=true` can be used to remedy this and the result array center position will agree to the source array center position.
`keep_singleton=true` will not upsample dimensions of size one.
Note that upsample2_1D is based on Fourier-shifting and you may have to deal with wrap-around problems.
"""
function upsample2_1D(mat::AbstractArray{T, N}, dim=1, fix_center=false, keep_singleton=false) where {T,N}
    if keep_singleton && size(mat,dim) ==1
        return mat
    end
    newsize = Tuple((d==dim) ? 2*size(mat,d) : size(mat,d) for d in 1:N)
    res = zeros(eltype(mat), newsize)
    if fix_center && isodd(size(mat,dim))
        selectdim(res,dim,2:2:size(res,dim)) .= mat  
        shifts = Tuple((d==dim) ? 0.5 : 0.0 for d in 1:N)
        selectdim(res,dim,1:2:size(res,dim)) .= shift(mat, shifts, take_real=true) # this is highly optimized and all fft of zero-shift directions are automatically avoided
    else
        selectdim(res,dim,1:2:size(res,dim)) .= mat  
        shifts = Tuple((d==dim) ? -0.5 : 0.0 for d in 1:N)
        selectdim(res,dim,2:2:size(res,dim)) .= shift(mat, shifts, take_real=true) # this is highly optimized and all fft of zero-shift directions are automatically avoided
    end
    return res
end

"""
    upsample2(mat::AbstractArray{T, N}; dims=1:N, fix_center=false, keep_singleton=false)

Upsamples by a factor of two in all dimensions. 
The code is optimized for speed by using subpixelshifts rather than Fourier resizing.
By default the first pixel maintains its position. However, this leads to a shift of the center (size[d]÷2+1) in the resulting array for  uneven array sizes.
`fix_center=true` can be used to remedy this and the result array center position will agree to the source array center position.
`keep_singleton=true` will not upsample dimensions of size one.
Note that upsample2 is based on Fourier-shifting and you may have to deal with wrap-around problems.
```jdoctest
julia> upsample2(collect(collect(1.0:9.0)'))
2×18 Matrix{Float64}:
 1.0  0.24123  2.0  3.24123  3.0  2.93582  4.0  5.0  5.0  5.0  6.0  7.06418  7.0  6.75877  8.0  9.75877  9.0  5.0
 1.0  0.24123  2.0  3.24123  3.0  2.93582  4.0  5.0  5.0  5.0  6.0  7.06418  7.0  6.75877  8.0  9.75877  9.0  5.0

julia> upsample2(collect(collect(1.0:9.0)'); fix_center=true, keep_singleton=true)
1×18 Matrix{Float64}:
 5.0  1.0  0.24123  2.0  3.24123  3.0  2.93582  4.0  5.0  5.0  5.0  6.0  7.06418  7.0  6.75877  8.0  9.75877  9.0
 ```
"""
function upsample2(mat::AbstractArray{T, N}; dims=1:N, fix_center=false, keep_singleton=false) where {T,N}
    res = mat
    for d in dims
        res = upsample2_1D(res,d, fix_center, keep_singleton)
    end
    return res
end

"""
    upsample2_abs2(mat::AbstractArray{T, N}; dims=1:N)

Upsamples by a factor of two and applies the abs2 operation. The code is optimized for speed.
"""
function upsample2_abs2(mat::AbstractArray{T, N}; dims=1:N) where {T,N}
    return abs2.(upsample2(mat, dims=dims))
end

"""
    resample_czt(arr, rel_zoom; shear=nothing, shear_dim=nothing, fix_nyquist=false, new_size = size(arr), rel_pad=0.2)

resamples the image with fixed factors or a list of separable functions using the chirp z transform algorithm.
The data is first padded by a relative amount `rel_pad` which is needed to avoid wrap-around problems.
As opposed to `resample()`, this routine allows for arbitrary non-integer zoom factors.
It is reasonably fast but only allows a stretch (via `rel_zoom`) and a shift (via `shear` in pixels) per line or column

Note that each entry of the tuple in `rel_zoom` or `shear` describes the zoom or shear to apply to all other dimensions individually
per entry along this dimension number. 

# Examples
```jdoctest
julia> using TestImages, NDTools, View5D

julia> a = Float32.(testimage("resolution"));

julia> b = resample_czt(a, (0.99,1.3)); # just zooming and shrinking

julia> c = resample_czt(a, (x->0.8+x^2/3,x->1.0+0.2*x)); # a more complicated distortion

julia> d = resample_czt(a, (x->1.0,x->1.0), shear=(x->50*x^2,0.0)); # a more complicated distortion

julia> @ve a,b,c,d # visualize distortions
```
"""
function resample_czt(arr::AbstractArray{T,N}, rel_zoom; shear=nothing, shear_dim=nothing, fix_nyquist=false, new_size = size(arr), rel_pad=0.2, do_damp=false, center=CtrMid) where {T,N}
    RT = real(T)
    orig_size = size(arr)
    if do_damp
        arr = damp_edge_outside(arr, rel_pad)
    else
        arr = copy(arr)
    end
    for d in 1:length(rel_zoom)
        sd = mod(d,ndims(arr))+1
        if !isnothing(shear_dim)
            sd = shear_dim[d]
        end
        my_zoom = 1.0
        # case of a list of zoom numbers
        if (isa(rel_zoom, Tuple) && isa(rel_zoom[1], Number)) || isa(rel_zoom, Number)
            my_zoom = rel_zoom[d]
            f_res = ift(arr, d)
            myshear = 0
            if !isnothing(shear)
                myshear =  shear[d]
            end
            if !iszero(myshear)
                shifts = ramp(real(eltype(f_res)),d, size(f_res,d), scale=ScaFT)
                FourierTools.apply_shift_strength!(f_res, f_res, shifts, d, sd, -myshear, fix_nyquist)
            end
            if T<:Real
                f_res = real(FourierTools.czt_1d(f_res, my_zoom, d))
            else
                f_res = T.(FourierTools.czt_1d(f_res, my_zoom, d))
            end
            select_region!(f_res, arr) 
        # case of position dependent zoom functions
        elseif (isa(rel_zoom, Tuple) && isa(rel_zoom[1], Function))
            pos = 0
            p = 1
            shifts = ramp(real(eltype(arr)),d, size(arr,d), scale=ScaFT)
            # note that the slicing removes the dimension opposed to the version above.
            for slice in eachslice(arr, dims=d)
                myshear = let 
                    if !isnothing(shear)
                        if isa(shear[sd], Function)
                            # position-dependent shear
                            shear[sd](pos) 
                        else
                            shear[sd]
                        end
                    else
                        zero(RT)
                    end
                end
                my_zoom = rel_zoom[d](pos)
                # one-d since a slice was selected
                f_res = ift(slice,1) 
                if !iszero(myshear)
                    FourierTools.apply_shift_strength!(f_res, f_res, shifts[p], 1, 1, -myshear, fix_nyquist)                
                end
                f_res = let 
                    if T<:Real
                        real(FourierTools.czt_1d(f_res, my_zoom, 1))
                    else
                        FourierTools.czt_1d(f_res, my_zoom, 1)
                    end
                end
                select_region!(f_res, slice)
                pos += one(eltype(orig_size))/orig_size[d]
                p += 1
            end
        else
            error("expected list of numbers or list of functions as argument `rel_zoom`")
        end
    end
    res = select_region(arr, new_size=new_size)
    return res
end

"""
    barrel_pin(arr, rel=0.5)
emulates a barrel (`rel>0`) or a pincushion (`rel<0`) distortion. The distortions are calculated using `resample_czt()` with separable quadratic zooms.

See also: `resample_czt()`
# Examples
```jdoctest
julia> using TestImages, NDTools, View5D

julia> a = Float32.(testimage("resolution"))

julia> b = barrel_pin(a,0.5) # strong barrel distortion

julia> c = barrel_pin(a,-0.5) # strong pin-cushion distortion

julia> @ve a,b,c # visualize distortions
```
"""
function barrel_pin(arr::AbstractArray{T,N}, rel=0.5) where {T,N}
    RT = real(T)
    fk = x -> one(RT) + RT(rel) .* (x-RT(0.5))^2
    fkts = ntuple(n -> fk, ndims(arr))
    return resample_czt(arr, fkts)
end

"""
    resample_nfft(img, new_pos, dst_size=nothing; pixel_coords=false, is_local_shift=false, is_src_coords=true, reltol=1e-9)
    
resamples an ND-array to a set of new positions `new_pos` measured in either in pixels (`pixel_coords=true`) or relative (Fourier-) image coordinates (`pixel_coords=false`).
`new_pos` can be 
+ an array of `Tuples` specifying the zoom along each direction
+ an `N+1` dimensional array (for `N`-dimensional imput data `img`) of destination postions, the last dimension enumerating the respective destination corrdinate dimension.
+ a function accepting a coordinate `Tuple` and yielding a destination position `Tuple`.

`resample_nfft` can perform a large range of possible resamplings. Note that the default setting is `is_src_coords=true` which means that the source coordinates of each destination
position have to be specified. This has the advantage that the result has usually less artefacts, but the positions may be more less convenient to specify.

# Arguements
+ `img`: the image to apply resampling to
+ `new_pos``: specifies the resampling. See description above.
+ `dst_size`: this argument optionally defines the output size. If you require a different result size for `new_pos` being a function or with `is_src_coords=true`, state it here. By defaul (`dst_size=nothing`) the 
              destination size will be inferred form the argument `new_pos` or assumed to be `size(img)`.
+ `is_local_shift`: specifies, whether the resampling coordinates refer to a relative shift or absoluter coordinates
+ `is_in_pixels`: specifies whether the coordinates (or relative distances) are given in pixel pitch units (`is_in_pixels=true`) or in units relative to the array sizes (Fourier convention) 
+ `is_src_coords`: specifies, whether the resampling positions refer to sampling at source (`is_src_coords=true`) or destination coordinates 
+ `reltol`: will be used as an argument to the `nfft` function spedifying the relative precision to calculate to

See also: `resample`, `resample_czt`
# Examples
```julia-repl
julia> using FourierTools, TestImages, NDTools, View5D, IndexFunArrays

julia> a = Float32.(testimage("resolution"));

julia> b = resample_nfft(a, t -> (2.5f0 *sign(t[1])*t[1]^2, t[2]*(0.5f0+t[1]))); # a complicated deformation

julia> sz = size(a);

# stacking only the displacement along the last dimension:
julia> new_pos = cat(xx(sz,scale=ScaFT), zeros(sz), dims=3);

julia> c = resample_nfft(a, new_pos, is_local_shift=true); # stretch along x using an array

julia> new_pos = cat(.-xx(sz,scale=ScaFT)./2, zeros(sz), dims=3);

julia> c2 = resample_nfft(a, new_pos, is_local_shift=true, is_src_coords=false); # stretch along x using an array

# Notice the difference in brightness between c and c2
julia> @ve a b c c2 # visualize distortion and x-shrinks. 

# Lets try a 2D rotation:
# define a rotation operation
julia> rot_alpha(a, t) = (cosd(a)*t[1] + sind(a)*t[2], -sind(a)*t[1]+cosd(a)*t[2])

# postions as an array of tuples
julia> new_pos = rot_alpha.(10.0, idx(a, scale=ScaFT))

# lets do the resampling, this time by specifying the destination coordinates:
julia> d = resample_nfft(a, new_pos, is_src_coords=false);

#display the result
julia> @ve a d

#how about a spiral deformation?
julia> new_pos = rot_alpha.(rr(a), idx(a, scale=ScaFT))

julia> e = resample_nfft(a, new_pos);

julia> f = resample_nfft(a, new_pos, is_src_coords=false);

# observe the artefacts generated by undersampling in the destination grid
julia> @ve a e f
```
"""
function resample_nfft(img::AbstractArray{T,D}, new_pos::AbstractArray{T2,D2}, dst_size=nothing; pad_value=nothing, is_in_pixels=false, is_local_shift=false, is_src_coords=true, reltol=1e-9)::AbstractArray{T,D} where {T,D,T2,D2} # 

    Fimg = let
        if is_src_coords
            p = plan_nfft_nd(img, new_pos, dst_size; pad_value=pad_value, is_in_pixels=is_in_pixels, is_local_shift=is_local_shift, is_adjoint=false, reltol=reltol)
            p * ift(img)
        else
            p = plan_nfft_nd(img, new_pos, dst_size; pad_value=pad_value, is_in_pixels=is_in_pixels, is_local_shift=is_local_shift, is_adjoint=true, reltol=reltol)
            ft(p * img) ./ prod(size(img))
        end
    end

    if T<:Real
        img = real.(Fimg)
    else
        img = Fimg
    end
    return img
end

function resample_nfft(img::AbstractArray{T,D}, new_pos_fkt::Function, dst_size=nothing; pad_value=nothing,  is_in_pixels=false, is_local_shift=false, is_src_coords=true, reltol=1e-9)::AbstractArray{T,D} where {T,D} # 
    Fimg = let
        if is_src_coords
            p = plan_nfft_nd(img, new_pos_fkt, dst_size; pad_value=pad_value, is_in_pixels=is_in_pixels, is_local_shift=is_local_shift, is_adjoint=false, reltol=reltol)
            p * ift(img)
        else
            p = plan_nfft_nd(img, new_pos_fkt, dst_size; pad_value=pad_value, is_in_pixels=is_in_pixels, is_local_shift=is_local_shift, is_adjoint=true, reltol=reltol)
            ft(p * img) ./ prod(size(img))
        end
    end

    if T<:Real
        img = real.(Fimg)
    else
        img = Fimg
    end
    return img
end
