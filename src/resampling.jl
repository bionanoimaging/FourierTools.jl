export resample
export resample_by_FFT
export resample_by_RFFT
export upsample2_abs2
export upsample2
export upsample2_1D
export resample_var
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
function resample_by_RFFT(mat, new_size) where {T}
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

function upsample2_1D(mat::AbstractArray{T, N}, dim=1, fix_center=false) where {T,N}
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
    upsample2(mat; dims=1:N)

Upsamples by a factor of two in all dimensions. 
The code is optimized for speed by using subpixelshifts rather than Fourier resizing.
"""
function upsample2(mat::AbstractArray{T, N}; dims=1:N, fix_center=false) where {T,N}
    res = mat
    for d in dims
        res = upsample2_1D(res,d, fix_center)
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
                FourierTools.apply_shift_strength!(f_res, f_res, shifts, d, sd,myshear, fix_nyquist)
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
                    FourierTools.apply_shift_strength!(f_res, f_res, shifts[p], 1, 1, myshear, fix_nyquist)                
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
    resample_var(img, rel_shift, dim=(2,))
    
Applies a variable shift `rel_shift` measured in pixel to each pixel in the source `img`.
`rel_shift` can be 
+ a collection (e.g. a tuple) of number specifying the zoom along each direction
+ a collection (e.g. a tuple) of vectors (or tuples) each with the same length as the dimensions
+ a collection of functions projecting the ScaFT (approximately -0.5 .. 0.5) range to the local shift in pixels to apply.

`resample_var` can perform a large range of possible resamplings. 
However care has to be taken when choosing the functions to apply.
Each function in the tuple `rel_shift` corresponds to the direction to apply resampling to 
whereas the (cyclicly) next dimension is the onethat is iterated over to extract slices to resample. 
Since these operations are looped over all tuple entries, the x positions are already changed when t
he resampling is performed over y. This possibly needs to be accounted for. See 2D rotation example below.
The supplied functions need two arguments, where the first argument corresponds to the current shift 
direction and the second argument to the orthogonal direction over which the slices are extracted. 
In 2D this means that `rel_shift` should typically look like this: (x´(x,y),y(y,x´)).
See the rotation example below.

See also: `resample`, `resample_czt`
# Examples
```julia-repl
julia> using TestImages, NDTools, View5D, IndexFunArrays

julia> a = Float32.(testimage("resolution"))

julia> b = resample_var(a, ((x,y)-> 50 *x^2,))

julia> c = resample_var(a, (yy(size(a)).*xx(size(a),scale=ScaFT).^3,), 1e-2); # stretch along x using an array

julia> @ve a,b,c # visualize distortions

# Lets try a 2D rotation:
julia> img = Float64.(testimage("resolution"));

julia> fx(x,y,α) = x*cos(α)+y*sin(α)-x;

# the function below needs some care: We need to change the order of the arguments and first compensate the x-shift that was previously applied 
julia> fy(y,xp,α) = y*cos(α)-tan(α)*(xp-sin(α)*y)-y;

# angle by which to rotate
julia> α = 15*pi/180;

# lets do the resampling
julia> b = resample_var(img, ((x,y)->size(img,1)*fx(x,y,α),(x,y)->size(img,2)*fy(x,y,α)));

#display the result
julia> @ve img, b
```
"""
function resample_var(img::AbstractArray{T,D} , rel_shift, myeps=eps(T))::AbstractArray{T,D} where {T,D} # 
    dims = length(rel_shift)
    RT = real(T)
    myeps = RT(myeps)
    for d in 1:dims
        N = size(img,d)
        k = (zero(RT):N-one(RT))
        k = k .- k[size(k,1)÷2+1]
        x0 = k ./ size(k,1)
        Fimg = ft(img,(d,)) # nufft3(Complex.(img), x0,k, eps(eltype(img)))
        s_dim = mod(d,ndims(img))+1
        n = 0
        for s in eachslice(Fimg, dims=s_dim)
            rs = rel_shift[d]
            x = x0
            if isa(rs, Function)
                y0 = RT((n - (size(img,s_dim)÷2))/size(img,s_dim))
                rs = RT.(rs.(x0,y0)./N)
                x = x0 .- rs
            elseif isa(rs, Number)
                x = x0 ./ RT.(rs)
            else
                # apply the local shifts of this line
                x = x0 .- RT.(rs[:,n+1]./N)
            end
            # real space phase change to apply to emulate the centering of the Fourier coordinates
            x = .-x
            # the exponential below is needed to correct for the Fourier-space not being properly centered in nufft2.
            # This cannot be remedied by fftshift due to the non-uniform sampling!
            s .= cispi.(2*(N÷2) .*x).*nufft2(s, x, myeps)./N  # is faster that nufft3
            n += 1
        end
        if T<:Real
            # return Fimg
            img = real.(Fimg)
        else
            img = Fimg
        end
    end
    return img
end
