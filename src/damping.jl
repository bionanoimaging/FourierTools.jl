using IndexFunArrays, LinearAlgebra, NDTools
export damp_edge_outside

"""
    damp_edge_outside(img, border, mykernel, usepixels)
    extrapolates the data by filling in blurred information outside the edges. This is a bit like DampEdge but using a normalized convolution with a kernel
#Arguments:
+ img : image to extrapolate from
+ border : percentage of border pixels to add
+ mykernel : A kernel can be provided. Default: 1/r^3 kernel of support norm(border size) * sqrt(2)
+ usepixels : number of border pixels in the image to use on each edge. Default: 2. Zero means use all pixels
#Example:
```julia
julia> using TestImages, FourierTools, Colors

julia> img = permutedims(Float32.(testimage("fabio_gray_512.png")),[2 1]);

julia> img_d = damp_edge_outside(img);

julia> Gray.(log.(1 .+abs.(ft(img)))) ./ 10

julia> Gray.(log.(1 .+abs.(ft(img_d)))) ./ 10
```
"""
function damp_edge_outside(img, border=0.1, mykernel=nothing, usepixels=2)
    kernelpower=3;
    rborder=ceil.(Int,border.*size(img));
    new_size=size(img).+ rborder;
    if isnothing(mykernel)
        mykernel = (1.0./rr(new_size)) .- 1.0./norm(rborder.*sqrt(2.0));
        mykernel .= max.(mykernel, 0.0); # clip at zero
        mykernel=mykernel .^ kernelpower;
        midpos = size(mykernel) .÷2 .+ 1;
        mykernel[midpos...] = 0.0; # This does not matter as it only applies to points directly having data.
    end
    transfer=ft(mykernel);
    wimg=ones(eltype(img),size(img)); # weight image
    nimg=copy(img);
    if usepixels > 0
        inner_nimg = NDTools.select_region(nimg, new_size = size(img) .- 2 .*usepixels)
        inner_wimg = NDTools.select_region(wimg, new_size = size(img) .- 2 .*usepixels)
        inner_nimg .= 0.0
        inner_wimg .= 0.0
    end
    nimg = NDTools.select_region(nimg,new_size=new_size);
    wimg = NDTools.select_region(wimg,new_size=new_size); # just mark every pixel

    nimg2 = real(ift(ft(nimg) .* transfer)); # should this only work for 2D?
    wimg2 = real(ift(ft(wimg) .* transfer));

    nimg = nimg2 ./ wimg2;
    if usepixels > 0
        roi = NDTools.select_region(nimg, new_size=size(img))
        roi .= img; # replace the original in the middle
    end
    return nimg;

    # ToDo: Phase subpixel peak determination
    # ToDo: DampEdge (with Gaussian filter)
    # ToDo: make it work in ND 
end
