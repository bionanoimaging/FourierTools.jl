using FourierTools, TestImages, ImageTransformations, FileIO, Colors, ImageShow, IndexFunArrays, SpecialFunctions
using PyPlot, ImageCore
using ColorSchemes
using Interpolations

function main()
    r = rr((24, 24), scale=0.8)
    img_s = sinc.(r)#sqrt.(abs.(jinc.(r)))
    img_s ./= maximum(img_s)
    img = clamp01.(abs.(Float32.(img_s)))
    @time img_rr = clamp01.(abs.(resample(img_s, (1000, 1000))))
    @time img_interp = clamp01.(abs.(imresize(img_s, (1000, 1000), method=BSpline(Cubic(Line(OnGrid()))))))
    img1 = get.(Ref(ColorSchemes.thermal), img_s)
    img2 = get.(Ref(ColorSchemes.thermal), img_rr)
    img3 = get.(Ref(ColorSchemes.thermal), img_interp)

    #v = @view_image img_rr
    #@add_image v img_interp
    #@show typeof(img1)
    save("../../paper/figures/jinc_small.png", img1) 
    save("../../paper/figures/jinc_sinc.png", img2) 
    save("../../paper/figures/jinc_imresize.png", img3) 
    return img
end


