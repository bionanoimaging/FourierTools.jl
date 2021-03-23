import Napari
using Colors, ImageShow 


nap = (x) -> Napari.napari.view_image(x)
function vv(mat; gamma=nothing, viewer=nothing)
    #if isa(mat,FFTView)
    #    start = CartesianIndex(size(mat).÷2)
    #    stop = CartesianIndex(size(mat).÷2 .+size(mat))
    #    mat = mat[start:stop]
    #end
    if isnothing(gamma)
        if  eltype(mat) <: Complex
            gamma=0.2
        else
            gamma=1.0
        end
    end
    mat = (abs.(collect(mat))).^gamma
    mat = (mat./maximum(mat))
    if isnothing(viewer)
        return Gray.(mat)
    else
        viewer(mat)
    end
end


