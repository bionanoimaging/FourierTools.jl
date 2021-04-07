# Rotation with FFT
Via shear it is possible to rotate an image. Since we also implemented a shear algorithm, rotation can be implemented as well.
For details look at this [webpage](https://www.ocf.berkeley.edu/~fricke/projects/israel/paeth/rotation_by_shearing.html).


## Examples
For full interactivity, have a look at this [Pluto notebook](https://github.com/bionanoimaging/FourierTools.jl/tree/main/examples/rotate.jl).
```julia
using Revise, FourierTools, Plots, TestImages, PlutoUI, ImageShow

begin
    img = Float32.(testimage("fabio_512_gray"))
    z = zeros(Float32, (768, 768))
    FourierTools.center_set!(z, img)
end


Gray.(FourierTools.rotate(z, 26))
```
![](assets/rotation.png)


# Function references
```@docs
    FourierTools.rotate
    FourierTools.rotate!
```
