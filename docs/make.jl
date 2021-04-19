using FourierTools, Documenter 

 # set seed fixed for documentation
DocMeta.setdocmeta!(FourierTools, :DocTestSetup, :(using FourierTools); recursive=true)
makedocs(modules = [FourierTools], 
         sitename = "FourierTools.jl", 
         pages = Any[
            "FourierTools.jl" => "index.md",
            "FFT Helpers" => "helpers.md",
            "FFT Based Convolutions" => "convolutions.md",
            "Resampling (sinc Interpolation)" => "resampling.md",
            "Shifting with FFTs" => "shifting.md",
            "Image Shearing with FFTs" => "shear.md",
            "Image Rotation with FFTs" => "rotate.md",
            "Utility Functions" => "utils.md",
         ]
        )

deploydocs(repo = "github.com/bionanoimaging/FourierTools.jl.git", devbranch="main")
