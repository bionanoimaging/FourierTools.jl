using FourierTools, Documenter 

 # set seed fixed for documentation
DocMeta.setdocmeta!(FourierTools, :DocTestSetup, :(using FourierTools); recursive=true)
makedocs(modules = [FourierTools], 
         sitename = "FourierTools.jl", 
         pages = Any[
            "FourierTools.jl" => "index.md",
            "Resampling (sinc Interpolation)" => "resampling.md",
            "Shifting with FFTs" => "shifting.md",
         ]
        )

deploydocs(repo = "github.com/bionanoimaging/FourierTools.jl.git", devbranch="main")
