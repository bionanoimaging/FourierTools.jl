# doesn't save too much but a little
@precompile_setup begin
    img64 = abs.(randn(Float32, (4,4)))
    img32 = abs.(randn(Float64, (4,4)))
    imgCF64 = abs.(randn(ComplexF64, (4,4)))
    imgCF32 = abs.(randn(ComplexF32, (4,4)))

    function f(img)
        resample(img, (5,5))
        resample(img, (2,2))
        shift(img, (2,2))
        shift(img, (2,2))
        ft(img)
        ffts(img)
        ift(img)
        iffts(img)
        rotate(img, 1.1)
        conv(img, img)
    end

    @precompile_all_calls begin
        f(img64)
        f(img32)
        f(imgCF64)
        f(imgCF32)
    end

end
