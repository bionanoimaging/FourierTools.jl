@testset "Fourier Rotate" begin

    @testset "Compare with ImageTransformations" begin
        
        function f(θ)
            x = 1.0 .* range(0.0, 1.0, length=256)' .* range(0.0, 1.0, length=256)
            f(x) = sin(x * 20) + tan(1.2 * x) + sin(x) + cos(1.1323 * x) * x^3 + x^3 + 0.23 * x^4 + sin(1/(x+0.1))
            img = 5 .+ abs.(f.(x))
            img ./= maximum(img)
            img[20:40, 100:200] .= 1
            img[20:200, 20:90] .= 0.3
            img[20:200, 100:102] .= 0.7
        
            m = sum(img) / length(img)

            img_1 = parent(ImageTransformations.imrotate(img, θ; fillvalue = m))
            z = ones(Float32, size(img_1))
            z .*= m
            FourierTools.center_set!(z, img)
            img_2 = FourierTools.rotate(z, θ, pad_value=img_1[1,1])
            img_2b = FourierTools.center_extract(FourierTools.rotate(z, θ, pad_value=img_1[1,1], keep_new_size=true), size(img_2))
            img_3 = real(FourierTools.rotate(z .+ 0im, θ, pad_value=img_1[1,1]))
            img_4 = FourierTools.rotate!(z, θ)
   
            @test all(.≈(img_1, img_2, rtol=0.6))
            @test ≈(img_1, img_2, rtol=0.03)
            @test ≈(img_3, img_2, rtol=0.01)
            @test ==(img_4, z)
            @test ==(img_2, img_2b)

            img_1c = FourierTools.center_extract(img_1, (100, 100))
            img_2c = FourierTools.center_extract(img_2, (100, 100))
            @test all(.≈(img_1c, img_2c, rtol=0.3))
            @test ≈(img_1c, img_2c, rtol=0.05)
        end

        f(deg2rad(-54.31))
        f(deg2rad(-95.31))
        f(deg2rad(107.55))
        f(deg2rad(-32.31))
        f(deg2rad(32.31))
        f(deg2rad(0))
    end
end
