Random.seed!(42)

@testset "Fourier moving methods" begin

    @testset "Integer shifts for complex and real arrays" begin
        x = randn(ComplexF32, (11, 12, 13))

        s = (2,2,2)
        @test FourierTools.shift!(copy(x), s) ≈ circshift(x, s) 
        s = (3,2,1)
        @test FourierTools.shift!(copy(x), s) ≈ circshift(x, s) 

        x = randn(Float32, (11, 12, 13))

        s = (2,2,2)
        @test FourierTools.shift!(copy(x), s) ≈ circshift(x, s) 
        s = (3,2,1)
        @test FourierTools.shift!(copy(x), s) ≈ circshift(x, s) 
    end

    @testset "Half integer shifts" begin

        x = [0.0, 1.0, 0.0, 1.0]
        xc = ComplexF32.(x)
        
        s = [0.5]
        @test FourierTools.shift!(copy(x), s) ≈ FourierTools.shift!(copy(xc), s)
        @test FourierTools.shift!(copy(x), s) ≈ 0.5 .* ones(4)
    end


    @testset "Random shifts consistency between both methods" begin
        x = randn((11, 12, 13))
        s = randn((3,)) .* 10
        @test FourierTools.shift!(copy(x), s) ≈ FourierTools.shift!(copy(x) .+ 0im, s)
        x = randn((11, 12, 13))
        s = randn((3,)) .* 10
        @test FourierTools.shift!(copy(x), s) ≈ FourierTools.shift!(copy(x) .+ 0im, s)
    end

end
