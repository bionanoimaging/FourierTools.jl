Random.seed!(42)

@testset "Fourier shifting methods" begin

    @testset "Integer shifts for complex and real arrays" begin
        x = randn(ComplexF32, (11, 12, 13))

        s = (2,2,2)
        @test FourierTools.shift(x, s) ≈ circshift(x, s) 
        s = (3,2,1)
        @test FourierTools.shift(x, s) ≈ circshift(x, s) 

        @test FourierTools.shift(x, (0,0,0)) == x
        x = randn(Float32, (11, 12, 13))

        s = (2,2,2)
        @test FourierTools.shift!(copy(x), s) ≈ circshift(x, s) 
        s = (3,2,1)
        @test FourierTools.shift!(copy(x), s) ≈ circshift(x, s) 
        
        @test sum(x) ≈ sum(FourierTools.shift!(copy(x), s))
    end

    @testset "Half integer shifts" begin

        x = [0.0, 1.0, 0.0, 1.0]
        xc = ComplexF32.(x)
        
        s = [0.5]
        @test FourierTools.shift!(copy(x), s) ≈ FourierTools.shift!(copy(xc), s)
        @test FourierTools.shift!(copy(x), s) ≈ FourierTools.shift!(copy(xc), 0.5)
        @test sum(x) ≈ sum(FourierTools.shift!(copy(x), s))
        
        xc = [0.0, 1im, 0.0, 1im]
        @test FourierTools.shift!(copy(xc), s) ≈ 1im .* 0.5 .* ones(4)
        @test sum(xc) ≈ sum(FourierTools.shift!(copy(xc), s))
    end


    @testset "Random shifts consistency between both methods" begin
        x = randn((11, 12, 13))
        s = randn((3,)) .* 10
        @test sum(x) ≈ sum(FourierTools.shift!(copy(x), s))
        @test FourierTools.shift!(copy(x), s) ≈ FourierTools.shift!(copy(x) .+ 0im, s)
        x = randn((11, 12, 13))
        s = randn((3,)) .* 10
        @test FourierTools.shift!(copy(x), s) ≈ FourierTools.shift!(copy(x) .+ 0im, s)
        @test sum(x) ≈ sum(FourierTools.shift!(copy(x), s))
    end

end
