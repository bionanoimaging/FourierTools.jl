Random.seed!(42)

@testset "Fourier shifting methods" begin

    @testset "Empty shifts" begin
        x = randn(ComplexF32, (11, 12, 13))
        @test FourierTools.shift(x, []) == x
        
        x = randn(Float32, (11, 12, 13))
        @test FourierTools.shift(x, []) == x
    end

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
        @test FourierTools.shift!(copy(x), s) ≈ real(FourierTools.shift!(copy(xc), s))
        @test FourierTools.shift!(copy(x), s) ≈ real(FourierTools.shift!(copy(xc), 0.5))
        @test sum(x) ≈ sum(FourierTools.shift!(copy(x), s))
        
        @test sum(xc) ≈ sum(FourierTools.shift!(copy(xc), s))
    end

    @testset "Check shifts with soft_fraction" begin
        a = shift(delta((255,255)), (1.5,1.25),soft_fraction=0.1);
        @test abs(sum(a[real(a).<0])) < 3.0
        a = shift(delta((255,255)), (1.5,1.25),soft_fraction=0.0);
        @test abs(sum(a[real(a).<0])) > 5.0
    end

    @testset "Random shifts consistency between both methods" begin
        x = randn((11, 12, 13))
        s = randn((3,)) .* 10
        @test sum(x) ≈ sum(FourierTools.shift!(copy(x), s))
        @test FourierTools.shift!(copy(x), s) ≈ real(FourierTools.shift!(copy(x) .+ 0im, s))
        x = randn((11, 12, 13))
        s = randn((3,)) .* 10
        @test FourierTools.shift!(copy(x), s) ≈ real(FourierTools.shift!(copy(x) .+ 0im, s))
        @test sum(x) ≈ sum(FourierTools.shift!(copy(x), s))
    end


    @testset "Check revertibility for complex and real data" begin
        @testset "Complex data" begin
            x = randn(ComplexF32, (11, 12, 13))
            s = (-1.1, 12.123, 0.21)
            @test x ≈ shift(shift(x, s), .- s, fix_nyquist_frequency=true) 
        end
        @testset "Real data" begin
            x = randn(Float32, (11, 12, 13))
            s = (-1.1, 12.123, 0.21)
            @test x ≈ shift(shift(x, s), .- s, fix_nyquist_frequency=true) 
        end
    end

end
