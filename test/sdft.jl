@testset "Sliding DFT" begin
    
    # Piecewise sinusoidal signal
    function signal(x)
        if x < 1
            5*cos(4π*x)
        elseif x < 2
            (-2x+7)*cos(2π*(x^2+1))
        else
            3*cos(10π*x)
        end
    end

    y = signal.(range(0, 3, length=61))
    n = 20
    sample_offsets = (0, 20, 40)
    dfty_sample = [fft(view(y, (1:n) .+ offset)) for offset in sample_offsets]

    # Compare SDFT
    @testset "SDFT" begin
        method = SDFT(n)
        dfty = collect(method(y))
        @testset "stateless" for i in eachindex(sample_offsets)
            @test dfty[1 + sample_offsets[i]] ≈ dfty_sample[i]
        end
        dfty = collect(method(Iterators.Stateful(y)))
        @testset "stateful" for i in eachindex(sample_offsets)
            @test dfty[1 + sample_offsets[i]] ≈ dfty_sample[i]
        end
    end

end