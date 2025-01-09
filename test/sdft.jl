import FourierTools:
    sdft_windowlength,
    sdft_update!,
    sdft_previousdft,
    sdft_previousdata,
    sdft_nextdata,
    sdft_iteration,
    sdft_backindices,
    sdft_dataoffsets

# Dummy method to test more complex designs
struct TestSDFT{T,C} <: AbstractSDFT
    n::T
    factor::C
end
TestSDFT(n) = TestSDFT(n, exp(2π*im/n))
sdft_windowlength(method::TestSDFT) = method.n
sdft_backindices(::TestSDFT) = [0, 2]
sdft_dataoffsets(::TestSDFT) = [0, 1]

function sdft_update!(dft, x, method::TestSDFT{T,C}, state) where {T,C}
    twiddle = one(C)
    dft0 = sdft_previousdft(state, 0)
    unused_dft = sdft_previousdft(state, 2) # not used - add for coverage
    unused_data = sdft_previousdata(state, 1) # not used - add for coverage
    unused_count = sdft_iteration(state) # not used - add for coverage
    for k in eachindex(dft)
        dft[k] = twiddle * (dft0[k] + sdft_nextdata(state) - sdft_previousdata(state)) +
            0.0 * (unused_dft[k] + unused_data + unused_count)
        twiddle *= method.factor
    end
end

# Dummy method to test exceptions
struct ErrorSDFT  <: AbstractSDFT end
sdft_windowlength(method::ErrorSDFT) = 2
function sdft_update!(dft, x, ::ErrorSDFT, state)
    doesnotexist = sdft_previousdft(state, 1)
    nothing
end

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

@testset "Sliding DFT" begin
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

    # Method with dft history and more data points
    @testset "TestSDFT" begin
        method = TestSDFT(n)
        dfty = collect(method(y))
        @testset for i in eachindex(sample_offsets)
            @test dfty[1 + sample_offsets[i]] ≈ dfty_sample[i]
        end
    end

    # Exceptions
    @testset "Exceptions" begin
        @test_throws "insufficient data" iterate(SDFT(10)(ones(5)))
        @test_throws "insufficient data" iterate(SDFT(10)(Float64[]))
        @test_throws "previous DFT results not available" collect(ErrorSDFT()(y))
    end

    # Additional coverage
    @testset "Extra" begin
        itr = SDFT(n)(y)
        _, state = iterate(itr)
        @test ismissing(Base.isdone(itr))
        @test ismissing(Base.isdone(itr, state))
        FourierTools.sdft_updatedfthistory!(nothing)
        FourierTools.sdft_updatefragment!(nothing, nothing, nothing)
        dummy_state = FourierTools.SDFTStateData(nothing, nothing, 1.0, 1, 1)
        @test FourierTools.haspreviousdata(dummy_state) == false
        # sdft_dataoffsets
        @test iszero(FourierTools.sdft_dataoffsets(SDFT(n)))
        @test isnothing(FourierTools.sdft_dataoffsets(nothing))
    end
end