using FourierTools
using ImageTransformations
using IndexFunArrays
using Zygote
using NDTools
using LinearAlgebra # for the assigned nfft function LinearAlgebra.mul!
using FractionalTransforms
using TestImages
using Random, Test, FFTW
using CUDA

Random.seed!(42)

use_cuda = false
function opt_cu(img, use_cuda=false)
    if (use_cuda)
        CuArray(img)
    else
        img
    end
end

function run_all_tests()
    include("fft_helpers.jl");
    include("fftshift_alternatives.jl");
    include("utils.jl");
    include("fourier_shifting.jl");
    include("fourier_shear.jl");
    include("fourier_rotate.jl");
    include("resampling_tests.jl"); ### nfft does not work with CUDA -> warning for this method
    include("convolutions.jl"); # spurious buffer problem in conv_p4 in CUDA?
    include("correlations.jl");
    include("custom_fourier_types.jl"); 
    include("damping.jl");
    include("czt.jl"); 
    include("nfft_tests.jl");
    include("fractional_fourier_transform.jl");
    include("fourier_filtering.jl");
    include("sdft.jl");
end

use_cuda=false
run_all_tests();

if CUDA.functional()
    use_cuda=true
    @testset "all in CUDA" begin
    CUDA.allowscalar(false);
    run_all_tests()
    end
else
    @testset "no CUDA available!" begin
        @test true == true
    end
end;

return
