@testset "Test resampling  methods" begin
    @testset "Test that upsample and downsample is reversible" begin
        for dim = 1:3
            for _ in 1:5
                s_small = ntuple(_ -> rand(1:13), dim)
                s_large = ntuple(i -> max.(s_small[i], rand(10:16)), dim)
                                
                x = opt_cu(randn(Float32, (s_small)), use_cuda)
                @test x == resample(x, s_small)
                @test Float32.(x) ≈ Float32.(resample(resample(x, s_large), s_small))
                @test x ≈ resample_by_FFT(resample_by_FFT(x, s_large), s_small)
                @test Float32.(x) ≈ Float32.(resample_by_RFFT(resample_by_RFFT(x, s_large), s_small))
                @test x ≈ FourierTools.resample_by_1D(FourierTools.resample_by_1D(x, s_large), s_small)
                x = opt_cu(randn(ComplexF32, (s_small)), use_cuda)
                @test x ≈ resample(resample(x, s_large), s_small)
                @test x ≈ resample_by_FFT(resample_by_FFT(x, s_large), s_small)
                @test x ≈ resample_by_FFT(resample_by_FFT(real(x), s_large), s_small) + 1im .* resample_by_FFT(resample_by_FFT(imag(x), s_large), s_small) 
                @test x ≈ FourierTools.resample_by_1D(FourierTools.resample_by_1D(x, s_large), s_small)
            end
        end
    end

    @testset "Test that different resample methods are consistent" begin
        for dim = 1:3
            for _ in 1:5
                s_small = ntuple(_ -> rand(1:13), dim)
                s_large = ntuple(i -> max.(s_small[i], rand(10:16)), dim)
                
                x = opt_cu(randn(Float32, (s_small)), use_cuda)
                @test ≈(FourierTools.resample(x, s_large), FourierTools.resample_by_1D(x, s_large))
            end
        end
    end

    @testset "Test that complex and real routine produce same result for real array" begin
        for dim = 1:3
            for _ in 1:5
                s_small = ntuple(_ -> rand(1:13), dim)
                s_large = ntuple(i -> max.(s_small[i], rand(10:16)), dim)
                
                x = opt_cu(randn(Float32, (s_small)), use_cuda)
                @test Float32.(resample(x, s_large)) ≈ Float32.(real(resample(ComplexF32.(x), s_large)))
                @test FourierTools.resample_by_1D(x, s_large) ≈ real(FourierTools.resample_by_1D(ComplexF32.(x), s_large))
            end
        end
    end

    @testset "Tests that resample_by_FFT is purely real" begin
        function test_real(s_1, s_2)
            x = opt_cu(randn(Float32, (s_1)), use_cuda)
            y = resample_by_FFT(x, s_2)
            @test all(( imag.(y) .+ 1 .≈ 1))
            y = FourierTools.resample_by_1D(x, s_2)
            @test all(( imag.(y) .+ 1 .≈ 1))
        end

        for dim = 1:3
            for _ in 1:5
                s_1 = ntuple(_ -> rand(1:13), dim)
                s_2 = ntuple(i -> rand(1:13), dim)
                test_real(s_1, s_2)
            end
        end
            
        test_real((4, 4),(6, 6))
        test_real((4, 4),(6, 7))
        test_real((4, 4),(9, 9))
        test_real((4, 5),(9, 9))
        test_real((4, 5),(9, 8))
        test_real((8, 8),(6, 7))
        test_real((8, 8),(6, 5))
        test_real((8, 8),(4, 5))
        test_real((9, 9),(4, 5))
        test_real((9, 9),(4, 5))
        test_real((9, 9),(7, 8))
        test_real((9, 9),(6, 5))

    end

    @testset "Sinc interpolation based on FFT" begin

    function test_interpolation_sum_fft(N_low, N)
	    x_min = 0.0
	    x_max = 16π
	    
	    xs_low = opt_cu(range(x_min, x_max, length=N_low+1)[1:N_low], use_cuda)
	    xs_high = opt_cu(range(x_min, x_max, length=N)[1:end-1], use_cuda)
	    f(x) = sin(0.5*x) + cos(x) + cos(2 * x) + sin(0.25*x)
	    arr_low = f.(xs_low)
	    arr_high = f.(xs_high)

	    xs_interp = range(x_min, x_max, length=N+1)[1:N]
	    arr_interp = resample(arr_low, N)
	    arr_interp2 = FourierTools.resample_by_1D(arr_low, N)


        @test ≈(arr_interp[2*N ÷10: N*8÷10], arr_high[2* N ÷10: N*8÷10], rtol=0.05)
        @test ≈(arr_interp2[2*N ÷10: N*8÷10], arr_high[2* N ÷10: N*8÷10], rtol=0.05)
    end

    test_interpolation_sum_fft(128, 1000)
    test_interpolation_sum_fft(129, 1000)
    test_interpolation_sum_fft(120, 1531)
    test_interpolation_sum_fft(121, 1211)
    end

    @testset "Upsample2 compared to resample" begin
    for sz in ((10,10),(5,8,9),(20,5,4))
        a = opt_cu(rand(sz...), use_cuda)
        @test ≈(upsample2(a), resample(a,sz.*2))
        @test ≈(upsample2_abs2(a),abs2.(resample(a,sz.*2)))
        a = opt_cu(rand(ComplexF32, sz...), use_cuda)
        @test ≈(upsample2(a),resample(a,sz.*2))
        @test ≈(upsample2_abs2(a),abs2.(resample(a,sz.*2)))
        s2 = (d == 2 ? sz[d]*2 : sz[d] for d in 1:length(sz))
        @test ≈(upsample2(a, dims=(2,)),resample(a,s2))
        @test ≈(upsample2_abs2(a, dims=(2,)),abs2.(resample(a,s2)))
        @test size( upsample2(collect(collect(1.0:9.0)'); fix_center=true, keep_singleton=true)) == (1,18)
        @test upsample2(collect(1.0:9.0); fix_center=false)[1:16] ≈ upsample2(collect(1.0:9.0); fix_center=true)[2:17]
    end
    end

    @testset "Downsampling based on frequency cutting" begin
    function test_resample(N_low, N)
	    x_min = 0.0
	    x_max = 16π
	    
	    xs_low = opt_cu(range(x_min, x_max, length=N_low+1)[1:N_low], use_cuda)
	    f(x) = sin(0.5*x) + cos(x) + cos(2 * x) + sin(0.25*x)
	    arr_low = f.(xs_low)

	    xs_interp = range(x_min, x_max, length=N+1)[1:N]
	    arr_interp = resample(arr_low, N)

	    xs_interp_s = range(x_min, x_max, length=N+1)[1:N]

        arr_ds = resample(arr_interp, (N_low,) )
        @test ≈(arr_ds, arr_low)
        @test eltype(arr_low) === eltype(arr_ds)
        @test eltype(arr_interp) === eltype(arr_ds)
    end

    test_resample(128, 1000)
    test_resample(128, 1232)
    test_resample(128, 255)
    test_resample(253, 254)
    test_resample(253, 1001)
    test_resample(99, 100101)
    end

    @testset "FFT resample in 2D" begin    
        function test_2D(in_s, out_s)
            x = opt_cu(range(-10.0, 10.0, length=in_s[1] + 1)[1:end-1], use_cuda)
            y = opt_cu(range(-10.0, 10.0, length=in_s[2] + 1)[1:end-1]', use_cuda)
    	    arr = abs.(x) .+ abs.(y) .+ sinc.(sqrt.(x .^2 .+ y .^2))
    	    arr_interp = resample(arr[1:end, 1:end], out_s);
    	    arr_ds = resample(arr_interp, in_s)
            @test arr_ds ≈ arr
        end
    
        test_2D((128, 128), (150, 150))
        test_2D((128, 128), (151, 151))
        test_2D((129, 129), (150, 150))
        test_2D((129, 129), (151, 151))
        
        test_2D((150, 128), (151, 150))
        test_2D((128, 128), (151, 153))
        test_2D((129, 128), (150, 153))
        test_2D((129, 128), (129, 153))
    
    
        x = opt_cu(range(-10.0, 10.0, length=129)[1:end-1], use_cuda)
        x2 = opt_cu(range(-10.0, 10.0, length=130)[1:end-1], use_cuda)
        x_exact = opt_cu(range(-10.0, 10.0, length=2049)[1:end-1], use_cuda)
        y = x'
        y2 = x2'
        y_exact = x_exact'
        arr = abs.(x) .+ abs.(y) .+sinc.(sqrt.(x .^2 .+ y .^2))
        arr2 = abs.(x) .+ abs.(y) .+sinc.(sqrt.(x .^2 .+ y .^2))
        arr_exact = abs.(x_exact) .+ abs.(y_exact) .+ sinc.(sqrt.(x_exact .^2 .+ y_exact .^2))
        arr_interp = resample(arr[1:end, 1:end], (131, 131));
        arr_interp2 = resample(arr[1:end, 1:end], (512, 512));
        arr_interp3 = resample(arr[1:end, 1:end], (1024, 1024));
        arr_ds = resample(arr_interp, (128, 128))
        arr_ds2 = resample(arr_interp, (128, 128))
        arr_ds23 = resample(arr_interp2, (512, 512))
        arr_ds3 = resample(arr_interp, (128, 128))
    
        @test ≈(arr_ds3, arr)
        @test ≈(arr_ds2, arr)
        @test ≈(arr_ds, arr)
        @test ≈(arr_ds23, arr_interp2)
    
    end
    
    
    @testset "FFT resample 2D for a complex signal" begin
    
        function test_2D(in_s, out_s)
        	x = opt_cu(range(-10.0, 10.0, length=in_s[1] + 1)[1:end-1], use_cuda)
        	y = opt_cu(range(-10.0, 10.0, length=in_s[2] + 1)[1:end-1]', use_cuda)
        	f(x, y) = 1im * (abs(x) + abs(y) + sinc(sqrt(x ^2 + y ^2)))
        	f2(x, y) =  abs(x) + abs(y) + sinc(sqrt((x - 5) ^2 + (y - 5)^2))
        
        	arr = f.(x, y) .+ f2.(x, y)
        	arr_interp = resample(arr[1:end, 1:end], out_s);
        	arr_ds = resample(arr_interp, in_s)
            
            @test eltype(arr) === eltype(arr_ds)
            @test eltype(arr_interp) === eltype(arr_ds)
            @test imag(arr) ≈ imag(arr_ds)
            @test real(arr) ≈ real(arr_ds)
        end
    
        test_2D((128, 128), (150, 150))
        test_2D((128, 128), (151, 151))
        test_2D((129, 129), (150, 150))
        test_2D((129, 129), (151, 151))
        
        test_2D((150, 128), (151, 150))
        test_2D((128, 128), (151, 153))
        test_2D((129, 128), (150, 153))
        test_2D((129, 128), (129, 153))
    end
    
    
    @testset "FFT resample in 2D for a purely imaginary signal" begin
        function test_2D(in_s, out_s)
        	x = opt_cu(range(-10.0, 10.0, length=in_s[1] + 1)[1:end-1], use_cuda)
        	y = opt_cu(range(-10.0, 10.0, length=in_s[2] + 1)[1:end-1]', use_cuda)
        	f(x, y) = 1im * (abs(x) + abs(y) + sinc(sqrt(x ^2 + y ^2)))
        
        	arr = f.(x, y)
        	arr_interp = resample(arr[1:end, 1:end], out_s);
        	arr_ds = resample(arr_interp, in_s)
            
            @test imag(arr) ≈ imag(arr_ds)
            @test all(real(arr_ds) .< 1e-13)
            @test all(real(arr_interp) .< 1e-13)
        end 
    
        test_2D((128, 128), (150, 150))
        test_2D((128, 128), (151, 151))
        test_2D((129, 129), (150, 150))
        test_2D((129, 129), (151, 151))
        
        test_2D((150, 128), (151, 150))
        test_2D((128, 128), (151, 153))
        test_2D((129, 128), (150, 153))
        test_2D((129, 128), (129, 153))
    end

    @testset "test select_region_ft" begin
        x = opt_cu([1,2,3,4], use_cuda)
        res = select_region_ft(ffts(x), (5,))
        @test res == opt_cu(ComplexF64[-1.0 + 0.0im, -2.0 - 2.0im, 10.0 + 0.0im, -2.0 + 2.0im, -1.0 + 0.0im], use_cuda)
        x = opt_cu([3.1495759241275225 0.24720770605505335 -1.311507800204285 -0.3387627167144301; -0.7214121984874265 -0.02566249380406308 0.687066447881175 -0.09536748694092163; -0.577092696986848 -0.6320809680268722 -0.09460071173365793 0.7689715736798227; 0.4593837753047561 -1.0204193548690512 -0.28474772376166907 1.442443602597533], use_cuda)
        res = select_region_ft(ffts(x), (7, 7))
        @test collect(res) ≈ opt_cu(ComplexF64[0.0 + 0.0im 0.0 + 0.0im 0.0 + 0.0im 0.0 + 0.0im 0.0 + 0.0im 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.32043577156395486 + 0.0im 2.321469443190397 + 0.7890379226962572im 0.38521287113798636 + 0.0im 2.321469443190397 - 0.7890379226962572im 0.32043577156395486 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 1.3691035744780353 + 0.16703621316206385im 2.4110077589815555 - 0.16558718095884828im 2.2813159163314163 - 0.7520360306228049im 7.47614366018844 - 4.139633109911205im 1.3691035744780353 + 0.16703621316206385im 0.0 + 0.0im; 0.0 + 0.0im 0.4801675770812479 + 0.0im 3.3142445917764407 - 3.2082400832669373im 1.6529948781166373 + 0.0im 3.3142445917764407 + 3.2082400832669373im 0.4801675770812479 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 1.3691035744780353 - 0.16703621316206385im 7.47614366018844 + 4.139633109911205im 2.2813159163314163 + 0.7520360306228049im 2.4110077589815555 + 0.16558718095884828im 1.3691035744780353 - 0.16703621316206385im 0.0 + 0.0im; 0.0 + 0.0im 0.32043577156395486 + 0.0im 2.321469443190397 + 0.7890379226962572im 0.38521287113798636 + 0.0im 2.321469443190397 - 0.7890379226962572im 0.32043577156395486 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im 0.0 + 0.0im 0.0 + 0.0im 0.0 + 0.0im 0.0 + 0.0im 0.0 + 0.0im], use_cuda)
    end

    @testset "test resample_czt" begin
        dim =2
        s_small = (12,16) # ntuple(_ -> rand(1:13), dim)
        s_large = (20,18) # ntuple(i -> max.(s_small[i], rand(10:16)), dim)
        dat = select_region(opt_cu(randn(Float32, (5,6)), use_cuda), new_size= s_small)
        rs1 = FourierTools.resample(dat, s_large)
        rs1b = select_region(rs1, new_size=size(dat))
        rs2 = FourierTools.resample_czt(dat, s_large./s_small, do_damp=false)
        @test rs1b ≈ rs2
        rs2 = FourierTools.resample_czt(dat, (x->s_large[2]./s_small[2], y->s_large[1]./s_small[1]), do_damp=false)
        @test rs1b ≈ rs2
        rs2 = FourierTools.resample_czt(dat, (x->1.0, y->1.0), shear=(x->10.0,y->0.0),do_damp=false)
        @test shear(dat,10) ≈ rs2
        rs2 = FourierTools.resample_czt(dat, (x->1.0, y->1.0), shear=(10.0,0.0),do_damp=false)
        @test shear(dat,10) ≈ rs2
        rs2 = barrel_pin(dat, 0.5)
        rs2b = FourierTools.resample_czt(dat, (x -> 1.0 + 0.5 .* (x-0.5)^2,x -> 1.0 + 0.5 .* (x-0.5)^2))
        @test rs2b ≈ rs2
    end

    if (dat isa Array)
        @testset "test resample_nfft" begin
        dim =2
        s_small = (12,16) # ntuple(_ -> rand(1:13), dim)
        s_large = (20,18) # ntuple(i -> max.(s_small[i], rand(10:16)), dim)
        dat = select_region(opt_cu(randn(Float32, (5,6)), use_cuda), new_size= s_small)
        rs1 = FourierTools.resample(dat, s_large)
        rs1b = select_region(rs1, new_size=size(dat))
        mymap = (t) -> t .* s_small ./ s_large  
        rs3 = FourierTools.resample_nfft(dat, mymap)
        @test isapprox(rs1b, rs3, rtol=0.1)
        new_pos = mymap.(idx(size(dat), scale=ScaFT))
        rs4 = FourierTools.resample_nfft(dat, new_pos)
        @test rs4 ≈ rs3
        new_pos = cat(s_small[1]./s_large[1] .* xx(size(dat), scale=ScaFT), s_small[2]./s_large[2] .* yy(size(dat), scale=ScaFT),dims=3)
        rs5 = FourierTools.resample_nfft(dat, new_pos)
        @test rs5 ≈ rs3
        # @test rs1b ≈ rs3
        # test both modes: src and destination but only for a 1-pixel shift
        rs6 = FourierTools.resample_nfft(dat, t->t .+ 1.0, is_src_coords=false, is_in_pixels=true)
        rs7 = FourierTools.resample_nfft(dat, t->t .- 1.0, is_src_coords=true, is_in_pixels=true)
        @test rs6 ≈ rs7
        # test shrinking by a factor of two
        new_pos = cat(xx(s_small.÷2, scale=ScaFT),yy(s_small.÷2, scale=ScaFT), dims=3)
        rs8 = FourierTools.resample_nfft(dat, t->t, s_small.÷2, is_src_coords=true)
        rs9 = FourierTools.resample_nfft(dat, new_pos,  is_src_coords=true)
        rss = FourierTools.resample(dat, s_small.÷2)
        @test rs8 ≈ rs9
        rs10 = FourierTools.resample_nfft(dat, t->t, s_small.÷2; is_src_coords=false, is_in_pixels=true)
        new_pos = cat(xx(s_small, offset=(0,0)),yy(s_small,offset=(0,0)), dims=3)
        rs11 = FourierTools.resample_nfft(dat, new_pos, s_small.÷2; is_src_coords=false, is_in_pixels=true)
        @test rs10 ≈ rs11    
        # test the non-strided array
        rs6 = FourierTools.resample_nfft(Base.PermutedDimsArray(dat,(2,1)), t->t .+ 1.0, is_src_coords=false, is_in_pixels=true)
        rs7 = FourierTools.resample_nfft(Base.PermutedDimsArray(dat,(2,1)), t->t .- 1.0, is_src_coords=true, is_in_pixels=true)
        @test rs6 ≈ rs7
        rs6 = FourierTools.resample_nfft(1im .* dat , t->t .* 2.0, s_small.÷2, is_src_coords=false, is_in_pixels=false, pad_value=0.0)
        rs7 = FourierTools.resample_nfft(1im .* dat, t->t .* 0.5, s_small.÷2, is_src_coords=true, is_in_pixels=false, pad_value=0.0)
        @test rs6.*4 ≈ rs7
        end
    else
        @warn "Skipping test for CuArray, since nfft does not support CuArray"
    end

end
