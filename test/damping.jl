using IndexFunArrays
@testset "Test damping functions" begin

    @testset "Test damp_edge_outside" begin
        sz = (512,512)
        data = disc(sz,150.0, offset=CtrCorner);
        data_d = damp_edge_outside(data);
        fta = abs.(ft(data));
        ftb = abs.(ft(data_d));
        @test fta[size(fta)[1]÷2+1,1] > 50
        @test ftb[size(ftb)[1]÷2+1,1] < 15
    end

end
