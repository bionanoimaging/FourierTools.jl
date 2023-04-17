@testset "Convolution methods" begin
    # a = reshape(1:1000000,(1000,1000)) .+ 0
    sz = (15,12)
    myshift = (4,3)
    a = reshape(1:prod(sz),sz) .+ 0
    c = CircShiftedArray(a,myshift);
    b = copy(a)
    d = c .+ c;

    @test c == circshift(a,myshift)
    # adding a constant does not change the type
    @test typeof(c) == typeof(c .+ 0)
    # adding another CSA does not change the type

end