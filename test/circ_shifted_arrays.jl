@testset "Convolution methods" begin
    # a = reshape(1:1000000,(1000,1000)) .+ 0
    # CUDA.allowscalar(false);
    sz = (15,12)
    myshift = (4,3)
    a = reshape(1:prod(sz),sz) .+ 0
    c = CircShiftedArray(a,myshift);
    b = copy(a)
    d = c .+ c;

    @test (c == c .+0)

    ca = circshift(a, myshift)
    # they are not the same but numerically the same:
    @test (c != ca)
    @test (collect(c) == ca)

    # adding a constant does not change the type
    @test typeof(c) == typeof(c .+ 0)
    # adding another CSA does not change the type
    b .= c
    @test b == collect(c)
    cc = CircShiftedArray(c,.-myshift)
    @test a == collect(cc)

    # assignment into a CSA
    d .= a
    @test d[1,1] == a[1,1]
    @test collect(d) == a


    # try a complicated broadcast expression
    @test ca.+ 2 .* sin.(ca) == collect(c.+2 .*sin.(c))

    #@run foo(c)
    @test sum(a, dims=1) != collect(sum(c,dims=1))
    @test sum(ca,dims=1) == collect(sum(c,dims=1))
    @test sum(a, dims=2) != collect(sum(c,dims=2))
    @test sum(ca,dims=2) == collect(sum(c,dims=2))

end