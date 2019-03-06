at_gfu(x) = 3.0.*x[1,:] .+ 5.0.*x[2,:] + 3.0.*x[3,:] + 3.0.*x[4,:]# + 2*MgO + 2*MgO
aCpl(x) = 81.37.*x[1,:] .+ 27.21.*x[2,:] .+ 100.6.*x[3,:]  .+ 50.13.*x[4,:] .+ x[1,:].*(x[4,:].*x[4,:]).*151.7

ap(x) = reshape(aCpl(x) - 3.0.*8.314.*at_gfu(x),1,size(x,2))
b(x) = reshape(0.0943.*x[2,:] + 0.01578.*x[4,:],1,size(x,2)) #bCpl

#
# With two subnetworks for Be and Sc
#

function thousands(size)
    return ones(size).*log.(100000)
    end

function tens(size)
    return ones(size).*log(10)
end

function init_both(dims)
    return ones(dims).*[log.(100000.);log.(10.)]
end

Be(x) = reshape(exp.(m1(x[1:4,:])[1,:]),1,size(x,2))
ScTg(x) = reshape(exp.(m1(x[1:4,:])[2,:]),1,size(x,2))


tg(x) = Be(x)./((12.0.-Ae).*ScTg(x))

dCp(x,T, ap, b) = ap.*(log.(T).-log.(tg(x))) .+ b.*(T.-tg(x))

model(x,T, ap, b) = Ae .+ Be(x) ./ (T.* (ScTg(x) .+ dCp(x,T, ap, b)))

mse(yp, y) = sqrt(sum((yp .- y).^2)./size(y, 2))
mse_weighted(yp, y, w) = sqrt(sum((yp .- y).^2.0./(w.^2))./size(y, 2))

predict(x,T, ap, b) = model(x,T, ap, b)

# loss functions

loss_n(x, T, ap, b, y_target) = mse(model(x, T, ap, b), y_target)

loss_sc(x,sc) = mse(ScTg(x),sc)

loss_tg(x,target) = mse(tg(x),target)

loss_tg_sc(x,tg_target,sc_target,) = loss_tg(x,tg_target) .+ loss_sc(x,sc_target).*1000.0

loss_global(x, T, ap, b, y_target, x2, tg2_target, sc2_target) = loss_n(x, T, ap, b, y_target) + loss_tg_sc(x2,tg2_target,sc2_target)
