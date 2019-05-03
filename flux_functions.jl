#
# Data preparation
#
function prepare_datas(X_,y_)

    y = reshape(y_[:],1,length(y_))

    x = X_[1:4,:]
    ap = reshape(X_[8,:],1,size(X_,2))
    b = reshape(X_[9,:],1,size(X_,2))
    T = reshape(X_[10,:],1,size(X_,2))
    sc = reshape(X_[11,:],1,size(X_,2))
    tg = reshape(X_[12,:],1,size(X_,2))
    return Float32.(x), Float32.(y), Float32.(T), Float32.(ap), Float32.(b), Float32.(sc), Float32.(tg)
end

function gkfolds(X_, y_, idx_label; k = 5)

    dd = kfolds(shuffleobs(unique(X_[idx_label,:])), k = k);

    out = []

    for j = 1:k
        train_lab, vald_lab = dd[j]
        train_idx = Int64[]
        valid_idx = Int64[]

        for i = 1:size(X_,2)
            if findall(X_[idx_label,i] .== train_lab) != []
                push!(train_idx, i)
            else
                push!(valid_idx, i)
            end
        end

        push!(out,((X_[:,train_idx],y_[train_idx]),(X_[:,valid_idx],y_[valid_idx])))

    end

    return out
end

#
# Cp calculations
#
at_gfu(x) = 3.0.*x[1,:] .+ 5.0.*x[2,:] + 3.0.*x[3,:] + 3.0.*x[4,:]# + 2*MgO + 2*MgO
aCpl(x) = 81.37.*x[1,:] .+ 27.21.*x[2,:] .+ 100.6.*x[3,:]  .+ 50.13.*x[4,:] .+ x[1,:].*(x[4,:].*x[4,:]).*151.7

ap(x) = reshape(aCpl(x) - 3.0.*8.314.*at_gfu(x),1,size(x,2))
b(x) = reshape(0.0943.*x[2,:] + 0.01578.*x[4,:],1,size(x,2)) #bCpl

#
# Function for initial bias values
#

function thousands(size)
    return ones(size).*log.(1000.0)
    end

function tens(size)
    return ones(size).*log.(10.0)
end

function init_both(dims)
    return ones(dims).*[log.(1000.);log.(10.);log.(10.)]
end

init_random(dims...) = randn(Float32, dims...) .* [5.;10.]
# Function for extracting parameters from the network

function tg(x,network)
    return reshape(exp.(network(x[1:4,:])[1,:]),1,size(x,2))
end

function ScTg(x,network)
    return reshape(exp.(network(x[1:4,:])[2,:]),1,size(x,2))
end

function fragility(x,network)
    return reshape(exp.(network(x[1:4,:])[3,:]),1,size(x,2))
end

#
# Thermodynamic equations : Adam and Gibbs model
#

function Be(x,network, Ae)
    return (12.0.-Ae).*(tg(x,network) .* ScTg(x,network))
end

function dCp(x, T, ap, b, network)
    return ap.*(log.(T).-log.(tg(x,network))) .+ b.*(T.-tg(x,network))
end

# AG EQUATION
function ag(x, T, ap, b, network, Ae)
    return Ae .+ Be(x,network, Ae) ./ (T.* (ScTg(x,network) .+ dCp(x, T, ap, b,network)))
end

# MYEGA EQUATION
function myega(x, T, network, Ae)
    return Ae .+ (12.0 .- Ae).*(tg(x,network)./T).*exp.((fragility(x,network)./(12.0.-Ae).-1.0).*(tg(x,network)./T.-1.0))
end

#
# LOSS FUNCTIONS
#

function mse(yp, y)
    return sqrt(sum((yp .- y).^2)./size(y, 2))
end

function loss_n(x, T, ap, b, y_target,network, Ae)
    return mse(ag(x, T, ap, b,network, Ae), y_target) # viscosity AG
end

function loss_n_myega(x, T, y_target,network, Ae)
    return mse(myega(x, T,network, Ae), y_target) # viscosity MYEGA
end

function loss_n_tvf(x, T, y_target,network)
    return mse(tvf(x,T,network),y_target) # viscosity TVF
end

function loss_sc(x,sc,network)
    return mse(ScTg(x,network),sc) # Configurational entropy
end

function loss_tg(x,target,network)
    return mse(tg(x,network),target) # glass transition T
end

function loss_raman(x,raman_target,network)
    return mse(network(x),raman_target) # Raman spectra
end
