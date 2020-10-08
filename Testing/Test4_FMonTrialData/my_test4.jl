module FullMethod

using Pkg
Pkg.activate(".")

using Gridap

domain = (0,1,0,1)
partition = (2,2)
model = CartesianDiscreteModel(domain,partition)
trian = Triangulation(model)
writevtk(trian,"trian")

end #module
