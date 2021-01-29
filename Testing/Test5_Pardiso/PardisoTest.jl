module FEMDriver

using Pkg
Pkg.activate("/scratch/bt62/cm8825")

using Test
using Gridap
using GridapPardiso
using BenchmarkTools
using .Threads
@show nthreads()

function driver()

u(x) = x[1] - 2 * x[2]
f(x) = -Δ(u)(x)

tol = 1e-10

n = 100
domain = (0,1,0,1,0,1)
partition = (n,n,n)

# Simple 2D data for debugging. TODO: remove when fixed.
model = CartesianDiscreteModel(domain,partition)

order = 1 
degree = 2 * order
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

V = TestFESpace(model,ReferenceFE(lagrangian,Float64,order),conformity=:H1, dirichlet_tags="boundary")
U = TrialFESpace(V,u)

a(u,v) = ∫( ∇(v)⋅∇(u) )*dΩ 
l(v) = ∫( v*f )*dΩ 

# with non-sym storage
op = AffineFEOperator(a,l,U,V)

ls = PardisoSolver(op)
solver = LinearFESolver(ls)

uh = solve(solver,op)

x = get_free_values(uh)
A = get_matrix(op)
b = get_vector(op)

r = A*x - b
@test maximum(abs.(r)) < tol

end

driver()
@time driver()

end #module
