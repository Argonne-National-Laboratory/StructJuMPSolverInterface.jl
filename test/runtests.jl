# push!(LOAD_PATH,"../src")
using MPI
using Test
using StructJuMP

case = "./test/PowerGrid/data/case9"
prof = false
contingencies = ["2", "8"]
    
# Compute reference solution using Ipopt and JuMP

include("./PowerGrid/scopf.jl")

@testset "Testing PIPS Solver Interface" begin
    
    opfdata = opf_loaddata(case)
    
    lines_off=Array{Line}(undef, length(contingencies))
    for l in 1:length(lines_off)
        lines_off[l] = opfdata.lines[parse(Int,contingencies[l])]
    end
    @show lines_off
    scopfdata = SCOPFData(opfdata,lines_off)
    scopfmodel = scopf_model(scopfdata,case)
    opfmodel,status = scopf_solve(scopfmodel,scopfdata,case)
    @test status==:Optimal
    if status==:Optimal
        scopf_outputAll(opfmodel,scopfdata)
    end
    refobjective = getobjectivevalue(opfmodel)
    refVm=getvalue(getindex(opfmodel,:Vm)) 
    refVa=getvalue(getindex(opfmodel,:Va))
    refPg=getvalue(getindex(opfmodel,:Pg)) 
    refQg=getvalue(getindex(opfmodel,:Qg))
    
    include("./PowerGrid/scopf_structjump.jl")
    
    solver = "Ipopt"
    
    raw = loadrawdata(case)
    
    scopfmodel, scopfdata = scopf_model(raw, contingencies)
    
    scopf_init_x(scopfmodel,scopfdata)
    
    model,status = scopf_solve(scopfmodel,scopfdata, solver, prof)
    @test status == :Solve_Succeeded
    
    @test refobjective ≈ getObjectiveVal(model)
    @test refVa[:,0] ≈ getvalue(getindex(model,:Va)) atol=1e-5
    @test refVm[:,0] ≈ getvalue(getindex(model,:Vm)) atol=1e-5
    @show refPg
    @show getvalue(getindex(model,:Pg))
    @test refPg ≈ getvalue(getindex(model,:Pg)) atol=1e-5
    @test refQg ≈ getvalue(getindex(model,:Qg)) atol=1e-5
    if status==:Optimal || status == :Solve_Succeeded
        scopf_structjump_outputAll(scopfmodel,scopfdata)
    end
    
    solver = "PipsNlp"
    
    raw = loadrawdata(case)
    
    scopfmodel, scopfdata = scopf_model(raw, contingencies)
    
    scopf_init_x(scopfmodel,scopfdata)
    
    model,status = scopf_solve(scopfmodel,scopfdata, solver, prof)
    @test status == :Solve_Succeeded
    
    @test refobjective ≈ getObjectiveVal(model)
    @test refVa[:,0] ≈ getvalue(getindex(model,:Va)) atol=1e-5
    @test refVm[:,0] ≈ getvalue(getindex(model,:Vm)) atol=1e-5
    @test refPg ≈ getvalue(getindex(model,:Pg)) atol=1e-5
    @test refQg ≈ getvalue(getindex(model,:Qg)) atol=1e-5
    if status==:Optimal || status == :Solve_Succeeded
        scopf_structjump_outputAll(scopfmodel,scopfdata)
    end
end
