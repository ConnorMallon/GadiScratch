#!/bin/bash
#PBS -P bt62
#PBS -q normal
#PBS -l walltime=48:00:00
#PBS -l ncpus=16
#PBS -l mem=64gb
#PBS -N Solver.jl
#PBS -l software=Gridap.jl
#PBS -o /scratch/bt62/cm8825/MRI_Solver_nt3only/stdout.txt
#PBS -e /scratch/bt62/cm8825/MRI_Solver_nt3only/stderr.txt
#PBS -l wd

BIN=/home/565/cm8825/julia-1.4.2/bin/julia
dir=/scratch/bt62/cm8825/MRI_Solver_nt3only

cd $dir

export MKL_NUM_THREADS=16
export OMP_NUM_THREADS=16

/home/565/cm8825/julia-1.4.2/bin/julia /scratch/bt62/cm8825/MRI_Solver_nt3only/Solver.jl
