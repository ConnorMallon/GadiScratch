#!/bin/bash
#PBS -P bt62
#PBS -q normal
#PBS -l walltime=20:00:00
#PBS -l ncpus=48
#PBS -l mem=192gb
#PBS -N Solver_Testing_c48.jl
#PBS -l software=Gridap.jl
#PBS -o /scratch/bt62/cm8825/MRI_Solver/tmp/stdoutd_c48.txt
#PBS -e /scratch/bt62/cm8825/MRI_Solver/tmp/stderrd_c48.txt
#PBS -l wd

BIN=/home/565/cm8825/julia-1.4.2/bin/julia
dir=/scratch/bt62/cm8825/MRI_Solver

cd $dir

export MKL_NUM_THREADS=48
export OMP_NUM_THREADS=48

/home/565/cm8825/julia-1.4.2/bin/julia /scratch/bt62/cm8825/MRI_Solver/Solver.jl