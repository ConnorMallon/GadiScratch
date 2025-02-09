#!/bin/bash
#PBS -P bt62
#PBS -q normal
#PBS -l walltime=48:00:00
#PBS -l ncpus=24
#PBS -l mem=96gb
#PBS -N Convergence500.jl
#PBS -l software=Gridap.jl
#PBS -o /scratch/bt62/cm8825/Convergence/stdout.txt
#PBS -e /scratch/bt62/cm8825/Convergence/stderr.txt
#PBS -l wd

BIN=/home/565/cm8825/julia-1.4.2/bin/julia
dir=/scratch/bt62/cm8825/Convergence

cd $dir

export MKL_NUM_THREADS=24
export OMP_NUM_THREADS=24

/home/565/cm8825/julia-1.4.2/bin/julia /scratch/bt62/cm8825/Convergence/FullMethodConvergence.jl
