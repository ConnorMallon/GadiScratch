#!/bin/bash
#PBS -P bt62
#PBS -q normal
#PBS -l walltime=20:00:00
#PBS -l ncpus=7
#PBS -l mem=28gb
#PBS -N Solver_c7_Pardiso.jl
#PBS -l software=Gridap.jl
#PBS -o /scratch/bt62/cm8825/Testing/Test6_PardisoFMonTrialData/stdout_c7.txt
#PBS -e /scratch/bt62/cm8825/Testing/Test6_PardisoFMonTrialData/stderr_c7.txt
#PBS -l wd

BIN=/home/565/cm8825/julia-1.4.2/bin/julia
dir=/scratch/bt62/cm8825/Testing/Test6_PardisoFMonTrialData

cd $dir

export MKL_NUM_THREADS=7
export OMP_NUM_THREADS=7

/home/565/cm8825/julia-1.4.2/bin/julia /scratch/bt62/cm8825/Testing/Test6_PardisoFMonTrialData/2_FullMethodDataTest.jl