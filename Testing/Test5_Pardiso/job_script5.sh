#!/bin/bash
#PBS -P bt62
#PBS -q normal
#PBS -l walltime=00:05:00
#PBS -l ncpus=16
#PBS -l mem=96gb
#PBS -N PardisoTest16.jl
#PBS -l software=Gridap.jl
#PBS -o /scratch/bt62/cm8825/Testing/Test5_Pardiso/stdout_PardisoTest16.txt
#PBS -e /scratch/bt62/cm8825/Testing/Test5_Pardiso/stderr_PardisoTest16.txt
#PBS -l wd

BIN=/home/565/cm8825/julia-1.4.2/bin/julia
dir=/scratch/bt62/cm8825/Testing/Test5_Pardiso

cd $dir

export MKL_NUM_THREADS=16
export OMP_NUM_THREADS=16

time /home/565/cm8825/julia-1.4.2/bin/julia /scratch/bt62/cm8825/Testing/Test5_Pardiso/PardisoTest.jl

