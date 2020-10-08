#!/bin/bash
#PBS -P bt62
#PBS -q normal
#PBS -l walltime=00:20:00
#PBS -l ncpus=1
#PBS -l mem=4gb
#PBS -N my_test.jl
#PBS -l software=Gridap.jl
#PBS -o /scratch/bt62/cm8825/stdout.txt
#PBS -e /scratch/bt62/cm8825/stderr.txt
#PBS -l wd

BIN=/home/565/cm8825/julia-1.4.2/bin/julia
dir=/scratch/bt62/cm8825/Testing/Test4

cd $dir

/home/565/cm8825/julia-1.4.2/bin/julia /scratch/bt62/cm8825/Testing/Test4/x2_FMonDataDevelopment.jl
