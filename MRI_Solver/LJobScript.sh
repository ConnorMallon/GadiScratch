#!/bin/bash
#PBS -P bt62
#PBS -q normal
#PBS -l walltime=20:00:00
#PBS -l ncpus=4
#PBS -l mem=16gb
#PBS -N LSolver18.jl
#PBS -l software=Gridap.jl
#PBS -o /scratch/bt62/cm8825/Lstdout.txt
#PBS -e /scratch/bt62/cm8825/Lstderr.txt
#PBS -l wd

BIN=/home/565/cm8825/julia-1.4.2/bin/julia
dir=/scratch/bt62/cm8825/MRI_Solver

cd $dir

/home/565/cm8825/julia-1.4.2/bin/julia /scratch/bt62/cm8825/MRI_Solver/LSolver.jl
