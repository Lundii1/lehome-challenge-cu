@echo off
call I:\temp\miniconda3\condabin\conda.bat activate I:\temp\miniconda3
call conda activate I:\temp\miniconda3\envs\lehome
set OMNI_KIT_ACCEPT_EULA=YES
echo Activated Conda environment: lehome
echo Isaac Sim EULA accepted for this shell session.
cmd /k
