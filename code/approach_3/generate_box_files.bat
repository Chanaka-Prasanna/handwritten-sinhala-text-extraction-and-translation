@echo off
setlocal enabledelayedexpansion

:: Set the directory paths
set INPUT_DIR=D:\AI_ML_Internship_Assignment_Chanaka_Prasanna_Dissanayaka\data\train
set OUTPUT_DIR=D:\AI_ML_Internship_Assignment_Chanaka_Prasanna_Dissanayaka\fine-tune

:: Create output directory if it doesn't exist
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

:: Loop through all image files in the input directory and its subdirectories
for /r "%INPUT_DIR%" %%f in (*.jpg) do (
    echo Processing %%f
    set "filename=%%~nf"
    set "filepath=%%f"
    set "outputfile=%OUTPUT_DIR%\!filename!"
    tesseract "!filepath!" "!outputfile!" -l sin --psm 6 makebox
    if errorlevel 1 echo Error processing %%f
)

echo Done!
pause
