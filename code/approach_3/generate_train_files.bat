@echo off
setlocal enabledelayedexpansion

:: Set the directory paths
set INPUT_DIR=D:\AI_ML_Internship_Assignment_Chanaka_Prasanna_Dissanayaka\data\train
set OUTPUT_DIR=D:\AI_ML_Internship_Assignment_Chanaka_Prasanna_Dissanayaka\fine-tune

:: Create output directory if it doesn't exist
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

:: Loop through all image files in the input directory
for /r "%INPUT_DIR%" %%f in (*.png) do (
    echo Processing %%f
    :: Extract the base name of the image file
    set "basename=%%~nf"
    :: Generate the .tr file
    tesseract "%%f" "%OUTPUT_DIR%\!basename!" -l sin --psm 6 box.train
)

echo Done!
pause
