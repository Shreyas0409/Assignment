## Excutable Commands:

1) get the executable file name :
Get-ChildItem .\build\Release\ppe_detector.exe

2) Test path of model
Test-Path .\models\glove_detector.onnx

3)classes 
Test-Path .\models\classes.txt

4)list of classes
.\build\Release\ppe_detector.exe --list-classes

5) Run file
.\build\Release\ppe_detector.exe --input .\input\b10.jpg --class "Gloves"


## Input
Images are uploaded to test the images

## to Remove the build 
Remove-Item -Recurse -Force build -ErrorAction SilentlyContinue

## rebuild
cmake --build --config Release

## Rebuild
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake