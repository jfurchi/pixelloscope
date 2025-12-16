@echo off
echo =======================================
echo   Compilando CUDA Raytracer con MSVC
echo =======================================
echo.

REM Limpiar archivos anteriores
if exist *.obj del *.obj
if exist raytracer.exe del raytracer.exe

REM Paso 1: Compilar kernel CUDA
echo [1/3] Compilando kernel CUDA...
nvcc -c src/raytracer.cu -o raytracer.obj -arch=sm_86 --compiler-options "/MD" -Xcompiler "/EHsc"
if errorlevel 1 (
    echo ERROR: Fallo al compilar CUDA kernel
    pause
    exit /b 1
)

REM Paso 2: Compilar main.cpp
echo [2/3] Compilando main.cpp...
nvcc -c src/main.cpp -o main.obj ^
   -I"C:\SDL2\include" ^
   --compiler-options "/EHsc /MD /O2"
if errorlevel 1 (
    echo ERROR: Fallo al compilar main.cpp
    pause
    exit /b 1
)

REM Paso 3: Enlazar
echo [3/3] Enlazando ejecutable...
nvcc -o raytracer.exe main.obj raytracer.obj ^
   -L"C:\SDL2\lib\x64" ^
   SDL2main.lib SDL2.lib shell32.lib ^
   -Xlinker "/SUBSYSTEM:CONSOLE,/ENTRY:mainCRTStartup,/NODEFAULTLIB:libcmt" ^
   --machine 64
if errorlevel 1 (
    echo ERROR: Fallo al enlazar
    pause
    exit /b 1
)

echo.
echo =======================================
echo   Compilacion exitosa!
echo =======================================
echo.
echo Ejecutando raytracer...
echo.

raytracer.exe