
Visualizer of the Mandelbrot fractal. 

Compile using gcc:
$ g++ -mavx2 -o Frame Frame.cpp `sdl2-config --cflags --libs`

This requires Intel's AVX2 (Advanced Vector Extensions) to be available on the CPU, and SDL 2.0 to be installed.

Execute using the command 
$ ./Frame
