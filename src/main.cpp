#include <SDL2/SDL.h>
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include "raytracer.cuh"

const int WIDTH = 1280;
const int HEIGHT = 720;

int main(int argc, char* argv[]) {
    // Inicializar SDL
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "Error al inicializar SDL: " << SDL_GetError() << std::endl;
        return 1;
    }
    
    SDL_Window* window = SDL_CreateWindow("CUDA Raytracer",
                                          SDL_WINDOWPOS_CENTERED,
                                          SDL_WINDOWPOS_CENTERED,
                                          WIDTH, HEIGHT,
                                          SDL_WINDOW_SHOWN);
    if (!window) {
        std::cerr << "Error al crear ventana: " << SDL_GetError() << std::endl;
        SDL_Quit();
        return 1;
    }
    
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (!renderer) {
        std::cerr << "Error al crear renderer: " << SDL_GetError() << std::endl;
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }
    
    SDL_Texture* texture = SDL_CreateTexture(renderer,
                                            SDL_PIXELFORMAT_RGBA32,
                                            SDL_TEXTUREACCESS_STREAMING,
                                            WIDTH, HEIGHT);
    
    // Alocar memoria en GPU
    unsigned char* d_output;
    size_t imageSize = WIDTH * HEIGHT * 4;
    cudaMalloc(&d_output, imageSize);
    
    // Buffer en CPU
    unsigned char* h_output = new unsigned char[imageSize];
    
    // Loop principal
    bool running = true;
    SDL_Event event;
    Uint32 startTime = SDL_GetTicks();
    int frameCount = 0;
    
    std::cout << "Raytracer iniciado. Presiona ESC para salir." << std::endl;
    std::cout << "GPU: NVIDIA RTX 3090" << std::endl;
    
    while (running) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT || 
                (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_ESCAPE)) {
                running = false;
            }
        }
        
        // Calcular posición de cámara (rotación suave)
        float time = (SDL_GetTicks() - startTime) / 1000.0f;
        float radius = 4.0f;  // Órbita DENTRO de la esfera (radio < 8)
        float angle = time * 0.5f; // Velocidad de rotación
        
        Vec3 camPos(radius * cosf(angle), 1.0f + 0.5f * sinf(time * 0.3f), radius * sinf(angle));
        Vec3 camTarget(0, 0, 0);
        Vec3 lightPos(2.0f * cosf(time * 1.5f), 3.0f, 2.0f * sinf(time * 1.5f));  // Luz girando DENTRO
        
        // Renderizar en GPU
        launchRaytraceKernel(d_output, WIDTH, HEIGHT, camPos, camTarget, lightPos, time);
        
        // Verificar errores CUDA
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Error CUDA: " << cudaGetErrorString(err) << std::endl;
        }
        
        // Copiar resultado a CPU
        cudaMemcpy(h_output, d_output, imageSize, cudaMemcpyDeviceToHost);
        
        // Debug: verificar que hay datos
        if (frameCount == 1) {
            std::cout << "Primer pixel: R=" << (int)h_output[0] 
                      << " G=" << (int)h_output[1] 
                      << " B=" << (int)h_output[2] << std::endl;
        }
        
        // Actualizar textura
        SDL_UpdateTexture(texture, NULL, h_output, WIDTH * 4);
        
        // Renderizar
        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, texture, NULL, NULL);
        SDL_RenderPresent(renderer);
        
        // FPS counter
        frameCount++;
        if (frameCount % 60 == 0) {
            float fps = frameCount / ((SDL_GetTicks() - startTime) / 1000.0f);
            std::cout << "FPS: " << fps << std::endl;
        }
    }
    
    // Limpiar
    delete[] h_output;
    cudaFree(d_output);
    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    
    return 0;
}