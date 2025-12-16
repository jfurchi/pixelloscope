#include "raytracer.cuh"
#include <stdio.h>

#define MAX_BOUNCES 64

__device__ Vec3 trace(Ray ray, const Sphere* spheres, int numSpheres, const Vec3& lightPos, int depth) {
    Vec3 color(1.0f, 1.0f, 1.0f);
    
    for (int bounce = 0; bounce < MAX_BOUNCES; bounce++) {
        float closest_t = 1e10f;
        int hit_sphere = -1;
        
        // Encontrar intersección más cercana
        for (int i = 0; i < numSpheres; i++) {
            float t = 0.0f;
            if (spheres[i].intersect(ray, t)) {
                if (t > 0.001f && t < closest_t) {
                    closest_t = t;
                    hit_sphere = i;
                }
            }
        }
        
        if (hit_sphere == -1) {
            // Color de fondo (gradiente cielo)
            float t = 0.5f * (ray.direction.y + 1.0f);
            Vec3 white(1.0f, 1.0f, 1.0f);
            Vec3 blue(0.5f, 0.7f, 1.0f);
            color.x *= (white.x * (1.0f - t) + blue.x * t);
            color.y *= (white.y * (1.0f - t) + blue.y * t);
            color.z *= (white.z * (1.0f - t) + blue.z * t);
            return color;
        }
        
        // Punto de intersección
        Vec3 hitPoint = ray.origin + ray.direction * closest_t;
        Vec3 normal = spheres[hit_sphere].getNormal(hitPoint);
        
        // Si es espejo, reflejar
        if (spheres[hit_sphere].mirrored) {
            Vec3 reflected = ray.direction.reflect(normal);
            ray.origin = hitPoint + normal * 0.001f;
            ray.direction = reflected;
            color.x *= 0.95f;
            color.y *= 0.95f;
            color.z *= 0.95f;
            continue;
        }
        
        // Objeto difuso - calcular iluminación
        Vec3 lightDir = (lightPos - hitPoint).normalize();
        float diff = fmaxf(normal.dot(lightDir), 0.0f);
        
        Vec3 objColor(0.8f, 0.3f, 0.3f);
        color.x *= objColor.x * (0.3f + 0.7f * diff);
        color.y *= objColor.y * (0.3f + 0.7f * diff);
        color.z *= objColor.z * (0.3f + 0.7f * diff);
        return color;
    }
    
    return Vec3(0, 0, 0);
}

__global__ void raytraceKernel(unsigned char* output, int width, int height,
                                Vec3 camPos, Vec3 camTarget, Vec3 lightPos,
                                Sphere sphere1, Sphere sphere2) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Configurar cámara
    Vec3 forward = (camTarget - camPos).normalize();
    Vec3 up(0, 1, 0);
    Vec3 right = Vec3(up.y * forward.z - up.z * forward.y,
                      up.z * forward.x - up.x * forward.z,
                      up.x * forward.y - up.y * forward.x).normalize();
    up = Vec3(forward.y * right.z - forward.z * right.y,
              forward.z * right.x - forward.x * right.z,
              forward.x * right.y - forward.y * right.x);
    
    // Calcular rayo
    float u = ((float)x / width - 0.5f) * 2.0f;
    float v = (0.5f - (float)y / height) * 2.0f;
    float aspect = (float)width / height;
    
    Vec3 rayDir = (forward + right * (u * aspect * 0.5f) + up * (v * 0.5f)).normalize();
    Ray ray(camPos, rayDir);
    
    // Definir escena
    Sphere spheres[2];
    spheres[0] = sphere1;
    spheres[1] = sphere2;
    
    // Trazar rayo
    Vec3 color = trace(ray, spheres, 2, lightPos, 0);
    
    // Clamp y gamma correction
    color.x = sqrtf(fminf(fmaxf(color.x, 0.0f), 1.0f));
    color.y = sqrtf(fminf(fmaxf(color.y, 0.0f), 1.0f));
    color.z = sqrtf(fminf(fmaxf(color.z, 0.0f), 1.0f));
    
    // Escribir píxel
    int idx = (y * width + x) * 4;
    output[idx + 0] = (unsigned char)(color.x * 255);
    output[idx + 1] = (unsigned char)(color.y * 255);
    output[idx + 2] = (unsigned char)(color.z * 255);
    output[idx + 3] = 255;
}

void launchRaytraceKernel(unsigned char* d_output, int width, int height,
                          Vec3 camPos, Vec3 camTarget, Vec3 lightPos, float time) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    // DEBUGGING: Esfera que cambia de tamaño con el tiempo
    Sphere sphere1, sphere2;
    
    // Radio que oscila entre 3 y 15 cada 10 segundos
    float bigRadius = 9.0f + 8.0f * sinf(time * 0.3f);
    
    // Esfera GRANDE espejada POR DENTRO (pulsante)
    sphere1.center = Vec3(0, 0, 0);
    sphere1.radius = 5.0f;
    sphere1.mirrored = true;
    sphere1.inside = false;
    
    // Esfera PEQUEÑA en el centro
    sphere2.center = Vec3(0, 0, 0);
    sphere2.radius = 0.5f;
    sphere2.mirrored = true;
    sphere2.inside = false;
    
    // Debug: imprimir info cada 30 frames
    static int frameCounter = 0;
    if (frameCounter++ % 30 == 0) {
        printf("Radio esfera grande: %.2f | Distancia camara: %.2f\n", 
               bigRadius, sqrtf(camPos.x*camPos.x + camPos.y*camPos.y + camPos.z*camPos.z));
    }
    
    raytraceKernel<<<gridSize, blockSize>>>(d_output, width, height, camPos, camTarget, lightPos, sphere1, sphere2);
    cudaDeviceSynchronize();
}