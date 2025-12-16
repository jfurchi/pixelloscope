#ifndef RAYTRACER_CUH
#define RAYTRACER_CUH

#include <cuda_runtime.h>

struct Vec3 {
    float x, y, z;
    
    __host__ __device__ Vec3() : x(0), y(0), z(0) {}
    __host__ __device__ Vec3(float x, float y, float z) : x(x), y(y), z(z) {}
    
    __host__ __device__ Vec3 operator+(const Vec3& v) const {
        return Vec3(x + v.x, y + v.y, z + v.z);
    }
    
    __host__ __device__ Vec3 operator-(const Vec3& v) const {
        return Vec3(x - v.x, y - v.y, z - v.z);
    }
    
    __host__ __device__ Vec3 operator*(float t) const {
        return Vec3(x * t, y * t, z * t);
    }
    
    __host__ __device__ float dot(const Vec3& v) const {
        return x * v.x + y * v.y + z * v.z;
    }
    
    __host__ __device__ float length() const {
        return sqrtf(x * x + y * y + z * z);
    }
    
    __host__ __device__ Vec3 normalize() const {
        float len = length();
        return len > 0 ? Vec3(x / len, y / len, z / len) : Vec3(0, 0, 0);
    }
    
    __host__ __device__ Vec3 reflect(const Vec3& normal) const {
        return *this - normal * (2.0f * this->dot(normal));
    }
};

struct Ray {
    Vec3 origin;
    Vec3 direction;
    
    __device__ Ray() : origin(), direction() {}
    __device__ Ray(const Vec3& o, const Vec3& d) : origin(o), direction(d) {}
};

struct Sphere {
    Vec3 center;
    float radius;
    bool mirrored;
    bool inside;
    
    __device__ bool intersect(const Ray& ray, float& t) const {
        Vec3 oc = ray.origin - center;
        float a = ray.direction.dot(ray.direction);
        float b = 2.0f * oc.dot(ray.direction);
        float c = oc.dot(oc) - radius * radius;
        float discriminant = b * b - 4 * a * c;
        
        if (discriminant < 0.0f) return false;
        
        float sqrt_d = sqrtf(discriminant);
        float t1 = (-b - sqrt_d) / (2.0f * a);
        float t2 = (-b + sqrt_d) / (2.0f * a);
        
        // Para esfera espejada por dentro, queremos estar dentro
        if (inside) {
            // Estamos dentro de la esfera, usamos la intersección lejana
            if (t2 > 0.001f) {
                t = t2;
                return true;
            }
        } else {
            // Esfera normal, usamos la intersección cercana
            if (t1 > 0.001f) {
                t = t1;
                return true;
            } else if (t2 > 0.001f) {
                t = t2;
                return true;
            }
        }
        
        return false;
    }
    
    __device__ Vec3 getNormal(const Vec3& point) const {
        Vec3 normal = (point - center).normalize();
        return inside ? normal * -1.0f : normal;
    }
};

// Declaración de la función kernel
void launchRaytraceKernel(unsigned char* d_output, int width, int height, 
                          Vec3 camPos, Vec3 camTarget, Vec3 lightPos,
                          float time);

#endif // RAYTRACER_CUH