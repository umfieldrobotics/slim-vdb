// Apache License
//
// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
// License link: https://github.com/AcademySoftwareFoundation/openvdb/blob/master/LICENSE
//
// Modified by Anja Sheppard, University of Michigan, 2025

#pragma once

#include <cmath>
#include <chrono>
#include <fstream>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/IO.h>
#include <nanovdb/util/cuda/CudaDeviceBuffer.h>
#include <nanovdb/util/Ray.h>
#include <nanovdb/util/HDDA.h>
#include <nanovdb/util/GridBuilder.h>
#include "ComputePrimitives.h"

#include "utils/Utils.h"
#include "Config.h"

#if defined(NANOVDB_USE_CUDA)
using BufferT = nanovdb::cuda::DeviceBuffer;
#else
using BufferT = nanovdb::HostBuffer;
#endif

template <slimvdb::Language L, int S>
using LabelGridT = std::conditional_t<L == slimvdb::CLOSED, nanovdb::VecXIGrid<S>, nanovdb::VecXFGrid<S>>;

template <slimvdb::Language L, int S>
void runNanoVDB(nanovdb::GridHandle<BufferT>& handle, nanovdb::GridHandle<BufferT>& label_handle, nanovdb::GridHandle<BufferT>& weight_handle, nanovdb::GridHandle<BufferT>& beta_handle,
                int width, int height, BufferT& imageBuffer, int index,
                const std::vector<double> origin, const std::vector<double> quaternion, const float min_range, const float max_range, const float p_threshold=0.1, const float* embeddings=nullptr, const uint16_t num_open_semantic_classes=0);


inline __hostdev__ uint32_t CompactBy1(uint32_t x)
{
    x &= 0x55555555;
    x = (x ^ (x >> 1)) & 0x33333333;
    x = (x ^ (x >> 2)) & 0x0f0f0f0f;
    x = (x ^ (x >> 4)) & 0x00ff00ff;
    x = (x ^ (x >> 8)) & 0x0000ffff;
    return x;
}

inline __hostdev__ uint32_t SeparateBy1(uint32_t x)
{
    x &= 0x0000ffff;
    x = (x ^ (x << 8)) & 0x00ff00ff;
    x = (x ^ (x << 4)) & 0x0f0f0f0f;
    x = (x ^ (x << 2)) & 0x33333333;
    x = (x ^ (x << 1)) & 0x55555555;
    return x;
}

inline __hostdev__ void mortonDecode(uint32_t code, uint32_t& x, uint32_t& y)
{
    x = CompactBy1(code);
    y = CompactBy1(code >> 1);
}

inline __hostdev__ void mortonEncode(uint32_t& code, uint32_t x, uint32_t y)
{
    code = SeparateBy1(x) | (SeparateBy1(y) << 1);
}

template<slimvdb::Language L, int S, typename RenderFn, typename GridT>
inline float renderImage(bool useCuda, const RenderFn renderOp, int width, int height, float* image,
                        const GridT* grid, const LabelGridT<L, S>* label_grid, const GridT* weight_grid=nullptr, const LabelGridT<L, S>* beta_grid=nullptr, const float* embeddings=nullptr, const uint16_t num_open_semantic_classes=0)
{
    using ClockT = std::chrono::high_resolution_clock;
    auto t0 = ClockT::now();

    computeForEach(
        useCuda, width * height, 512, __FILE__, __LINE__, [renderOp, image, grid, label_grid, weight_grid, beta_grid, embeddings, num_open_semantic_classes] __hostdev__(int start, int end) {
            renderOp(start, end, image, grid, label_grid, weight_grid, beta_grid, embeddings);
        });
    computeSync(useCuda, __FILE__, __LINE__);

    auto t1 = ClockT::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.f;
    return duration;
}

inline void saveImage(const std::string& filename, int width, int height, const float* image)
{
    const auto isLittleEndian = []() -> bool {
        static int  x = 1;
        static bool result = reinterpret_cast<uint8_t*>(&x)[0] == 1;
        return result;
    };

    float scale = 1.0f;
    if (isLittleEndian())
        scale = -scale;

    std::fstream fs(filename, std::ios::out | std::ios::binary);
    if (!fs.is_open()) {
        throw std::runtime_error("Unable to open file: " + filename);
    }

    fs << "PF\n"
       << width << "\n"
       << height << "\n"
       << scale << "\n";

    for (int i = 0; i < width * height; ++i) {
        float r1 = image[i];
        float r2 = image[width*height + i];
        float r3 = image[2*width*height + i];
        fs.write((char*)&r1, sizeof(float));
        fs.write((char*)&r2, sizeof(float));
        fs.write((char*)&r3, sizeof(float));
    }
}

template<typename Vec3T>
struct RayGenOp
{
    float mWBBoxDimZ;
    Vec3T mWBBoxCenter;

    inline RayGenOp(float wBBoxDimZ, Vec3T wBBoxCenter)
        : mWBBoxDimZ(wBBoxDimZ)
        , mWBBoxCenter(wBBoxCenter)
    {
    }

    inline __hostdev__ void operator()(int i, int w, int h, Vec3T& outOrigin, Vec3T& outDir) const
    {
        // perspective camera along Z-axis...
        uint32_t x, y;
#if 0
        mortonDecode(i, x, y);
#else
        x = i % w;
        y = i / w;
#endif
        const float fov = 45.f;
        const float u = (float(x) + 0.5f) / w;
        const float v = (float(y) + 0.5f) / h;
        const float aspect = w / float(h);
        const float Px = (2.f * u - 1.f) * tanf(fov / 2 * 3.14159265358979323846f / 180.f) * aspect;
        const float Py = (2.f * v - 1.f) * tanf(fov / 2 * 3.14159265358979323846f / 180.f);
        const Vec3T origin = mWBBoxCenter + Vec3T(0, 0, mWBBoxDimZ);
        Vec3T       dir(Px, Py, -1.f);
        dir.normalize();
        outOrigin = origin;
        outDir = dir;
    }
};

struct RGB {
    uint8_t r;
    uint8_t g;
    uint8_t b;
};

struct CompositeOp
{
    const uint8_t color_map[26][3] = {
        {  0,   0,   0},    // "unlabeled"
        {255,   0,   0},    // "outlier"
        {64,    0, 128},    // "car"
        {100, 230, 245},    // "bicycle"
        {100,  80, 250},    // "bus"
        { 30,  60, 150},    // "motorcycle"
        {  0,   0, 255},    // "on-rails"
        { 80,  30, 180},    // "truck"
        {  0,   0, 255},    // "other-vehicle"
        {255,  30,  30},    // "person"
        {255,  40, 200},    // "bicyclist"
        {150,  30,  90},    // "motorcyclist"
        {128,  64, 128},    // "road"
        {255, 150, 255},    // "parking"
        { 75,   0,  75},    // "sidewalk"
        {175,   0,  75},    // "other-ground"
        {128,   0,   0},    // "building"
        {255, 120,  50},    // "fence"
        {255, 150,   0},    // "other-structure"
        {150, 255, 170},    // "lane-marking"
        {128, 128,   0},    // "vegetation"
        {135,  60,   0},    // "trunk"
        {150, 240,  80},    // "terrain"
        {255, 240, 150},    // "pole"
        {255,   0,   0},    // "traffic-sign"
        { 50, 255, 255},    // "other-object"
    };

    const uint8_t color_map_scenenet[14][3] = {
        {  0,   0,   0},    // black (UNKNOWN)
        {  0,   0, 255},    // bright blue (BED)
        {233,  89,  48},    // warm reddish orange (BOOKS)
        {  0, 218,   0},    // bright green (CEILING)
        {149,   0, 240},    // purple (CHAIR)
        {222, 241,  24},    // light yellow-green (FLOOR)
        {255, 206, 206},    // pale pink (FURNITURE)
        {  0, 224, 229},    // cyan (OBJECTS)
        {106, 136, 204},    // medium blue (PAINTING)
        {117,  29,  41},    // deep red (SOFA)
        {240,  35, 235},    // magenta (TABLE)
        {  0, 167, 156},    // teal (TV)
        {249, 139,   0},    // bright orange (WALL)
        {225, 229, 194},    // beige (WINDOW)
    };

    // FOR CLOSED-SET
    template <uint16_t S>
    inline __hostdev__ void operator()(float* outImage, int i, int w, int h, nanovdb::math::VecXi<S>& alpha, float depth, const float k, const float min_range, const float p_threshold) const
    {
        int offset;
#if 0
        uint32_t x, y;
        mortonDecode(i, x, y);
        offset = x + y * w;
#else
        offset = i;
#endif

        // Posterior predictive evaluation and thresholding
        float sum = 0.f;
        for (int i = 0; i < S; ++i) sum += alpha[i];

        int sem_label = 0;
        for (int i = 1; i < S; ++i) {
            if (alpha[i] > alpha[sem_label])
                sem_label = i;
        }

        float max_p = alpha[sem_label] / sum;
        if (max_p < p_threshold)
            sem_label = 0; // "no class" for uncertain voxels


        // Color according to color map and shading factor
        float factor = expf(-k * (depth - min_range));

        RGB color;
        color.r = color_map_scenenet[sem_label][0] * factor;
        color.g = color_map_scenenet[sem_label][1] * factor;
        color.b = color_map_scenenet[sem_label][2] * factor;

        outImage[offset] = (color.r / 255.0);
        outImage[w*h + offset] = (color.g / 255.0);
        outImage[2*w*h + offset] = (color.b / 255.0);
    }


    // FOR OPEN-SET
    template <uint16_t S>
    inline __hostdev__ void operator()(float* outImage, int i, int w, int h, nanovdb::math::VecXf<S>& m, float depth, const float k, const float p_threshold,
                                       const float* embeddings, const float tsdf_weight, const nanovdb::math::VecXf<S>& beta, const uint16_t num_open_semantic_classes) const
    {
        int offset;
#if 0
        uint32_t x, y;
        mortonDecode(i, x, y);
        offset = x + y * w;
#else
        offset = i;
#endif

        float lambda = tsdf_weight;
        float nu = tsdf_weight / 2;

        // Compute log posterior predictive per class
        constexpr int MAX_CLASSES = 64;
        float logp[MAX_CLASSES]; // placeholder because it must be known at compile time
        float max_logp = -1e30f;

        for (int c = 0; c < num_open_semantic_classes; ++c) {
            // compute log predictive Student-t likelihood
            float lp = 0.f;
            const float dof = 2.0f * nu;

            lp += -0.5f * S * logf(dof * M_PI);
            for (int j = 0; j < S; ++j) {
                float s2 = beta[j] * (lambda + 1.0f) / (lambda * nu);
                float x  = (embeddings[c * S + j] - m[j]) / sqrtf(s2);
                lp += -0.5f * (dof + 1.f) * log1pf((x * x) / dof);
                // (dropped constants since they cancel in softmax)
            }
            logp[c] = lp;
            if (lp > max_logp)
                max_logp = lp;
        }

        // Softmax normalization for posterior probabilities
        float denom = 0.f;
        for (int c = 0; c < num_open_semantic_classes; ++c)
            denom += expf(logp[c] - max_logp);

        // Most probable class
        int best_class = 0;
        float best_prob = 0.0;
        float p = 0.0;
        for (int c = 0; c < num_open_semantic_classes; ++c) {
            p = expf(logp[c] - max_logp) / denom;
            if (p > best_prob) {
                best_prob = p;
                best_class = c;
            }
        }

        // Threshold probabilities
        if (best_prob < p_threshold)
            best_class = 0; // "unknown"


        // Make image for render
        float min_range = 0.1;
        float factor = expf(-k * (depth - min_range));

        RGB color;

        color.r = color_map_scenenet[best_class][0];
        color.g = color_map_scenenet[best_class][1];
        color.b = color_map_scenenet[best_class][2];

        outImage[offset] = color.r / 255.0;
        outImage[w*h + offset] = color.g / 255.0;
        outImage[2*w*h + offset] = color.b / 255.0;
    }

    inline __hostdev__ void operator()(float* outImage, int i, int w, int h, int bg) const
    {
        int offset;
#if 0
        uint32_t x, y;
        mortonDecode(i, x, y);
        offset = x + y * w;
#else
        offset = i;
#endif

        outImage[offset] = bg; // should be 0
        outImage[w*h + offset] = bg;
        outImage[2*w*h + offset] = bg;
    }
};