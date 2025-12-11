// Apache License
//
// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
// License link: https://github.com/AcademySoftwareFoundation/openvdb/blob/master/LICENSE
//
// Modified by Anja Sheppard, University of Michigan, 2025


#define _USE_MATH_DEFINES

#include <nanovdb/io/IO.h>
#include <nanovdb/math/HDDA.h>
#include <nanovdb/math/Ray.h>
#include <nanovdb/util/GridBuilder.h>

#include <chrono>
#include <cmath>
#include <opencv2/opencv.hpp>

#include "common.h"

#if defined(NANOVDB_USE_CUDA)
#include <nanovdb/cuda/DeviceBuffer.h>
using BufferT = nanovdb::cuda::DeviceBuffer;
#else
using BufferT = nanovdb::HostBuffer;
#endif

void runNanoVDB(nanovdb::GridHandle<BufferT>& handle,
                nanovdb::GridHandle<BufferT>& label_handle,
                int width,
                int height,
                BufferT& imageBuffer,
                int index,
                const std::vector<double> origin,
                const std::vector<double> quaternion,
                const float min_range,
                const float max_range) {
    constexpr int num_semantic_classes = NCLASSES; // TODO FIX
    using GridT = nanovdb::FloatGrid;
    using LabelGridT = nanovdb::VecXIGrid<num_semantic_classes>;
    using CoordT = nanovdb::Coord;
    using RealT = float;
    using Vec3T = nanovdb::math::Vec3<RealT>;
    using RayT = nanovdb::math::Ray<RealT>;

    auto* h_grid = handle.grid<float>();
    auto* h_label_grid = label_handle.grid<nanovdb::math::VecXi<num_semantic_classes>>();
    if (!h_grid) throw std::runtime_error("GridHandle does not contain a valid host grid");
    if (!h_label_grid)
        throw std::runtime_error("GridHandle does not contain a valid host label grid");

    float* h_outImage = reinterpret_cast<float*>(imageBuffer.data());

#if defined(NANOVDB_USE_CUDA)
    double* d_origin;
    cudaMalloc((void**)&d_origin, 3 * sizeof(double));
    cudaMemcpy(d_origin, origin.data(), 3 * sizeof(double), cudaMemcpyHostToDevice);

    double* d_quaternion;
    cudaMalloc((void**)&d_quaternion, 4 * sizeof(double));
    cudaMemcpy(d_quaternion, quaternion.data(), 4 * sizeof(double), cudaMemcpyHostToDevice);
#else
    double d_origin[3] = {origin[0], origin[1], origin[2]};
    double d_quaternion[4] = {quaternion[0], quaternion[1], quaternion[3], quaternion[4]};
#endif

    float wBBoxDimZ = (float)h_grid->worldBBox().dim()[2] * 2;
    Vec3T wBBoxCenter = Vec3T(h_grid->worldBBox().min() + h_grid->worldBBox().dim() * 0.5f);
    nanovdb::CoordBBox treeIndexBbox = h_grid->tree().bbox();
    std::clog << "Bounds: "
              << "[" << treeIndexBbox.min()[0] << "," << treeIndexBbox.min()[1] << ","
              << treeIndexBbox.min()[2] << "] -> [" << treeIndexBbox.max()[0] << ","
              << treeIndexBbox.max()[1] << "," << treeIndexBbox.max()[2] << "]" << std::endl;

    RayGenOp<Vec3T> rayGenOp(wBBoxDimZ, wBBoxCenter);
    CompositeOp compositeOp;

    // Calculate factor for shading depth in the rendered image
    const float f_min = 0.7; // you can tune this parameter if you want to change how dark the darkest part of the rendered image is (0.3 = 30% of full brightness)
    const float k = -logf(f_min) / (max_range - min_range);

    auto renderOp = [width, height, rayGenOp, compositeOp, treeIndexBbox, wBBoxDimZ,
                     d_origin, d_quaternion, k, min_range] __hostdev__(int start, int end, float* image, const GridT* grid,
                                           const LabelGridT* label_grid) {
        // get an accessor.
        auto acc = grid->tree().getAccessor();
        auto label_acc = label_grid->tree().getAccessor();

        for (int i = start; i < end; ++i) {
            Vec3T rayEye;
            Vec3T rayDir;
            rayGenOp(i, width, height, rayEye, rayDir);

            double x = d_quaternion[0];
            double y = d_quaternion[1];
            double z = d_quaternion[2];
            double w = d_quaternion[3];

            // Compute intermediate values
            double uv0 = y * rayDir[2] - z * rayDir[1];
            double uv1 = z * rayDir[0] - x * rayDir[2];
            double uv2 = x * rayDir[1] - y * rayDir[0];

            double uuv0 = y * uv2 - z * uv1;
            double uuv1 = z * uv0 - x * uv2;
            double uuv2 = x * uv1 - y * uv0;

            // Apply rotation
            rayDir[0] = rayDir[0] + 2.0 * (w * uv0 + uuv0);
            rayDir[1] = rayDir[1] + 2.0 * (w * uv1 + uuv1);
            rayDir[2] = rayDir[2] + 2.0 * (w * uv2 + uuv2);

            rayEye[0] = d_origin[0];
            rayEye[1] = d_origin[1];
            rayEye[2] = d_origin[2];

            // generate ray.
            RayT wRay(rayEye, rayDir);

            // transform the ray to the grid's index-space.
            RayT iRay = wRay.worldToIndexF(*grid);

            // intersect...
            float t0;
            CoordT ijk;
            float v;

            if (nanovdb::math::ZeroCrossing(iRay, acc, ijk, v, t0)) {
                // write distance to surface. (we assume it is a uniform voxel)
                float wT0 = t0 * float(grid->voxelSize()[0]);
                auto label = label_acc.getValue(ijk);
                compositeOp(image, i, width, height, label, wT0, k, min_range);
            } else {
                // write background value.
                compositeOp(image, i, width, height, 0);
            }
        }
    };

#if defined(NANOVDB_USE_CUDA)
    auto t5 = std::chrono::high_resolution_clock::now();
    handle.deviceUpload();
    label_handle.deviceUpload();

    auto* d_grid = handle.deviceGrid<float>();
    auto* d_label_grid = label_handle.deviceGrid<nanovdb::math::VecXi<num_semantic_classes>>();
    if (!d_grid) throw std::runtime_error("GridHandle does not contain a valid device grid");
    if (!d_label_grid)
        throw std::runtime_error("GridHandle does not contain a valid device label grid");

    imageBuffer.deviceUpload();
    float* d_outImage = reinterpret_cast<float*>(imageBuffer.deviceData());

    auto t6 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed5 = t6 - t5;
    std::clog << "Device Upload took: " << elapsed5.count() << " ms" << std::endl;

    {
        float duration =
            renderImage<num_semantic_classes>(true, renderOp, width, height, d_outImage, d_grid, d_label_grid);
        std::clog << "Duration(NanoVDB-Cuda) = " << duration << " ms" << std::endl;

        auto start3 = std::chrono::high_resolution_clock::now();

        imageBuffer.deviceDownload();
        auto image = (float*)imageBuffer.data();

        // Populate the cv::Mat with the image data
        cv::Mat r_channel(height, width, CV_32F, (void*)(image + 0));
        cv::Mat g_channel(height, width, CV_32F, (void*)(image + width * height));
        cv::Mat b_channel(height, width, CV_32F, (void*)(image + 2 * width * height));

        std::vector<cv::Mat> channels = {b_channel, g_channel, r_channel};  // OpenCV expects BGR
        cv::Mat mat;
        cv::merge(channels, mat);
        cv::flip(mat, mat, 0);
        mat.convertTo(mat, CV_8UC3, 255);

        cv::imshow("Image", mat);
        cv::waitKey(1);

        // cv::imwrite("render_out/" + std::to_string(index) + ".png", mat);

        auto end3 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed3 = end3 - start3;
        std::clog << "Buffer download and displaying the image took: " << elapsed3.count() << " ms"
                  << std::endl;
    }
#else
    {
        float duration =
            renderImage<num_semantic_classes>(false, renderOp, width, height, h_outImage, h_grid, h_label_grid);
        std::clog << "Duration(NanoVDB-Host) = " << duration << " ms" << std::endl;

        auto start3 = std::chrono::high_resolution_clock::now();

        // Create a cv::Mat of size height x width with 3 channels (CV_32FC3) for storing the image.
        cv::Mat mat(height, width, CV_32FC3);

        auto image = (float*)imageBuffer.data();

        // Populate the cv::Mat with the image data
        for (int i = 0; i < width * height; ++i) {
            int y = height - 1 - (i / width);  // Flip the row index (invert y-axis)
            int x = i % width;

            mat.at<cv::Vec3f>(y, x)[0] = image[2 * width * height + i];  // Blue channel
            mat.at<cv::Vec3f>(y, x)[1] = image[width * height + i];      // Green channel
            mat.at<cv::Vec3f>(y, x)[2] = image[i];                       // Red channel
        }

        mat.convertTo(mat, CV_8UC3, 255);

        cv::imshow("Image", mat);
        cv::waitKey(1);

        auto end3 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed3 = end3 - start3;
        std::clog << "Displaying the image took: " << elapsed3.count() << " ms" << std::endl;
    }
#endif
}