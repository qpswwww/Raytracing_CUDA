# A Parallel Ray Tracing Render Based on CUDA

## Project Info

**The course project of *Computer Graphics* and *the Design of Large-scale Software***

**Team member:** Yongkang Zhang

In this project, I implemented a parallel ray tracing render accelarated by NVIDIA CUDA. The algorithm was also optimized by a BVH (Boundary Volume Hierarchy) tree with SAH (Surface Area Heuristic). 

**This project is still ongoing.**


## Milestone

- **2019/10/4:** New feature: Capability of rendering Wavefront .obj format files!

![](https://qpswwww.github.io/Raytracing_CUDA/milestone2_1.png)

![](https://qpswwww.github.io/Raytracing_CUDA/milestone2_2.png)

- **2019/10/3:** Successfully adapt the render's code to CUDA and accelerate the code!

![](https://qpswwww.github.io/Raytracing_CUDA/milestone1.png)

## Experiments


teapot.obj, dieclectic(1.5), 900x450 resolution, ns=5, depth=50

CPU, wo. BVH: 109125ms

CPU, w. simplest BVH: 9835ms

CPU, w. BVH+SAH: 9605ms

CPU, w. BVH+SAH(non-recursive): 8962ms

## Acknowledgement

Thank Peter Shirley for providing detailed tutorials of ray tracing and its implementation in C++! [^oneweekend]

## References

[^oneweekend]: Ray Tracing in One Weekend. https://raytracing.github.io/
