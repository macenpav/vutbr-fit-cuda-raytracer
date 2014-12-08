#include <iostream>
#include "constants.h"

#include "ray.h"
#include "sphere.h"
#include "mathematics.h"
#include "camera.h"
#include "plane.h"

#include "scene.h"
#include "phong.h"

#include "bvh.h"

__constant__ Camera cst_camera;
__constant__ Sphere cst_spheres[NUM_SPHERES];
__constant__ PointLight cst_lights[NUM_LIGHTS];
__constant__ Plane cst_planes[NUM_PLANES];
__constant__ PhongMaterial cst_materials[NUM_MATERIALS];

using namespace CUDA;

/**
* Checks for error and if found writes to cerr and exits program. 
*/
void checkCUDAError()
{
	cudaError_t err = cudaGetLastError();
	if(cudaSuccess != err)
	{
		std::cerr << "Cuda error: " << cudaGetErrorString(err) << std::endl;
		exit(EXIT_FAILURE);
	}
}


__device__ HitInfo intersectRayWithScene(Ray const& ray, cuBVHnode* tree)
{
	HitInfo hitInfo, hit;

	float st = FLT_MAX;
	float pt = FLT_MAX;
	int maxPi = NO_HIT;
	int maxSi = NO_HIT;

	uint32 i;
	// SPHERES
#ifdef USE_BVH
		while (true) {
			if (!tree->prev && !tree->next) {
				for (uint32 i = 0; i < SPLIT_LIMIT; i++) {
					hit = tree->leaves[i].sphere.intersect(ray);
					if (hit.hit)
					{
						if (st > hit.t)
						{
							st = hit.t;
							maxSi = i;
						}	
					}
				}
				break;
			}

			
			if (tree->prev) {
				hit = tree->prev->intersect(ray);
				if (hit.hit) {
					tree = tree->prev;
					continue;
				}
			}
			if (tree->next) {
				hit = tree->next->intersect(ray);
				if (hit.hit) {
					tree = tree->next;
					continue;
				}
			}
			break;
		}
			
		
		
#else
	for (i = 0; i < NUM_SPHERES; ++i)
	{
		hit = cst_spheres[i].intersect(ray);
		if (hit.hit)
		{
			if (st > hit.t)
			{
				st = hit.t;
				maxSi = i;
			}	
		}
	}
#endif
	for (i = 0; i < NUM_PLANES; ++i){
		hit = cst_planes[i].intersect(ray);
		if (hit.hit){
			if (pt > hit.t){
				pt = hit.t;
				maxPi = i;
			}			
		}
	}
	// miss
	if ((maxPi<0) && (maxSi < 0))
	{ 		
		hitInfo.hit = false;
		return hitInfo;		
	}
	// PLANE hit
	else if (pt < st)
	{
		hitInfo.t = pt;
		hitInfo.point = ray.getPoint(pt);
		hitInfo.normal = cst_planes[maxPi].normal;		
		hitInfo.materialId = cst_planes[maxPi].materialId;
	}
	// SPHERE hit
	else if (st < pt)
	{
		hitInfo.t = st;
		hitInfo.point = ray.getPoint(st);

#ifdef USE_BVH
		hitInfo.normal = tree->leaves[maxSi].sphere.getNormal(hitInfo.point);		
		hitInfo.materialId = tree->leaves[maxSi].sphere.materialId;
#else
		hitInfo.normal = cst_spheres[maxSi].getNormal(hitInfo.point);		
		hitInfo.materialId = cst_spheres[maxSi].materialId;
#endif
	}
	hitInfo.hit = true;
	return hitInfo;	
}


__device__ Color TraceRay(const Ray &ray, int recursion, cuBVHnode* tree)
{
	Color color; color.set(0.f, 0.f, 0.f);

	HitInfo hitInfo = intersectRayWithScene(ray, tree);
	if (hitInfo.hit)
	{				
		const int matID = hitInfo.materialId;
		color = cst_materials[matID].ambient;
		const float3 hitPoint = hitInfo.point;		
		const float3 hitNormal = hitInfo.normal;
		for (uint32 i = 0; i < NUM_LIGHTS; ++i)
		{	

			const float3 lightPos = cst_lights[i].position;		
			//const float3 shadowDir = CUDA::normalize(CUDA::float3_sub(lightPos, hitPoint));
			const float3 shadowDir = cst_lights[i].getShadowRay(hitPoint).direction;

			float intensity = fabs(CUDA::dot(hitNormal, shadowDir));		

			//if (true /*intensity > 0.f*/) { // only if there is enought light
			Ray lightRay = Ray(cst_lights[i].position, CUDA::float3_sub(hitPoint, lightPos));

			HitInfo shadowHit = intersectRayWithScene(lightRay, tree);

			if ((shadowHit.hit) && (fabs(shadowHit.t - CUDA::length(CUDA::float3_sub(hitPoint, lightPos))) < 0.05f)) 
				//if ((shadowHit.hit) && (shadowHit.t < CUDA::length(CUDA::float3_sub(hitPoint, lightPos)) + 0.0001f)) 
			{
				color.accumulate(CUDA::mult(cst_materials[hitInfo.materialId].diffuse, cst_lights[i].color), intensity);

				if (cst_materials[matID].shininess > 0.f) {
					float3 shineDir = CUDA::float3_sub(shadowDir, CUDA::float3_mult(2.0f * CUDA::dot(shadowDir, hitNormal), hitNormal));
					intensity = CUDA::dot(shineDir, ray.direction);				
					intensity = pow(intensity, cst_materials[matID].shininess);					
					intensity = min(intensity, 10000.0f);

					color.accumulate(mult(cst_materials[matID].specular, cst_lights[i].color), intensity);
				}
			}
			//}
		}
		//reflected ray
		if ((cst_materials[matID].reflectance>0) && (recursion > 0)) {
			Ray rray(hitPoint, float3_sub(ray.direction, float3_mult(2*CUDA::dot(ray.direction,hitInfo.normal) ,hitInfo.normal)));
			rray.ShiftStart(1e-5);


			Color rcolor = TraceRay(rray, recursion-1, tree);
			//        color *= 1-phong.GetReflectance();
			color.accumulate(rcolor, cst_materials[matID].reflectance);
		}	
	}

	return color;
}	

/**
* CUDA kernel
*
* @param uchar* data
* @param uint32 width
* @param uint32 height
* @param float time
*/


__global__ void RTKernel(uchar3* data, cuBVHnode* tree, uint32 width, uint32 height)
{
#ifdef BILINEAR_SAMPLING
	__shared__ Color presampled[64];

	uint32 X = SUB_CONST*((blockIdx.x * blockDim.x) + threadIdx.x - blockIdx.x);
	uint32 Y = SUB_CONST*((blockIdx.y * blockDim.y) + threadIdx.y - blockIdx.y);

	if ((X >= (WINDOW_WIDTH-1+SUB_CONST)) || (Y >=(WINDOW_HEIGHT-1+SUB_CONST)))
	{	//potrebuju spocitat i ty co uz jsou za hranou abych mohl dopocitat poslednich 1 až SUB_CONST bodu do konce obrazovky maximalne X = 798 + 4 
		return;
	}
	float x = (2.f*X/WINDOW_WIDTH - 1.f);
	float y = (2.f*Y/WINDOW_HEIGHT - 1.f);

	Color c = TraceRay(cst_camera.getRay(x, y), 15, tree);

	uint32 spos = threadIdx.x + (threadIdx.y * THREADS_PER_BLOCK);

	presampled[spos].red = c.red;
	presampled[spos].green = c.green;
	presampled[spos].blue = c.blue;	

	if ((threadIdx.x == THREADS_PER_BLOCK-1) || (threadIdx.y == THREADS_PER_BLOCK-1)) // posledni sloupec radek je spocitan z predesleho
	{
		return;
	}

	__syncthreads();

	Color c0 = presampled[spos];
	Color c1 = presampled[spos+1];
	Color c2 = presampled[spos+8];
	Color c3 = presampled[spos+9];

	if ((X >= (WINDOW_WIDTH)) || (Y >=(WINDOW_HEIGHT)))
	{	//krajni uz pocitat nemusim
		return;
	}

	uint32 pos = WINDOW_WIDTH * (Y) + (X);

	for (uint32 i = 0, float k = 0.f; i < SUB_CONST; i++, k += 1.f / (SUB_CONST-1))
	{
		for (uint32 j = 0, float l = 0.f; j < SUB_CONST; j++, l += 1.f / (SUB_CONST-1))
		{
			uint32 p = pos+i+j*WINDOW_WIDTH;

			float w1 = (1-k) * (1-l);
			float w2 = k * (1-l);
			float w3 = (1-k) * l;
			float w4 = k*l;


			data[p].x = min( ( w1 * c0.red + w2 * c1.red + w3 * c2.red + w4 * c3.red ) * 255.f, 255.f);
			data[p].y = min( ( w1 * c0.green + w2 * c1.green + w3 * c2.green + w4 * c3.green ) * 255.f, 255.f);
			data[p].z =min( ( w1 * c0.blue + w2 * c1.blue + w3 * c2.blue + w4 * c3.blue ) * 255.f, 255.f);
		}
	}

#else

	uint32 X = (blockIdx.x * blockDim.x) + threadIdx.x;
	uint32 Y = (blockIdx.y * blockDim.y) + threadIdx.y;

	float x = (2.f*X/WINDOW_WIDTH - 1.f);
	float y = (2.f*Y/WINDOW_HEIGHT - 1.f);


	Color c = TraceRay(cst_camera.getRay(x, y), 15, tree);

	uint32 p = Y * WINDOW_WIDTH + X;

	data[p].x = min(c.red * 255.f, 255.f);
	data[p].y = min(c.green * 255.f, 255.f);
	data[p].z = min(c.blue * 255.f, 255.f);

#endif
}


/**
* Wrapper for the CUDA kernel
*
* @param uchar* data
* @param uint32 width
* @param uint32 height
* @param float time
*/
extern "C" void launchRTKernel(uchar3* data, uint32 imageWidth, uint32 imageHeight, Sphere* spheres, Plane* planes, PointLight* lights, PhongMaterial* materials, Camera* camera, cuBVHnode* tree)
{   
#ifdef BILINEAR_SAMPLING
	dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1); // 64 threads ~ 8*8 -> based on this shared memory for sampling is allocated !!!

	int blocksx = WINDOW_WIDTH / SUB_CONST / (threadsPerBlock.x-1);
	int blocksy = WINDOW_HEIGHT / SUB_CONST / (threadsPerBlock.y-1);
	blocksx = blocksx + ceil(float(blocksx) / THREADS_PER_BLOCK); //zjistit kolik tam je 
	blocksy = blocksy + ceil(float(blocksy) / THREADS_PER_BLOCK);

	dim3 numBlocks(blocksx,blocksy);
#else
	dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1); // 64 threads ~ 8*8 -> based on this shared memory for sampling is allocated !!!
	dim3 numBlocks(WINDOW_WIDTH / threadsPerBlock.x, WINDOW_HEIGHT / threadsPerBlock.y);
#endif

	cudaMemcpyToSymbol(cst_camera, camera, sizeof(Camera));
	cudaMemcpyToSymbol(cst_spheres, spheres, NUM_SPHERES * sizeof(Sphere));
	cudaMemcpyToSymbol(cst_planes, planes, NUM_PLANES * sizeof(Plane));
	cudaMemcpyToSymbol(cst_lights, lights, NUM_LIGHTS * sizeof(PointLight));
	cudaMemcpyToSymbol(cst_materials, materials, NUM_MATERIALS * sizeof(PhongMaterial));	


	RTKernel<<<numBlocks, threadsPerBlock>>>(data, tree, imageWidth, imageHeight);


	cudaThreadSynchronize();

	checkCUDAError();		
}
