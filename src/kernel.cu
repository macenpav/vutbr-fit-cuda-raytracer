#include <iostream>
#include "constants.h"

#include "ray.h"
#include "sphere.h"
#include "mathematics.h"
#include "camera.h"
#include "plane.h"

#include "scene.h"
#include "phong.h"
#include "proceduraltexture.h"

#include "bvh.h"
#ifdef BUILD_WITH_KDTREE
	#include "kdtree.h"
#endif

__constant__ Camera cst_camera;
__constant__ Sphere cst_spheres[NUM_SPHERES];
__constant__ PointLight cst_lights[NUM_LIGHTS];
__constant__ Plane cst_planes[NUM_PLANES];
__constant__ Cylinder cst_cylinders[NUM_CYLINDERS];
__constant__ Triangle cst_triangles[NUM_TRIANGLES];

__constant__ PhongMaterial cst_materials[NUM_MATERIALS];
__constant__ Plane cst_FocalPlane;

using namespace CUDA;

#define DEBUG

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

#ifdef BUILD_WITH_KDTREE
namespace CUDA {
	__device__ bool recursiveIntersect(HitInfo& hitData, Ray const& ray, KDNode* kdTree)
	{		
		if (kdTree->intersect(ray))
		{			
			if (!kdTree->left && !kdTree->right)
			{				
				
				float t = FLT_MAX;
				int32 sphereId = NO_HIT;

				for (uint32 i = 0; i < kdTree->countSpheres; ++i)
				{
					HitInfo currentHit = kdTree->spheres[i].intersect(ray);
					if (currentHit.hit && currentHit.t < t)
						hitData = currentHit;						
										
				}				
				return hitData.hit;
				
			}
			else
			{
				bool leftHit = recursiveIntersect(hitData, ray, kdTree->left);
				bool rightHit = recursiveIntersect(hitData, ray, kdTree->right);
				
				return leftHit || rightHit;
			}

		}	
		return false;
	}
}
#endif

__device__ HitInfo intersectRayWithScene(Ray const& ray, void* accelerationStructure)
{
	HitInfo hitInfo, hit;

	float st = FLT_MAX;
	float pt = FLT_MAX;
	float ct = FLT_MAX;
	float tt = FLT_MAX;
	int maxPi = NO_HIT;
	int maxSi = NO_HIT;
	int maxCi = NO_HIT;
	int maxTi = NO_HIT;

#if defined BUILD_WITH_BVH
		cuBVHnode* bvhTree = (cuBVHnode*) accelerationStructure;

		while (true) {
			if (!bvhTree->prev && !bvhTree->next) {
				for (uint32 i = 0; i < SPLIT_LIMIT; i++) {
					hit = bvhTree->leaves[i].sphere.intersect(ray);
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

			
			if (bvhTree->prev) {
				hit = bvhTree->prev->intersect(ray);
				if (hit.hit) {
					bvhTree = bvhTree->prev;
					continue;
				}
			}
			if (bvhTree->next) {
				hit = bvhTree->next->intersect(ray);
				if (hit.hit) {
					bvhTree = bvhTree->next;
					continue;
				}
			}
			break;
		}
#elif defined BUILD_WITH_KDTREE	
	CUDA::KDNode* kdTree = (CUDA::KDNode*) accelerationStructure;

	HitInfo kdHit;
	if (CUDA::recursiveIntersect(kdHit, ray, kdTree))
	{
		st = kdHit.t;
		maxSi = kdHit.sphereId;
	}
#else
	for (uint32 i = 0; i < NUM_SPHERES; ++i)
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
	for (uint32 i = 0; i < NUM_PLANES; ++i){
		hit = cst_planes[i].intersect(ray);
		if (hit.hit){
			if (pt > hit.t){
				pt = hit.t;
				maxPi = i;
			}			
		}
	}
	for (uint32 i = 0; i < NUM_CYLINDERS; ++i){
		hit = cst_cylinders[i].intersect(ray);
		if (hit.hit){
			if (ct > hit.t){
				ct = hit.t;
				maxCi = i;
			}
			
		}
	
	}
	for (uint32 i = 0; i < NUM_TRIANGLES; ++i){
		hit = cst_triangles[i].intersect(ray);
		if (hit.hit){
			if (tt > hit.t){
				tt = hit.t;
				maxTi = i;
			}
		}
	}


	// miss
	if ((maxPi == NO_HIT) && (maxSi == NO_HIT) && (maxCi == NO_HIT) && (maxTi == NO_HIT))
	{
		hitInfo.hit = false;
		return hitInfo;
	}
	// PLANE hit
	else if ((pt < st) && (pt < ct) && (pt < tt))
	{
		hitInfo.t = pt;
		hitInfo.point = ray.getPoint(pt);
		hitInfo.normal = cst_planes[maxPi].normal;
		hitInfo.materialId = cst_planes[maxPi].materialId;
	}
	//CYLINDER hit
	else if ((ct < st) && (ct < pt) && (ct < tt)){
		hitInfo.t = ct;
		hitInfo.point = ray.getPoint(ct);
		hitInfo.normal = cst_cylinders[maxCi].getNormal(hitInfo.point);
		hitInfo.materialId = cst_cylinders[maxCi].materialId;
	}//TRIANGLE hit
	else if ((tt < st) && (tt < ct) && (tt < pt))
	{
		hitInfo.t = tt;
		hitInfo.point = ray.getPoint(tt);
		hitInfo.normal = cst_triangles[maxTi].normal;
		hitInfo.materialId = cst_triangles[maxTi].materialId;
	}
	// SPHERE hit
	else if ((st < pt) && (st < ct))
	{
		hitInfo.t = st;
		hitInfo.point = ray.getPoint(st);

#if defined BUILD_WITH_BVH
		hitInfo.normal = bvhTree->leaves[maxSi].sphere.getNormal(hitInfo.point);		
		hitInfo.materialId = bvhTree->leaves[maxSi].sphere.materialId;
#elif defined BUILD_WITH_KDTREE
		hitInfo.normal = cst_spheres[maxSi].getNormal(hitInfo.point);
		hitInfo.materialId = cst_spheres[maxSi].materialId;
#else
		hitInfo.normal = cst_spheres[maxSi].getNormal(hitInfo.point);		
		hitInfo.materialId = cst_spheres[maxSi].materialId;
#endif
	}
	hitInfo.hit = true;
	return hitInfo;	
}

__device__ float IntensityMult(float3 lightPos, float3 hitPoint, void* accelerationStructure)
{
		const int shadowrayCount = 19;
		float3 origin[shadowrayCount];
		float value = 0;
		origin[0] =  make_float3(lightPos.x + LIGHTRADIUS, lightPos.y, lightPos.z); //umisteni vychozich bodu paprsku okolo primarniho paprsku
		origin[1] =  make_float3(lightPos.x - LIGHTRADIUS, lightPos.y, lightPos.z); 
		origin[2] =  make_float3(lightPos.x , lightPos.y + LIGHTRADIUS, lightPos.z); 
		origin[3] =  make_float3(lightPos.x , lightPos.y - LIGHTRADIUS, lightPos.z); 
		origin[4] =  make_float3(lightPos.x , lightPos.y, lightPos.z + LIGHTRADIUS); 
		origin[5] =  make_float3(lightPos.x , lightPos.y, lightPos.z - LIGHTRADIUS); 

		origin[6] =  make_float3(lightPos.x , lightPos.y, lightPos.z); 

		origin[7] =  make_float3(lightPos.x + LIGHTRADIUS/2.f, lightPos.y, lightPos.z); //umisteni vychozich bodu paprsku okolo primarniho paprsku
		origin[8] =  make_float3(lightPos.x - LIGHTRADIUS/2.f, lightPos.y, lightPos.z); 
		origin[9] =  make_float3(lightPos.x , lightPos.y + LIGHTRADIUS/2.f, lightPos.z); 
		origin[10] =  make_float3(lightPos.x , lightPos.y - LIGHTRADIUS/2.f, lightPos.z); 
		origin[11] =  make_float3(lightPos.x , lightPos.y, lightPos.z + LIGHTRADIUS/2.f); 
		origin[12] =  make_float3(lightPos.x , lightPos.y, lightPos.z - LIGHTRADIUS/2.f);
		
		origin[13] =  make_float3(lightPos.x + LIGHTRADIUS*2.f, lightPos.y, lightPos.z); //umisteni vychozich bodu paprsku okolo primarniho paprsku
		origin[14] =  make_float3(lightPos.x - LIGHTRADIUS*2.f, lightPos.y, lightPos.z); 
		origin[15] =  make_float3(lightPos.x , lightPos.y + LIGHTRADIUS*2.f, lightPos.z); 
		origin[16] =  make_float3(lightPos.x , lightPos.y - LIGHTRADIUS*2.f, lightPos.z); 
		origin[17] =  make_float3(lightPos.x , lightPos.y, lightPos.z + LIGHTRADIUS*2.f); 
		origin[18] =  make_float3(lightPos.x , lightPos.y, lightPos.z - LIGHTRADIUS*2.f); 

		for (int i =0; i<shadowrayCount; i++){
			Ray r = Ray(origin[i], CUDA::float3_sub(hitPoint,origin[i]));
			
			HitInfo shadowHit = intersectRayWithScene(r, accelerationStructure);
			if ((shadowHit.hit) && (fabs(shadowHit.t - CUDA::length(CUDA::float3_sub(hitPoint, origin[i]))) < 0.001f)) 
			{
				value += 1.f/shadowrayCount;
			}
			
		}
		return value;
	
}


__device__ Color TraceRay(const Ray &ray, int recursion, void* accelerationStructure)
{
	Color color; color.set(0.f, 0.f, 0.f);

	HitInfo hitInfo = intersectRayWithScene(ray, accelerationStructure);
	if (hitInfo.hit)
	{				
		const float3 hitPoint = hitInfo.point;
		int matID;
		if (hitInfo.materialId == MATERIAL_CHECKER)
		{
			matID  = CheckerProcedural(MATERIAL_WHITE, MATERIAL_BLACK, hitPoint);
		} else
		{
			 matID = hitInfo.materialId;
		}
		

		color = cst_materials[matID].ambient;		
		const float3 hitNormal = hitInfo.normal;
		for (uint32 i = 0; i < NUM_LIGHTS; ++i)
		{	

			const float3 lightPos = cst_lights[i].position;		
			//const float3 shadowDir = CUDA::normalize(CUDA::float3_sub(lightPos, hitPoint));
			const float3 shadowDir = cst_lights[i].getShadowRay(hitPoint).direction;

			float intensity = fabs(CUDA::dot(hitNormal, shadowDir));
#ifdef SOFTSHADOWS
			intensity = intensity * IntensityMult(lightPos,hitPoint,accelerationStructure);
			if (intensity > 0.f){
#endif


#ifndef SOFTSHADOWS
			//if (true /*intensity > 0.f*/) { // only if there is enought light
			Ray lightRay = Ray(cst_lights[i].position, CUDA::float3_sub(hitPoint, lightPos));

			HitInfo shadowHit = intersectRayWithScene(lightRay, accelerationStructure);

			if ((shadowHit.hit) && (fabs(shadowHit.t - CUDA::length(CUDA::float3_sub(hitPoint, lightPos))) < 0.01f)) 
				//if ((shadowHit.hit) && (shadowHit.t < CUDA::length(CUDA::float3_sub(hitPoint, lightPos)) + 0.0001f)) 
			{
#endif
				color.accumulate(CUDA::mult(cst_materials[matID].diffuse, cst_lights[i].color), intensity);

				if (cst_materials[matID].shininess > 0.f) {
					float3 shineDir = CUDA::float3_sub(shadowDir, CUDA::float3_mult(2.0f * CUDA::dot(shadowDir, hitNormal), hitNormal));
					intensity = CUDA::dot(shineDir, ray.direction);				
					intensity = pow(intensity, cst_materials[matID].shininess);					
					intensity = min(intensity, 10000.0f);

					color.accumulate(CUDA::mult(cst_materials[matID].specular, cst_lights[i].color), intensity);
				}

			}

			//}
		}
		//reflected ray
		if ((cst_materials[matID].reflectance>0) && (recursion > 0)) {
			Ray rray(hitPoint, float3_sub(ray.direction, float3_mult(2*CUDA::dot(ray.direction,hitInfo.normal) ,hitInfo.normal)));
			rray.ShiftStart(1e-5);


			Color rcolor = TraceRay(rray, recursion-1, accelerationStructure);
			//        color *= 1-phong.GetReflectance();
			color.accumulate(rcolor, cst_materials[matID].reflectance);
		}	
	}

	return color;
}	


__device__ Color DepthOfFieldRayTrace(const Ray &ray, int recursion, void* accelerationStructure) 
{
	HitInfo hit;
	Color c;
	c.set(0,0,0);
	c.accumulate(TraceRay(ray,recursion,accelerationStructure),0.20);
	hit = cst_FocalPlane.intersect(ray);
	
	if (hit.hit) {
		hit.point = ray.getPoint(hit.t);
		float3 origin1,origin2,origin3,origin4;
		origin1 =  make_float3(ray.origin.x+LENSRADIUS,ray.origin.y, ray.origin.z); //umisteni vychozich bodu paprsku okolo primarniho paprsku
		origin2 =  make_float3(ray.origin.x-LENSRADIUS,ray.origin.y, ray.origin.z);
		origin3 =  make_float3(ray.origin.x,ray.origin.y+LENSRADIUS, ray.origin.z);
		origin4 =  make_float3(ray.origin.x,ray.origin.y-LENSRADIUS, ray.origin.z);

		Ray r1,r2,r3,r4;
		r1 = Ray(origin1, CUDA::float3_sub(hit.point,origin1));  //paprsky se potkaji v miste hloubky ostrosti
		r2 = Ray(origin2, CUDA::float3_sub(hit.point,origin2));
		r3 = Ray(origin3, CUDA::float3_sub(hit.point,origin3));
		r4 = Ray(origin4, CUDA::float3_sub(hit.point,origin4));

		Color c1,c2,c3,c4;
		c1 = TraceRay(r1, recursion, accelerationStructure);
		c2 = TraceRay(r2, recursion, accelerationStructure);
		c3 = TraceRay(r3, recursion, accelerationStructure);
		c4 = TraceRay(r4, recursion, accelerationStructure);

		c.accumulate(c1,0.20);
		c.accumulate(c2,0.20);
		c.accumulate(c3,0.20);
		c.accumulate(c4,0.20);
	}
	return c;
}

/**
* CUDA kernel
*
* @param uchar* data
* @param uint32 width
* @param uint32 height
* @param float time
*/


__global__ void RTKernel(uchar3* data, void* accelerationStructure, uint32 width, uint32 height)
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

	Color c = TraceRay(cst_camera.getRay(x, y), 15, accelerationStructure);

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

	Ray r = cst_camera.getRay(x, y);
	Color c;

#ifndef DEPTHOFFIELD
	c = TraceRay(r,15,accelerationStructure);
#endif

#ifdef DEPTHOFFIELD
	c = DepthOfFieldRayTrace(r,15,accelerationStructure);
	
#endif
	

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
extern "C" void launchRTKernel(uchar3* data, uint32 imageWidth, uint32 imageHeight, Sphere* spheres, Plane* planes, Cylinder* cylinders,Triangle* triangles, PointLight* lights, PhongMaterial* materials, Camera* camera, Plane* focalPlane, void* accelerationStructure)
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
	cudaMemcpyToSymbol(cst_cylinders, cylinders, NUM_CYLINDERS * sizeof(Cylinder));
	cudaMemcpyToSymbol(cst_triangles, triangles, NUM_TRIANGLES * sizeof(Triangle));

	cudaMemcpyToSymbol(cst_FocalPlane, focalPlane, sizeof(Plane));	

	RTKernel <<<numBlocks, threadsPerBlock>>>(data, accelerationStructure, imageWidth, imageHeight);

	cudaThreadSynchronize();

	checkCUDAError();		
}
