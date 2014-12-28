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
#ifdef ACC_KD_TREE
	#include "kdtree.h"
#endif


#include "cuda.h"
#include "device_launch_parameters.h"
#include <cuda_runtime.h>

#include <iostream>
#include "constants.h"

#include <GL/glew.h>
#include <glm/glm.hpp>
#include <glut.h>
#include <cuda_gl_interop.h>

#include "mathematics.h"

#include "scene.h"
#include "sphere.h"
#include "plane.h"
#include "camera.h"
#include "phong.h"

#include <chrono>

#include "bvh.h"
#include "kdtree.h"
#include "objloader.h"

/** @brief primitives stored in constant memory */
__constant__ Camera cst_camera;
__constant__ Sphere cst_spheres[MAX_SPHERES];
__constant__ PointLight cst_lights[MAX_LIGHTS];
__constant__ Plane cst_planes[MAX_PLANES];
__constant__ Cylinder cst_cylinders[MAX_CYLINDERS];
__constant__ Triangle cst_triangles[MAX_TRIANGLES];

/** @brief stats, materials, helpers ... */
__constant__ SceneStats cst_scenestats;
__constant__ PhongMaterial cst_materials[NUM_MATERIALS];
__constant__ Plane cst_focalplane;

#define GPU_ERROR_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		system("pause");
		if (abort) exit(code);
	}
}

static float shiftx = 0.0f;
static float shifty = 7.5f;
static float shiftz = -13.f;
static float num = 0.0f;

void* acceleration_structure;

Scene scene;

/** @var GLuint pixel buffer object */
GLuint PBO;

/** @var GLuint texture buffer */
GLuint textureId;

/** @var cudaGraphicsResource_t cuda data resource */
cudaGraphicsResource_t cudaResourceBuffer;

/** @var cudaGraphicsResource_t cuda texture resource */
cudaGraphicsResource_t cudaResourceTexture;

float deltaTime = 0.0f;
float fps = 0.0f;
float delta;
bool switchb = true;

#ifdef OPT_DEPTH_OF_FIELD
float focalLength = FOCALLENGTH;
#endif

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
		system("pause");
		exit(EXIT_FAILURE);
	}
}

#ifdef ACC_KD_TREE
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

#if defined ACC_BVH
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
#elif defined ACC_KD_TREE	
	CUDA::KDNode* kdTree = (CUDA::KDNode*) accelerationStructure;

	HitInfo kdHit;
	if (CUDA::recursiveIntersect(kdHit, ray, kdTree))
	{
		st = kdHit.t;
		maxSi = kdHit.sphereId;
	}
#else
	for (uint32 i = 0; i < cst_scenestats.sphereCount; ++i)
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
	for (uint32 i = 0; i < cst_scenestats.planeCount; ++i){
		hit = cst_planes[i].intersect(ray);
		if (hit.hit){
			if (pt > hit.t){
				pt = hit.t;
				maxPi = i;
			}			
		}
	}
	for (uint32 i = 0; i < cst_scenestats.cylinderCount; ++i){
		hit = cst_cylinders[i].intersect(ray);
		if (hit.hit){
			if (ct > hit.t){
				ct = hit.t;
				maxCi = i;
			}
			
		}
	
	}	
		
	for (uint32 i = 0; i < cst_scenestats.triangleCount; ++i){
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

#if defined ACC_BVH
		hitInfo.normal = bvhTree->leaves[maxSi].sphere.getNormal(hitInfo.point);		
		hitInfo.materialId = bvhTree->leaves[maxSi].sphere.materialId;
#elif defined ACC_KD_TREE
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

#ifdef OPT_DEPTH_OF_FIELD
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
			
		HitInfo shadowHit = intersectRayWithScene(r, triangles, accelerationStructure);
		if ((shadowHit.hit) && (fabs(shadowHit.t - CUDA::length(CUDA::float3_sub(hitPoint, origin[i]))) < 0.001f)) 
		{
			value += 1.f/shadowrayCount;
		}
			
	}
	return value;	
}
#endif


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
		} 
		else
		{
			matID = hitInfo.materialId;
		}
		

		color = cst_materials[matID].ambient;		
		const float3 hitNormal = hitInfo.normal;
		for (uint32 i = 0; i < cst_scenestats.lightCount; ++i)
		{	

			const float3 lightPos = cst_lights[i].position;		
			//const float3 shadowDir = CUDA::normalize(CUDA::float3_sub(lightPos, hitPoint));
			const float3 shadowDir = cst_lights[i].getShadowRay(hitPoint).direction;

			float intensity = fabs(CUDA::dot(hitNormal, shadowDir));
#ifdef OPT_SOFT_SHADOWS
			intensity = intensity * IntensityMult(lightPos,hitPoint,accelerationStructure);
			if (intensity > 0.f){
#endif


#ifndef OPT_SOFT_SHADOWS
			//if (true /*intensity > 0.f*/) { // only if there is enought light
			Ray lightRay = Ray(cst_lights[i].position, CUDA::float3_sub(hitPoint, lightPos));

			HitInfo shadowHit = intersectRayWithScene(lightRay, accelerationStructure);

			if ((shadowHit.hit) && (fabs(shadowHit.t - CUDA::length(CUDA::float3_sub(hitPoint, lightPos))) < 0.01f)) 
				//if ((shadowHit.hit) && (shadowHit.t < CUDA::length(CUDA::float3_sub(hitPoint, lightPos)) + 0.0001f)) 
			{
#endif
				color.accumulate(cst_materials[matID].diffuse * cst_lights[i].color, intensity);

				if (cst_materials[matID].shininess > 0.f) {
					float3 shineDir = CUDA::float3_sub(shadowDir, CUDA::float3_mult(2.0f * CUDA::dot(shadowDir, hitNormal), hitNormal));
					intensity = CUDA::dot(shineDir, ray.direction);				
					intensity = pow(intensity, cst_materials[matID].shininess);					
					intensity = min(intensity, 10000.0f);

					color.accumulate(cst_materials[matID].specular * cst_lights[i].color, intensity);
				}

			}

			//}
		}
		//reflected ray
		if ((cst_materials[matID].reflectance>0) && (recursion > 0)) {
			Ray rray(hitPoint, float3_sub(ray.direction, float3_mult(2*CUDA::dot(ray.direction,hitInfo.normal) ,hitInfo.normal)));
			rray.ShiftStart(1e-5);


			Color rcolor = TraceRay(rray, recursion - 1, accelerationStructure);
			//        color *= 1-phong.GetReflectance();
			color.accumulate(rcolor, cst_materials[matID].reflectance);
		}	
	}

	return color;
}	


#ifdef OPT_DEPTH_OF_FIELD
__device__ Color DepthOfFieldRayTrace(const Ray &ray, int recursion, void* accelerationStructure) 
{
	HitInfo hit;
	Color c;
	c.set(0,0,0);
	c.accumulate(TraceRay(ray,recursion,accelerationStructure),0.20);
	hit = cst_focalplane.intersect(ray);
	
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
#endif

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
#ifdef OPT_BILINEAR_SAMPLING
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

#ifndef OPT_DEPTH_OF_FIELD
	c = TraceRay(r, 15, accelerationStructure);
#endif

#ifdef OPT_DEPTH_OF_FIELD
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
void launchRTKernel(uchar3* data, uint32 imageWidth, uint32 imageHeight, void* accelerationStructure = nullptr)
{   
#ifdef OPT_BILINEAR_SAMPLING
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
	GPU_ERROR_CHECK(cudaMemcpyToSymbol(cst_camera, scene.getCamera(), sizeof(Camera)));

	RTKernel <<<numBlocks, threadsPerBlock >>>(data, accelerationStructure, imageWidth, imageHeight);

	GPU_ERROR_CHECK(cudaThreadSynchronize());
}

/**
* 1. Maps the the PBO (Pixel Buffer Object) to a data pointer
* 2. Launches the kernel
* 3. Unmaps the PBO
*/
void runCuda()
{
	//static float shiftx = 0.0f;
	//static float shifty = 15.0f;
	//static float shiftz = -3.f;

	uchar3* data;
	size_t numBytes;

	cudaGraphicsMapResources(1, &cudaResourceBuffer, 0);
	// cudaGraphicsMapResources(1, &cudaResourceTexture, 0);
	cudaGraphicsResourceGetMappedPointer((void **)&data, &numBytes, cudaResourceBuffer);

#ifdef CAMERASHIFT
	num += 0.1f;
	shiftx += (sin(num) * 0.1f);
	shifty += (cos(num) * 0.1f);
	shiftz += (sin(num) * 0.1f);
#endif
	Camera* cam = scene.getCamera();

	cam->lookAt(make_float3(shiftx, shifty, shiftz),  // eye
		make_float3(0.f, 0.f, 15.f),   // target
		make_float3(0.f, 1.f, 0.f),   // sky
		30, (float)WINDOW_WIDTH / WINDOW_HEIGHT);

#ifdef OPT_DEPTH_OF_FIELD	
	float3 dir = make_float3(cam->direction.x, cam->direction.y, cam->direction.z);
	float3 pos = CUDA::float3_add(cam->position, CUDA::float3_mult(focalLength, cam->direction));

	scene.setFocalPlane(Plane(dir, pos, NUM_MATERIALS)); // no material
#endif	

	launchRTKernel(data, WINDOW_WIDTH, WINDOW_HEIGHT, acceleration_structure);

	cudaGraphicsUnmapResources(1, &cudaResourceBuffer, 0);
	// cudaGraphicsUnmapResources(1, &cudaResourceTexture, 0);	
}

/**
* Display callback
* Launches both the kernel and draws the scene
*/
void display()
{


	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

	std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
	// run the Kernel
	runCuda();
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

	// and draw everything
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, PBO);
	glBindTexture(GL_TEXTURE_2D, textureId);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, NULL);

	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 1.0f); glVertex3f(0.0f, 0.0f, 0.0f);
	glTexCoord2f(0.0f, 0.0f); glVertex3f(0.0f, 1.0f, 0.0f);
	glTexCoord2f(1.0f, 0.0f); glVertex3f(1.0f, 1.0f, 0.0f);
	glTexCoord2f(1.0f, 1.0f); glVertex3f(1.0f, 0.0f, 0.0f);
	glEnd();

	glutSwapBuffers();
	glutPostRedisplay();

	float delta = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0f;

	deltaTime += delta;
	deltaTime /= 2.0f;
	fps = 1.f / deltaTime;

	std::cout << std::fixed << fps << std::endl;
}

/**
* Initializes the CUDA part of the app
*
* @param int number of args
* @param char** arg values
*/
void initCuda(int argc, char** argv)
{
	int sizeData = sizeof(uchar3) * WINDOW_SIZE;

	// Generate, bind and register the Pixel Buffer Object (PBO)
	glGenBuffers(1, &PBO);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, PBO);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, sizeData, NULL, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	cudaGraphicsGLRegisterBuffer(&cudaResourceBuffer, PBO, cudaGraphicsMapFlagsNone);

	// Generate, bind and register texture
	glEnable(GL_TEXTURE_2D);
	glGenTextures(1, &textureId);
	glBindTexture(GL_TEXTURE_2D, textureId);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, WINDOW_WIDTH, WINDOW_HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glBindTexture(GL_TEXTURE_2D, 0); // unbind

	// cudaGraphicsGLRegisterImage(&cudaResourceTexture, textureId, GL_TEXTURE_2D, cudaGraphicsMapFlagsNone);
	cudaError_t stat;
	size_t myStackSize = STACK_SIZE;
	stat = cudaDeviceSetLimit(cudaLimitStackSize, myStackSize);

	// copy data to memories	
	GPU_ERROR_CHECK(cudaMemcpyToSymbol(cst_spheres, scene.getSpheres(), scene.getSphereCount() * sizeof(Sphere)));
	GPU_ERROR_CHECK(cudaMemcpyToSymbol(cst_planes, scene.getPlanes(), scene.getPlaneCount() * sizeof(Plane)));
	GPU_ERROR_CHECK(cudaMemcpyToSymbol(cst_lights, scene.getLights(), scene.getLightCount() * sizeof(PointLight)));
	GPU_ERROR_CHECK(cudaMemcpyToSymbol(cst_materials, scene.getMaterials(), scene.getMaterialCount() * sizeof(PhongMaterial)));
	GPU_ERROR_CHECK(cudaMemcpyToSymbol(cst_cylinders, scene.getCylinders(), scene.getCylinderCount() * sizeof(Cylinder)));
	GPU_ERROR_CHECK(cudaMemcpyToSymbol(cst_triangles, scene.getTriangles(), scene.getTriangleCount() * sizeof(Triangle)));
	GPU_ERROR_CHECK(cudaMemcpyToSymbol(cst_scenestats, &scene.getSceneStats(), sizeof(SceneStats)));
	GPU_ERROR_CHECK(cudaMemcpyToSymbol(cst_focalplane, scene.getFocalPlane(), sizeof(Plane)));
	
	runCuda();
}

void initMaterials()
{
	scene.add(MAT_RED);
	scene.add(MAT_GREEN);
	scene.add(MAT_BLUE);
	scene.add(MAT_RED_REFL);
	scene.add(MAT_GREEN_REFL);
	scene.add(MAT_BLUE_REFL);
	scene.add(MAT_WHITE);
	scene.add(MAT_BLACK);
	scene.add(MAT_MIRROR);
}

void initSpheres()
{
	scene.add(Sphere(make_float3(-4, 4, -2), 2.f, MATERIAL_MIRROR));

#if defined ACC_BVH
	std::vector<Sphere> spheres = scene.getSphereVector();
	std::vector<Obj> objects;
	for (std::vector<Sphere>::iterator it = spheres.begin(); it != spheres.end(); ++it) {
		Sphere sphere = *it;
		Obj o;
		o.sphere = sphere;
		o.x = sphere.center.x;
		o.y = sphere.center.y;
		o.z = sphere.center.z;
		o.radius = sphere.radius;

		objects.push_back(o);
	}

	BVHnode tree;
	tree.buildBVH(objects, nullptr, 0, objects.size() - 1, 'x');

	acceleration_structure = copyBVHToDevice(&tree);

#elif defined ACC_KD_TREE
	std::vector<Sphere> spheres = scene.getSphereVector();

	CPU::KDNode* kdTree = buildKDTree(spheres);

	acceleration_structure = copyKDTreeToDevice(kdTree);
#endif
}

void initPlanes()
{
	scene.add(Plane(make_float3(0, 0, 1), make_float3(0, 0, 15), MATERIAL_WHITE)); // vzadu	
	scene.add(Plane(make_float3(0, 1, 0), make_float3(0, 0, 0), MATERIAL_WHITE)); // podlaha	
	scene.add(Plane(make_float3(1, 0, 0), make_float3(-10, 0, 0), MATERIAL_RED)); // leva strana	
	scene.add(Plane(make_float3(-1, 0, 0), make_float3(10, 0, 0), MATERIAL_GREEN)); // prava strana	
	scene.add(Plane(make_float3(0, -1, 0), make_float3(0, 15, 0), MATERIAL_CHECKER)); // podlaha
	scene.add(Plane(make_float3(0, 0, 1), make_float3(0, 0, -15), MATERIAL_WHITE)); // podlaha
}


void initCylinders()
{
	scene.add(Cylinder(make_float3(6, 4, -2), 1.0, make_float3(0, -1, 0), MATERIAL_GREEN));
}

// BOX
// X <-10, 10> L-R
// Y <0,15> T-B
// Z <-inf,15> -B
void initTriangles()
{
	/*
	scene.add(Triangle(
	make_float3(-10, 0, 0), // LT
	make_float3(10, 0, 0), // RT
	make_float3(0, 15, 10), // B
	MATERIAL_RED_REFL));

	scene.add(Triangle(
	make_float3(-10, 0, 0), // LT
	make_float3(-10, 15, 15), // t1
	make_float3(0, 15, 10), // B
	MATERIAL_GREEN_REFL));

	scene.add(Triangle(
	make_float3(10, 15, 15), // t1
	make_float3(10, 0, 0), // RT
	make_float3(0, 15, 10), // B
	MATERIAL_BLUE_REFL));	*/
	
	ObjLoader* loader = new ObjLoader();

	if (loader->loadFile(OBJ))
	{
		loader->parse(scene.getTriangleVector());
		loader->closeFile();
	}

	delete loader;
}

void initLights()
{
	PointLight l1(make_float3(-2.f, 10.f, -15.f), COLOR_WHITE);
	scene.add(l1);
}

void initScene()
{
	initMaterials();
	initSpheres();
	initPlanes();
	initCylinders();
	initTriangles();
	initLights();

	scene.getCamera()->init();
}

/**
* @brief Initializes the OpenGL part of the app
*
* @param int number of args
* @param char** arg values
*/
void initGL(int argc, char** argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
	glutCreateWindow(APP_NAME);
	glutDisplayFunc(display);
	// check for necessary OpenGL extensions
	glewInit();
	if (!glewIsSupported("GL_VERSION_2_0")) {
		std::cerr << "ERROR: Support for necessary OpenGL extensions missing.";
		return;
	}

	glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glDisable(GL_DEPTH_TEST);

	// set matrices
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f);
}

void processKeys(unsigned char key, int x, int y)
{
	switch (key)
	{
	case 'd':
		if (shiftx < 9.8f)
			shiftx += 0.2f;
		break;
	case 'a':
		if (shiftx > -9.8f)
			shiftx -= 0.2f;
		break;
	case 's':
		if (shifty < 14.8f)
			shifty += 0.2f;
		break;
	case 'w':
		if (shifty > 0.2f)
			shifty -= 0.2f;
		break;
#ifdef OPT_DEPTH_OF_FIELD
	case GLUT_KEY_UP:
		focalLength += 0.5;
		break;
	case GLUT_KEY_DOWN:
		focalLength -= 0.5;
		break;
#endif
	default:
		break;
	}
}

/**
* Main
*
* @param int number of args
* @param char** arg values
*/
int main(int argc, char** argv)
{
	initGL(argc, argv);	
	initScene();
	initCuda(argc, argv);

	glutDisplayFunc(display);
	glutKeyboardFunc(processKeys);
	glutMainLoop();

	GPU_ERROR_CHECK(cudaThreadExit());

	return EXIT_SUCCESS;
}

