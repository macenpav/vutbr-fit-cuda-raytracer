#ifndef CONSTANTS_H
#define CONSTANTS_H

// custom types
typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint32;

typedef char int8;
typedef short int16;
typedef int int32;

const char APP_NAME[] = "Ray Tracer";

const uint32 WINDOW_WIDTH = 800;
const uint32 WINDOW_HEIGHT = 600;
const float WINDOW_ASPECT = WINDOW_WIDTH / WINDOW_HEIGHT;
const uint32 WINDOW_SIZE = WINDOW_WIDTH * WINDOW_HEIGHT;
const uint32 NUM_THREADS = 256;
const uint32 STACK_SIZE = 4096;
const uint32 THREADS_PER_BLOCK = 8;



#define M_PI 3.14159265358979323846
#define M_PI_2 1.57079632679489661923
#define NO_HIT -1

/** 
 * @brief Materials on device.   
 * @description Materials on device. Ordered in the same order as defined in scene init.
 */
enum CudaMaterials {
	MATERIAL_RED = 0,
	MATERIAL_GREEN,
	MATERIAL_BLUE,				
	MATERIAL_RED_REFL,
	MATERIAL_GREEN_REFL,
	MATERIAL_BLUE_REFL,	
	MATERIAL_WHITE,
	MATERIAL_BLACK,
	MATERIAL_MIRROR,
	MATERIAL_CHECKER,

	NUM_MATERIALS
};

#define SUB_CONST 4





/** @brief Acceleration structures */
#define ACC_NONE
// #define ACC_BVH
// #define ACC_KD_TREE

/** @brief Options */
// #define OPT_BILINEAR_SAMPLING	 
// #define OPT_CAMERA_SHIFT		
// #define OPT_DEPTH_OF_FIELD		
// #define OPT_SOFT_SHADOWS		

#ifdef OPT_DEPTH_OF_FIELD
	#define FOCALLENGTH		12.0
	#define LENSRADIUS		0.3
	#define LIGHTRADIUS		1
#endif

const uint32 SPLIT_LIMIT = 5;

/** @brief Default phong shininess for all materials. */
const float DEFAULT_PHONG_SHININESS = 15.f;
/** @brief Default phong reflectance for reflective materials. */
const float DEFAULT_PHONG_REFLECTANCE = 0.5f;

/** @brief max size of constant memory reservered for each primitive */
#define MAX_SPHERES 32
#define MAX_TRIANGLES 32
#define MAX_CYLINDERS 32
#define MAX_LIGHTS 32
#define MAX_PLANES 32


#endif
