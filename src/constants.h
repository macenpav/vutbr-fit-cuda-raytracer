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

enum Materials {
	MATERIAL_BLUE = 0,
	MATERIAL_RED,
	MATERIAL_GREEN,
	MATERIAL_YELLOW,
	MATERIAL_BLACK,
	MATERIAL_WHITE,
	MATERIAL_CHECKER,

	NUM_MATERIALS
};

#define NUM_PLANES 4
#define NUM_SPHERES 4
#define NUM_CYLINDERS 1
#define NUM_LIGHTS 1
#define NUM_TRIANGLES 1

#define SUB_CONST 4

#define FOCALLENGTH  12.0
#define LENSRADIUS 0.3
#define LIGHTRADIUS 1

//pøepínaèe pro ovládání funkcí raytracingu

//#define BILINEAR_SAMPLING 1 

  //#define BUILD_WITH_BVH
 //#define BUILD_WITH_KDTREE

#define CAMERASHIFT  //pohyb kamery
//#define DEPTHOFFIELD //hloubka ostrosti
//#define SOFTSHADOWS //hladke stiny

const uint32 SPLIT_LIMIT = 5;

#endif
