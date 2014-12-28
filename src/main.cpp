
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


static float shiftx = 0.0f;
static float shifty = 7.5f;
static float shiftz = -13.f;
static float num = 0.0f;

SceneStats* dev_sceneStats;

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


extern "C" void launchRTKernel(uchar3*, uint32, uint32, Scene*, void*);

float deltaTime = 0.0f;
float fps = 0.0f;
float delta;
bool switchb = true;

#ifdef OPT_DEPTH_OF_FIELD
	float focalLength = FOCALLENGTH;
#endif

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
		30, (float)WINDOW_WIDTH/WINDOW_HEIGHT);	

#ifdef OPT_DEPTH_OF_FIELD	
	float3 dir = make_float3(cam->direction.x,cam->direction.y,cam->direction.z);
	float3 pos = CUDA::float3_add(cam->position, CUDA::float3_mult(focalLength,cam->direction));
	
	scene.setFocalPlane(Plane(dir, pos, NUM_MATERIALS)); // no material
#endif	

	launchRTKernel(data, WINDOW_WIDTH, WINDOW_HEIGHT, &scene, acceleration_structure);

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
	glTexCoord2f(0.0f, 1.0f); glVertex3f(0.0f,0.0f,0.0f);
	glTexCoord2f(0.0f, 0.0f); glVertex3f(0.0f,1.0f,0.0f);
	glTexCoord2f(1.0f, 0.0f); glVertex3f(1.0f,1.0f,0.0f);
	glTexCoord2f(1.0f, 1.0f); glVertex3f(1.0f,0.0f,0.0f);
	glEnd();

	glutSwapBuffers();
	glutPostRedisplay();  

	

	float delta = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0f;

	deltaTime += delta;
	deltaTime /= 2.0f;
	fps = 1.f/ deltaTime;

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
	scene.add(Sphere(make_float3(-4, 4, -2), 4.f, MATERIAL_MIRROR));
	
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
		MATERIAL_BLUE_REFL));	
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
	Scene s;
	initScene();

	initCuda(argc, argv);  



	glutDisplayFunc(display);
	glutKeyboardFunc(processKeys);
	glutMainLoop();

	cudaThreadExit();  

	return EXIT_SUCCESS;
}
