#ifndef SCENE_H
#define SCENE_H

#include <vector>
#include "sphere.h"
#include "plane.h"
#include "camera.h"
#include "light.h"
#include "cylinder.h"

struct SceneStats{
	uint32 sphereCount;
	uint32 planeCount;
	uint32 lightCount;
	uint32 cylinderCount;
};

class Scene
{
public: 
	Scene() {
		planeId = 0;
		sphereId = 0;
		cylinderId = 0;
	}
	uint32 getSphereCount() const { return spheres.size(); }
	uint32 getPlaneCount() const { return planes.size(); }
	uint32 getCylinderCount() const { return cylinders.size(); }
	uint32 getLightCount() const { return lights.size(); }

	void add(Sphere s) { s.id = sphereId++; spheres.push_back(s); }
	void add(Plane p){ p.id = planeId++;  planes.push_back(p); }
	void add(Cylinder c){ c.id = cylinderId++; cylinders.push_back(c); }
	void add(PointLight p){ lights.push_back(p); }
	void add(PhongMaterial mat) { materials.push_back(mat); }

	Camera* getCamera() { return &camera; }
	Sphere* getSpheres() { return &spheres[0]; }
	Plane* getPlanes() { return &planes[0];	}
	Cylinder * getCylinders() { return &cylinders[0]; }
	PointLight* getLights() { return &lights[0]; }
	PhongMaterial* getMaterials() { return &materials[0]; }

	SceneStats* getSceneStats(){
		sceneStats.planeCount = planes.size();
		sceneStats.sphereCount = spheres.size();
		sceneStats.cylinderCount = cylinders.size();
		sceneStats.lightCount = lights.size();
		return &sceneStats;
	};

	std::vector<Sphere> getSphereVector() const { return spheres; }

private:
	std::vector<PhongMaterial> materials;
	std::vector<Sphere> spheres;
	std::vector<Plane> planes;
	std::vector<Cylinder> cylinders;
	std::vector<PointLight> lights;
	Camera camera;
	SceneStats sceneStats;
	uint32 sphereId;
	uint32 planeId;
	uint32 cylinderId;
};

#endif