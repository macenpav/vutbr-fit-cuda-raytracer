#ifndef OBJLOADER_H
#define OBJLOADER_H

#include <glm/glm.hpp>
#include <vector>
#include <fstream>
#include <sstream>
#include <vector_types.h>
#include <iostream>

#include "triangle.h"
#include "constants.h"

class ObjLoader
{
	public:
		bool loadFile(const char* path)
		{
			file = fopen(path, "r");
			if (file == NULL)
			{
				std::cerr << "Failed to read file." << std::endl;
				return false;
			}
		}

		void closeFile()
		{
			fclose(file);
		}

		bool parse(std::vector<Triangle>* triangles)
		{
			while (true)
			{
				char lineHeader[128];
				int res = fscanf(file, "%s", lineHeader);
				
				// end of file
				if (res == EOF)
					break;

				if (strcmp(lineHeader, "v") == 0) // vertex
				{
					float3 vertex;
					fscanf(file, "%f %f %f\n", &vertex.x, &vertex.y, &vertex.z);
					vertices.push_back(vertex);
				}			
				else if (strcmp(lineHeader, "f") == 0) // vertex
				{					
					unsigned int vertexIndex[3];
					int matches = fscanf(file, "%d %d %d\n", &vertexIndex[0], &vertexIndex[1], &vertexIndex[2]);
					if (matches != 3){
						printf("Failed to read f parameter.");
						return false;
					}

					
					float3 translate = make_float3(-1.f, 10.f, -10.f);
					Triangle t(
						vertices[vertexIndex[0] - 1] + translate,
						vertices[vertexIndex[1] - 1] + translate,
						vertices[vertexIndex[2] - 1] + translate,
						MATERIAL_RED
					);
					
					triangles->push_back(t);
					if (triangles->size() >= MAX_TRIANGLES)
						break;
				}
			}
			return true;
		}
		
	private:		
		FILE* file;
		
		std::vector<uint32> vertexIndices, uvIndices, normalIndices;
		std::vector<float3> vertices;
};

#endif
