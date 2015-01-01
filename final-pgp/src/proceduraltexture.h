#ifndef PROCEDURAL_H
#define PROCEDURAL_H

namespace CUDA
{
	__device__ int CheckerProcedural(int matID1, int matID2, float3 position)
	{
		int check = (int(floor(position.x + 1e-10)) ^ int(floor(position.y + 1e-10)) ^ int(floor(position.z + 1e-10)));
		if (check & 1){
			return matID1;
		} else {
			return matID2;
		}	
	}
}

#endif