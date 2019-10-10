#ifndef OBJ_PARSER_CUH
#define OBJ_PARSER_CUH

#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include "vec3.cuh"
#include "hittable.cuh"
#include "triangle.cuh"
#include "material.cuh"

using namespace std;

void read_obj_file(char *dir, hittable **list, int &list_size) {
	FILE *f;
	fopen_s(&f, dir, "r");
	if (f == NULL) {
		exit(9);
	}
	vector<vec3> points, p_normals;
	char line[100];
	if (f != NULL) {
		while (fgets(line, 100, f) != NULL) {
			int p = 0;
			while ((line[p] >= 'a'&&line[p] <= 'z') || (line[p] >= 'A'&&line[p] <= 'Z')) p++;
			if (line[0] == 'v'&&line[1] != 'n') {
				double x, y, z;
				sscanf_s(line + p, "%lf%lf%lf", &x, &y, &z);
				vec3 temp = vec3(x, y, z);
				//cout<<x*500<<' '<<y*500<<' '<<z*500<<endl;
				points.push_back(temp);
			}
			else if (line[0] == 'v'&&line[1] == 'n') {
				double x, y, z;
				sscanf_s(line + p, "%lf%lf%lf", &x, &y, &z);
				vec3 temp = vec3(x, y, z);
				//cout<<x*500<<' '<<y*500<<' '<<z*500<<endl;
				p_normals.push_back(temp);
			}
			else if (line[0] == 'f') {
				int x, y, z;
				sscanf_s(line + p, "%d%d%d", &x, &y, &z);
				list[list_size++] = new triangle(points[x - 1], points[y - 1], points[z - 1],
					p_normals[x - 1], p_normals[y - 1], p_normals[z - 1],
					new dielectric(1.5));
				//new lambertian(vec3(0.8, 0.8, 0.0)));
			}
		}
		fclose(f);
	}
}

#endif // !OBJ_PARSER_H
