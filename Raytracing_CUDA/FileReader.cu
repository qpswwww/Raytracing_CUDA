#include "rapidjson/document.h"
#include "rapidjson/istreamwrapper.h"
#include "camera.cuh"
#include "hittable.cuh"
#include "material.cuh"
#include "sphere.cuh"
#include "triangle.cuh"
#include <stdio.h>
#include <vector>
#include <fstream>
#include "FileReader.cuh"

using namespace rapidjson;
using namespace std;

void FileReader::read_obj_file(char *dir, vector<hittable*> &vec_obj_list, material *mat_ptr) {
	FILE *f;
	fopen_s(&f, dir, "r");
	if (f == NULL) {
		exit(9);
	}
	vector<vec3> points, p_normals;
	//vector<hittable*> vec_objs;
	char line[100];
	//int obj_id = 0;
	if (f != NULL) {
		while (fgets(line, 100, f) != NULL) {
			int p = 0;
			while ((line[p] >= 'a'&&line[p] <= 'z') || (line[p] >= 'A'&&line[p] <= 'Z')) p++;
			/*if (line[0] == 'o') {
			obj_id++;
			}*/
			if (line[0] == 'v'&&line[1] != 'n') {
				double x, y, z;
				sscanf_s(line + p, "%lf%lf%lf", &x, &y, &z);
				vec3 temp = vec3(x, y, z);
				//cout<<x*500<<' '<<y*500<<' '<<z*500<<endl;
				points.push_back(temp);
			}
			else if (line[0] == 'v'&&line[1] == 'n') {
				double x, y, z;
				//has_normal = true;
				sscanf_s(line + p, "%lf%lf%lf", &x, &y, &z);
				vec3 temp = vec3(x, y, z);
				//cout<<x*500<<' '<<y*500<<' '<<z*500<<endl;
				p_normals.push_back(temp);
			}
			else if (line[0] == 'f') {
				int x, y, z;
				sscanf_s(line + p, "%d%d%d", &x, &y, &z);
				//list[list_size++]
				//if (obj_id == 1)
				/*if (mat_ptr->type == type_lambertian) {
					vec_objs.push_back(new triangle(points[x - 1], points[y - 1], points[z - 1],
						new lambertian(((lambertian*)mat_ptr)->albedo)));
				}else if (mat_ptr->type == type_metal) {
					vec_objs.push_back(new triangle(points[x - 1], points[y - 1], points[z - 1],
						new metal(((metal*)mat_ptr)->albedo)));
				}*/
				vec_obj_list.push_back(new triangle(points[x - 1], points[y - 1], points[z - 1],
					mat_ptr));
			}
		}
		/*list_size = vec_objs.size();
		list = (hittable**)malloc(list_size * sizeof(hittable*));
		for (int i = 0; i < list_size; i++) {
			list[i] = vec_objs[i];
		}*/
		fclose(f);
	}
}

bool FileReader::readfile_to_render(
	const char *path,          // 配置文件的相对路径
	int &nx, int &ny, int &ns, // 画布大小，采样次数
	camera *&c,                // 摄像机
	hittable **&obj_list,
	int &o_list_size, // 需要渲染的物体的数组和这个数组的长度
	material **&mat_list,
	int &m_list_size)
{
	ifstream inputfile(path);
	IStreamWrapper _TMP_ISW(inputfile);
	Document json_tree;
	json_tree.ParseStream(_TMP_ISW);
	cerr << (nx = json_tree["nx"].GetInt()) << endl;
	cerr << (ny = json_tree["ny"].GetInt()) << endl;
	cerr << (ns = json_tree["ns"].GetInt()) << endl;

	c = new camera(
		vec3(
			json_tree["camera"]["lookfrom"][0].GetDouble(),
			json_tree["camera"]["lookfrom"][1].GetDouble(),
			json_tree["camera"]["lookfrom"][2].GetDouble()),
		vec3(
			json_tree["camera"]["lookat"][0].GetDouble(),
			json_tree["camera"]["lookat"][1].GetDouble(),
			json_tree["camera"]["lookat"][2].GetDouble()),
		vec3(
			json_tree["camera"]["vup"][0].GetDouble(),
			json_tree["camera"]["vup"][1].GetDouble(),
			json_tree["camera"]["vup"][2].GetDouble()),
		json_tree["camera"]["vfov"].GetDouble(),
		json_tree["camera"]["aspect"].GetDouble(),
		json_tree["camera"]["aperture"].GetDouble(),
		json_tree["camera"]["focus_dist"].GetDouble());

	cerr << (m_list_size = json_tree["materials"].Size()) << endl;
	mat_list = new material *[m_list_size];
	for (int i = 0; i < m_list_size; i++)
	{
		if (strcmp(json_tree["materials"][i]["type"].GetString(), "lamberian") == 0)
		{
			mat_list[i] = new lambertian(
				vec3(
					json_tree["materials"][i]["albedo"][0].GetDouble(),
					json_tree["materials"][i]["albedo"][1].GetDouble(),
					json_tree["materials"][i]["albedo"][2].GetDouble()));
		}
		else if ((strcmp(json_tree["materials"][i]["type"].GetString(), "metal") == 0))
		{
			mat_list[i] = new metal(
				vec3(
					json_tree["materials"][i]["albedo"][0].GetDouble(),
					json_tree["materials"][i]["albedo"][1].GetDouble(),
					json_tree["materials"][i]["albedo"][2].GetDouble()),
				json_tree["materials"][i]["fuzz"].GetDouble());
		}
		else if (((strcmp(json_tree["materials"][i]["type"].GetString(), "dieletric") == 0)))
		{
			mat_list[i] = new dielectric(
				vec3(
					json_tree["materials"][i]["albedo"][0].GetDouble(),
					json_tree["materials"][i]["albedo"][1].GetDouble(),
					json_tree["materials"][i]["albedo"][2].GetDouble()),
				json_tree["materials"][i]["ref_idx"].GetDouble());
		}
	}
	int sphere_cnt = json_tree["spheres"].Size(), obj_cnt = json_tree["objfile"].Size();
	//cout << (o_list_size = sphere_cnt + obj_cnt) << endl;
	//obj_list = new hittable *[sphere_cnt + obj_cnt];
	vector<hittable*> vec_obj_list;
	for (int i = 0; i < sphere_cnt; i++)
	{
		//obj_list[i] = 
		vec_obj_list.push_back(
			new sphere(
				vec3(
					json_tree["spheres"][i]["center"][0].GetDouble(),
					json_tree["spheres"][i]["center"][1].GetDouble(),
					json_tree["spheres"][i]["center"][2].GetDouble()),
				json_tree["spheres"][i]["radius"].GetDouble(),
				mat_list[json_tree["spheres"][i]["material"].GetInt() - 1]
			)
		);
	}
	//o_list_size = sphere_cnt;
	for (int i = 0; i < obj_cnt; i++) {
		hittable **tmp_list;
		int tmp_list_size;
		read_obj_file((char*)(json_tree["objfile"][i]["dir"].GetString()), vec_obj_list, mat_list[json_tree["objfile"][i]["material"].GetInt() - 1]);
		//memcpy(obj_list + o_list_size, tmp_list, tmp_list_size * sizeof(hittable*));
		//o_list_size += tmp_list_size;
	}
	o_list_size = vec_obj_list.size();
	obj_list = new hittable*[o_list_size];
	for (int i = 0; i < o_list_size; i++) {
		obj_list[i] = vec_obj_list[i];
	}
	return 1;
}