#ifndef FILEREADER_CUH
#define FILEREADER_CUH

#include <vector>

using namespace std;

class FileReader {
private:
	static void read_obj_file(char *dir, vector<hittable*> &vec_obj_list, material *mat_ptr);
public:
	static bool readfile_to_render(
		const char *path,          // 配置文件的相对路径
		int &nx, int &ny, int &ns, // 画布大小，采样次数
		camera *&c,                // 摄像机
		hittable **&obj_list,
		int &o_list_size, // 需要渲染的物体的数组和这个数组的长度
		material **&mat_list,
		int &m_list_size);
};

#endif