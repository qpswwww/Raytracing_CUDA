#ifndef FILEREADER_CUH
#define FILEREADER_CUH

#include <vector>

using namespace std;

class FileReader {
private:
	static void read_obj_file(char *dir, vector<hittable*> &vec_obj_list, material *mat_ptr);
public:
	static bool readfile_to_render(
		const char *path,          // �����ļ������·��
		int &nx, int &ny, int &ns, // ������С����������
		camera *&c,                // �����
		hittable **&obj_list,
		int &o_list_size, // ��Ҫ��Ⱦ�������������������ĳ���
		material **&mat_list,
		int &m_list_size);
};

#endif