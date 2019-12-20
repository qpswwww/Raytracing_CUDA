摄像机参数：
vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect,float aperture, float focus_dist

材质：
1. 塑料(漫反射模型)：lambertian(const vec3& a); a是RGB颜色
2. 金属：metal(const vec3& a, float f); a是RGB颜色，f是albedo反射率
3. 玻璃：dielectric(float ri); ri是折射率

模型：
1. 球体(在前端编辑时**插入**到场景中) sphere(vec3 cen, float r, material *m); cen是中心点坐标,r是半径，m是材质的指针
2. 三角网格面构成的obj模型(在前端编辑时**导入**到场景中)
- 由若干三角面组成：
- - vec3 vertexone, 第一个顶点坐标 
- - vec3 vertextwo, 
- - vec3 vertexthree,
- - vec3 vn1, 第一个顶点的法向量坐标
- - vec3 vn2, 
- - vec3 vn3,
- - material *mat_ptr 材质的指针

希望读出的：
1. 模型物体的数组：hittable **list, int &list_size  read_model(char *dir, hittable **list, int &list_size)
2. 摄像机: camera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect,float aperture, float focus_dist),摄像机只有一个