import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pyswarms as ps
from skopt import gp_minimize
from skopt.space import Real
from pyswarms.single.global_best import GlobalBestPSO
from pyswarms.single.local_best import LocalBestPSO
import numpy as np


# 定义视场角度
horizontal_angle = np.deg2rad(56) # 水平视场角
vertical_angle = np.deg2rad(46) # 垂直视场角



# scale_factor = 2 # 这是你的缩放因子
# max_angle = max(horizontal_angle, vertical_angle) * scale_factor # 将max_angle乘以缩放因子

max_angle = max(horizontal_angle, vertical_angle)

# 定义一个函数，返回一个随机的位置和朝向
def random_position_and_orientation():
    x = np.random.uniform(-1, 1)  # 假设x的范围是[-1, 1]
    y = np.random.uniform(-1, 1)  # 假设y的范围是[-1, 1]
    z =1   # 假设z的范围是[-1, 1]
    roll = np.random.uniform(0, 2*np.pi)  # 滚动角的范围是[0, 2π]
    pitch = np.random.uniform(0, np.pi)  # 俯仰角的范围是[0, π]
    yaw = np.random.uniform(0, 2*np.pi)  # 偏航角的范围是[0, 2π]
    return (x, y, z, roll, pitch, yaw)
   
def camera_to_center(cam, center):
    x, y, z, roll, pitch, yaw = cam
    # 计算相机到地面中心的向量
    vector_to_center = center - np.array([x, y, z])
    # 计算相机到地面中心的方向
    direction_to_center = vector_to_center / np.linalg.norm(vector_to_center)
    # 计算相机的俯仰角和偏航角
    pitch = np.arcsin(direction_to_center[2])
    yaw = np.arctan2(direction_to_center[1], direction_to_center[0])
    # 返回新的相机位置和朝向
    return np.array([x, y, z, roll, pitch, yaw])


#定义一个相机，调用random_position_and_orientation()函数
def camera():
    x, y, z, roll, pitch, yaw = random_position_and_orientation()
    return np.array([x, y, z, roll, pitch, yaw])

#根据相机的方向，结合视场角，生成视场范围
def fov(cam):
    x, y, z, roll, pitch, yaw = cam
    # 计算相机的方向向量
    direction = 3*np.array([np.cos(yaw) * np.cos(pitch), np.sin(yaw) * np.cos(pitch), np.sin(pitch)])
    # 计算相机的右向量
    right = np.array([-np.sin(yaw), np.cos(yaw), 0])
    # 计算相机的上向量
    up = np.cross(right, direction)
    # 计算相机的焦点
    focus = np.array([x, y, z]) + direction
    # 计算相机的焦平面
    focus_plane = focus + direction
    # 计算焦平面的四个角点
    fov_points = np.zeros((4, 3))
    fov_points[0] = focus_plane + horizontal_angle * up + vertical_angle * right
    fov_points[1] = focus_plane + horizontal_angle * up - vertical_angle * right
    fov_points[2] = focus_plane - horizontal_angle * up + vertical_angle * right
    fov_points[3] = focus_plane - horizontal_angle * up - vertical_angle * right
    # 返回视场范围
    return fov_points

#设置采样点
sample_points = np.random.uniform(-2, 2, (200, 3))




#判断采样点是否在相机视场内部
def is_point_in_fov(point, cam):
    x, y, z, roll, pitch, yaw = cam
    camera_position = np.array([x, y, z])
    camera_direction = np.array([np.cos(yaw) * np.cos(pitch), np.sin(yaw) * np.cos(pitch), np.sin(pitch)])

    vector_to_point = point - camera_position
    unit_vector_to_point = vector_to_point / np.linalg.norm(vector_to_point)

    # 计算点与相机之间的角度
    angle = np.arccos(np.clip(np.dot(camera_direction, unit_vector_to_point), -1.0, 1.0))

    # 如果角度小于视场角度的一半，那么点在视场内
    return angle <= max_angle / 2










#粒子群算法，寻找三个相机的最优朝向，使得采样点同时出现在两个及以上的相机视场范围内，
# 初始化粒子群

#相机位置已知，只需要优化朝向
n_cameras = 3
# 定义搜索空间的下限和上限
# 在这里，我们假设相机的X，Y和Z坐标在-10和10之间。
bounds = (np.full(n_cameras*3, 0), np.full(n_cameras*3, 10))

# 创建优化器的实例
options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=n_cameras*3, options=options, bounds=bounds)

# 定义目标函数
def f(X):
    n_particles = X.shape[0]  # 粒子的数量
    scores = []  # 用于存储每个粒子的得分
    for i in range(n_particles):
        # 从粒子中取出三个相机的参数
        cam = np.concatenate([[0, 2, 3.1], X[i, :3]])
        cam2 = np.concatenate([[2, 2, 3.1], X[i, 3:6]])
        cam3 = np.concatenate([[-2, 2, 3.1], X[i, 6:9]])
        score = 0
        for point in sample_points:
            # 如果采样点同时在两个相机的视场内，并且两个相机的视场都在地面中心，那么这个粒子的得分加一
            if is_point_in_fov(point, cam) and is_point_in_fov(point, cam2) and is_point_in_fov(point, cam3):
                score += 1
            if is_point_in_fov(point, cam) and is_point_in_fov(point, cam2):
                score += 1
            if is_point_in_fov(point, cam) and is_point_in_fov(point, cam3):
                score += 1
            if is_point_in_fov(point, cam2) and is_point_in_fov(point, cam3):
                score += 1
        scores.append(score)
    return -np.array(scores)  # 因为pyswarms是求最小值，所以取负数

# 执行优化
cost, pos = optimizer.optimize(f, iters=100)

# 重塑具有形状的最佳位置（n_cameras，3）
best_positions = pos.reshape((-1, 3))

cam=np.concatenate([[0, 2, 3.1], best_positions[0]])
cam2=np.concatenate([[2, 2, 3.1], best_positions[1]])
cam3=np.concatenate([[-2, 2, 3.1], best_positions[2]])







#可视化相机和视场角
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.set_aspect('equal')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-4, 4)


# 绘制采样点
for point in sample_points:
    ax.scatter(point[0], point[1], point[2], c='b', marker='o')


# 绘制相机
ax.scatter(cam[0], cam[1], cam[2], c='r', marker='^')
ax.scatter(cam2[0], cam2[1], cam2[2], c='r', marker='^')
ax.scatter(cam3[0], cam3[1], cam3[2], c='r', marker='^')
# 绘制相机的方向
ax.quiver(cam[0], cam[1], cam[2], np.cos(cam[5]) * np.cos(cam[4]), np.sin(cam[5]) * np.cos(cam[4]), np.sin(cam[4]), length=1, normalize=True, color='r')
ax.quiver(cam2[0], cam2[1], cam2[2], np.cos(cam2[5]) * np.cos(cam2[4]), np.sin(cam2[5]) * np.cos(cam2[4]), np.sin(cam2[4]), length=1, normalize=True, color='r')
ax.quiver(cam3[0], cam3[1], cam3[2], np.cos(cam3[5]) * np.cos(cam3[4]), np.sin(cam3[5]) * np.cos(cam3[4]), np.sin(cam3[4]), length=1, normalize=True, color='r')
# 绘制视场角,将视场清晰度调高，可以看到视场范围
fov_points = fov(cam)

ax.plot([cam[0], fov_points[0][0]], [cam[1], fov_points[0][1]], [cam[2], fov_points[0][2]], c='b')
ax.plot([cam[0], fov_points[1][0]], [cam[1], fov_points[1][1]], [cam[2], fov_points[1][2]], c='b')
ax.plot([cam[0], fov_points[2][0]], [cam[1], fov_points[2][1]], [cam[2], fov_points[2][2]], c='b')
ax.plot([cam[0], fov_points[3][0]], [cam[1], fov_points[3][1]], [cam[2], fov_points[3][2]], c='b')
ax.plot([fov_points[0][0], fov_points[1][0]], [fov_points[0][1], fov_points[1][1]], [fov_points[0][2], fov_points[1][2]], c='b')
ax.plot([fov_points[0][0], fov_points[2][0]], [fov_points[0][1], fov_points[2][1]], [fov_points[0][2], fov_points[2][2]], c='b')
ax.plot([fov_points[3][0], fov_points[1][0]], [fov_points[3][1], fov_points[1][1]], [fov_points[3][2], fov_points[1][2]], c='b')
ax.plot([fov_points[3][0], fov_points[2][0]], [fov_points[3][1], fov_points[2][1]], [fov_points[3][2], fov_points[2][2]], c='b')




fov_points = fov(cam2)
ax.plot([cam2[0], fov_points[0][0]], [cam2[1], fov_points[0][1]], [cam2[2], fov_points[0][2]], c='b')
ax.plot([cam2[0], fov_points[1][0]], [cam2[1], fov_points[1][1]], [cam2[2], fov_points[1][2]], c='b')
ax.plot([cam2[0], fov_points[2][0]], [cam2[1], fov_points[2][1]], [cam2[2], fov_points[2][2]], c='b')
ax.plot([cam2[0], fov_points[3][0]], [cam2[1], fov_points[3][1]], [cam2[2], fov_points[3][2]], c='b')
ax.plot([fov_points[0][0], fov_points[1][0]], [fov_points[0][1], fov_points[1][1]], [fov_points[0][2], fov_points[1][2]], c='b')
ax.plot([fov_points[0][0], fov_points[2][0]], [fov_points[0][1], fov_points[2][1]], [fov_points[0][2], fov_points[2][2]], c='b')
ax.plot([fov_points[3][0], fov_points[1][0]], [fov_points[3][1], fov_points[1][1]], [fov_points[3][2], fov_points[1][2]], c='b')
ax.plot([fov_points[3][0], fov_points[2][0]], [fov_points[3][1], fov_points[2][1]], [fov_points[3][2], fov_points[2][2]], c='b')


fov_points = fov(cam3)
ax.plot([cam3[0], fov_points[0][0]], [cam3[1], fov_points[0][1]], [cam3[2], fov_points[0][2]], c='b')
ax.plot([cam3[0], fov_points[1][0]], [cam3[1], fov_points[1][1]], [cam3[2], fov_points[1][2]], c='b')
ax.plot([cam3[0], fov_points[2][0]], [cam3[1], fov_points[2][1]], [cam3[2], fov_points[2][2]], c='b')
ax.plot([cam3[0], fov_points[3][0]], [cam3[1], fov_points[3][1]], [cam3[2], fov_points[3][2]], c='b')
ax.plot([fov_points[0][0], fov_points[1][0]], [fov_points[0][1], fov_points[1][1]], [fov_points[0][2], fov_points[1][2]], c='b')
ax.plot([fov_points[0][0], fov_points[2][0]], [fov_points[0][1], fov_points[2][1]], [fov_points[0][2], fov_points[2][2]], c='b')
ax.plot([fov_points[3][0], fov_points[1][0]], [fov_points[3][1], fov_points[1][1]], [fov_points[3][2], fov_points[1][2]], c='b')
ax.plot([fov_points[3][0], fov_points[2][0]], [fov_points[3][1], fov_points[2][1]], [fov_points[3][2], fov_points[2][2]], c='b')

#绘制采样点，如果采样点在视场范围内，绘制为红色
for point in sample_points:
    if is_point_in_fov(point, cam) and is_point_in_fov(point, cam2):
        ax.scatter(point[0], point[1], point[2], c='g', marker='o')
    if is_point_in_fov(point, cam) and is_point_in_fov(point, cam3):
        ax.scatter(point[0], point[1], point[2], c='r', marker='o')
    if is_point_in_fov(point, cam2) and is_point_in_fov(point, cam3):
        ax.scatter(point[0], point[1], point[2], c='y', marker='o')
    if is_point_in_fov(point, cam) and is_point_in_fov(point, cam2) and is_point_in_fov(point, cam3):
        ax.scatter(point[0], point[1], point[2], c='y', marker='o')

plt.show()






