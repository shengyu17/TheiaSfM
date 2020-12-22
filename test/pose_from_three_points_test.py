from pytheia_sfm import PoseFromThreePoints


import numpy as np
R = np.eye(3)
t = np.array([0.0,0.1,1.0])
pts3d = np.array([[-0.1,0.1,0.0],[-0.2,0.0,-0.1],[-0.3,0.4,-0.2]])
tmp = R@pts3d + t
tmp /= tmp[2,:]
pts2d = tmp[0:2,:]
Rl = []
tl=[]
d = PoseFromThreePoints(pts2d.T, pts3d.T)
print(d)
print(type(d))
print(d[0])
print(len(d[1]))
print(type(d[2][0]))


