import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from sklearn.preprocessing import MinMaxScaler
import yaml
from yaml import Loader


mm_scal_path = 'test_disc_functions/mMscaler_Acetone.yml'
with open(mm_scal_path, 'r') as file:
    mm_dict = yaml.load(file, Loader=Loader)

mm_acetone = MinMaxScaler()
mm_acetone.fit(np.random.rand(5, 3))  # fake fit
# UPDATE SCALER TO SAVED DATA
for attr in mm_dict:
    setattr(mm_acetone, attr, mm_dict[attr])

acetone_data = pd.read_csv('test_disc_functions/Acetonetable.csv')
acetone_X = acetone_data.loc[:, ['K', 'MPa', 'g/l=kg/m3']]
acetone_Xmm = mm_acetone.transform(acetone_X.values)
acetone_Xmm = acetone_Xmm[acetone_Xmm[:, 1] <= 1.]
acetone_Xmm = np.vstack([acetone_Xmm, np.array([[0., 1., 1.], [1., 0., 0.], [0., 0., 0.]])])

# SIMPLICES:
reg00_inds = list(range(24)) + list(range(28, 34)) + list(range(40, 44)) + [50, 51, 58, 59, 61, 62]
bound_inds = list(range(12)) + [23, 50, 58]
reg00_nobound_inds = list(set(reg00_inds).difference(bound_inds))
reg01_inds = set(range(acetone_Xmm.shape[0])).difference(reg00_nobound_inds)

tri00_simps = [
    [0, 1, 12],
    [1, 12, 13],
    [1, 2, 13],
    [2, 13, 14],
    [2, 3, 14],
    [3, 14, 15],
    [3, 15, 16],
    [3, 4, 16],
    [16, 17, 4],
    [16, 17, 28],
    [4, 5, 17],
    [5, 17, 18],
    [5, 6, 18],
    [6, 18, 19],
    [6, 7, 19],
    [19, 20, 7],
    [7, 8, 20],
    [8, 20, 21],
    [8, 9, 21],
    [9, 21, 22],
    [9, 10, 22],
    [10, 11, 22],  # 23 coincides with 11
    [10, 11, 50],  # 23 coincides with 11
    [11, 50, 51],  # 23 coincides with 11
    [11, 22, 51],  # 23 coincides with 11
    [50, 51, 59],
    [50, 58, 59],
    [17, 28, 29],
    [17, 18, 29],
    [18, 19, 29],
    [19, 29, 30],
    [19, 20, 40],
    [20, 21, 40],
    [19, 30, 40],
    [21, 40, 41],
    [30, 40, 41],
    [30, 31, 41],
    [31, 41, 42],
    [31, 32, 42],
    [32, 33, 42],
    [33, 42, 43],
    [21, 41, 42],
    [21, 22, 42],
    [22, 42, 43],
    [22, 43, 51],
    [12, 13, 61],
    [13, 14, 61],
    [14, 15, 61],
    [15, 16, 61],
    [16, 28, 29],
    [16, 29, 61],
    [29, 30, 61],
    [30, 31, 61],
    [31, 32, 61],
    [32, 33, 61],
    [33, 59, 61],
    [12, 61, 62]
]

tri01_simps = [
    [0, 44, 60],
    [0, 34, 44],
    [0, 24, 34],
    [0, 1, 24],
    [1, 2, 24],
    [2, 24, 25],
    [2, 3, 25],
    [3, 25, 26],
    [3, 4, 26],
    [4, 5, 26],
    [5, 26, 36],
    [5, 36, 37],
    [5, 6, 37],
    [6, 37, 38],
    [6, 7, 38],
    [7, 8, 38],
    [8, 38, 39],
    [8, 9, 39],
    [9, 10, 49],
    [10, 49, 50],
    [9, 39, 49],
    [39, 48, 49],
    [38, 39, 48],
    [38, 47, 48],
    [37, 38, 47],
    [37, 46, 47],
    [36, 37, 46],
    [36, 45, 46],
    [35, 36, 45],
    [35, 36, 25],
    [25, 26, 36],
    [25, 34, 35],
    [24, 25, 34],
    [34, 35, 44],
    [35, 44, 45],
    [44, 52, 60],
    [44, 45, 52],
    [45, 52, 53],
    [45, 46, 53],
    [46, 53, 54],
    [46, 47, 54],
    [47, 54, 55],
    [47, 48, 55],
    [48, 55, 56],
    [48, 49, 56],
    [49, 56, 57],
    [49, 50, 57],
    [50, 57, 58]
]

tri = Delaunay(acetone_Xmm[:, :2])
tri_simplices = np.vstack([np.array(tri01_simps, dtype=np.int32), np.array(tri00_simps, dtype=np.int32)])
tri.simplices = tri_simplices.copy()

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('TkAgg', force=True)  # Uncomment if you want to use this option
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # Use plot_trisurf to create a surface
    ax.plot_trisurf(acetone_Xmm[:, 0], acetone_Xmm[:, 1], acetone_Xmm[:, 2], triangles=tri.simplices, cmap='viridis', edgecolor='k', alpha=0.8)
    # Set labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Triangulated Surface with plot_trisurf")
    plt.legend()
    plt.show()


