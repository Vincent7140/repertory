import pandas as pd
import numpy as np
from pyproj import Transformer
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# === 1. Données centrales RPC ===
valeurs_centrales_data = {
    "ID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
           11, 12, 13, 14, 15, 16, 18, 19, 20, 22, 23, 25, 26],
    "Latitude Offset": [
        30.2992, 30.2994, 30.3007, 30.2996, 30.2998, 30.3003, 30.2999, 30.2835, 30.3027, 30.3151,
        30.2996, 30.2998, 30.3009, 30.3013, 30.3004, 30.301, 30.2994, 30.3, 30.2995, 30.2995,
        30.2944, 30.2783, 30.2749
    ],
    "Longitude Offset": [
        -81.6397, -81.6397, -81.6396, -81.6403, -81.6408, -81.6408, -81.6402, -81.6402, -81.6401, -81.64,
        -81.6406, -81.6409, -81.6401, -81.6386, -81.6414, -81.64, -81.6398, -81.6409, -81.6396, -81.6412,
        -81.6414, -81.6306, -81.619
    ],
    "Height Offset": [
        -21, -21, -19, -20, -19, -21, -21, -21, -19, -19,
        -21, -21, -21, -19, -19, -19, -21, -21, -19, -21,
        -21, -19, -19
    ]
}
df_valeurs = pd.DataFrame(valeurs_centrales_data)

# === 2. Données angulaires ===
angle_data = {
    "ID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
           11, 12, 13, 14, 15, 16, 18, 19, 20, 21,
           22, 23, 25, 26],
    "Az": [
        0.3455751918948773, 0.4188790204786391, 2.410299697004169, 4.665265090580843,
        1.9495327744776663, 0.7435102613495844, 3.8117990863556157, 3.7140606482439336,
        3.5883969421003417, 3.5238197597765515, 2.5185101106278176, 0.06632251157578452,
        3.2759830059933566, 4.54309204294124, 1.4905111812031575, 3.8117990863556157,
        4.32143522793796, 1.9442967867216832, 4.516912104161325, 4.434881629317592,
        0.9320058205649719, 2.9286624848464853, 3.7088246604879505, 4.679227724596798
    ],
    "El": [
        1.2095131716320704, 1.3334315485236679, 1.2356931104119853, 1.429424657383356,
        1.260127719939906, 1.2967796342317868, 1.3439035240356338, 1.2880529879718152,
        1.1501719770642633, 1.0088003076527223, 1.4765485471872026, 1.2636183784438948,
        1.2287117934040082, 1.171115928088195, 1.1030480872604163, 1.2147491593880533,
        1.457349925415265, 1.356120828799594, 1.101302758008422, 1.0053096491487339,
        1.3543754995475996, 1.2287117934040082, 1.1798425743481666, 1.0524335389525807
    ]
}
df_angles = pd.DataFrame(angle_data)

# Fusion
df_angles = df_angles[df_angles["ID"].isin(df_valeurs["ID"])]
df_combined = df_valeurs.merge(df_angles, on="ID")

# === 3. Estimation position satellite ===
transformer = Transformer.from_crs("epsg:4326", "epsg:4978", always_xy=True)
sat_alt = 617000

satellite_ecef = []
origin_ecef = []
for _, row in df_combined.iterrows():
    lon, lat, alt = row["Longitude Offset"], row["Latitude Offset"], row["Height Offset"]
    az, el = row["Az"], row["El"]

    dx = -np.cos(el) * np.sin(az)
    dy = -np.cos(el) * np.cos(az)
    dz =  np.sin(el)
    direction = np.array([dx, dy, dz])

    x0, y0, z0 = transformer.transform(lon, lat, alt)
    ground_point = np.array([x0, y0, z0])
    sat_pos = ground_point + direction * sat_alt

    origin_ecef.append(ground_point)
    satellite_ecef.append(sat_pos)

origin_ecef = np.array(origin_ecef)
satellite_ecef = np.array(satellite_ecef)

# === 4. Repère local verticalisé basé sur ECEF ===
center_ecef = np.mean(origin_ecef, axis=0)
origin = center_ecef

z_axis = center_ecef / np.linalg.norm(center_ecef)
tmp = np.array([0, 0, 1]) if not np.allclose(z_axis, [0, 0, 1]) else np.array([1, 0, 0])
x_axis = np.cross(tmp, z_axis)
x_axis /= np.linalg.norm(x_axis)
y_axis = np.cross(z_axis, x_axis)
R = np.vstack([x_axis, y_axis, z_axis]).T  # 3x3

def ecef_to_local(points_ecef, origin_ecef, R):
    return (points_ecef - origin_ecef) @ R

satellite_local = ecef_to_local(satellite_ecef, origin, R)
ground_local = ecef_to_local(origin_ecef, origin, R)

# === 5. Visualisation avec Z = Altitude ===
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=30, azim=135)

# Vecteurs vers le point moyen
vectors = -satellite_local

# Affichage : X = Est local, Y = Nord local, Z = Altitude (Up)
ax.scatter(
    satellite_local[:, 0],  # X
    satellite_local[:, 1],  # Y
    satellite_local[:, 2],  # Z (Altitude)
    color='red', s=40, label='Satellite'
)
ax.quiver(
    satellite_local[:, 0], satellite_local[:, 1], satellite_local[:, 2],
    vectors[:, 0], vectors[:, 1], vectors[:, 2],
    color='purple', arrow_length_ratio=0.0, linewidth=1.5, label='Vers point moyen'
)

# Points au sol
ax.scatter(
    ground_local[:, 0],
    ground_local[:, 1],
    ground_local[:, 2],
    color='green', s=20, label='Points au sol'
)

# Point moyen
ax.scatter(0, 0, 0, color='orange', s=80, marker='X', label='Point moyen')

# ID
for i, pos in enumerate(satellite_local):
    ax.text(pos[0], pos[1], pos[2], str(df_combined["ID"].iloc[i]), color='black', fontsize=8)

# Axes
ax.set_xlabel("Local X (m)")
ax.set_ylabel("Local Y (m)")
ax.set_zlabel("Altitude (m)")
ax.set_title("Représentation verticalisée (Z = altitude réelle)")
ax.legend(loc='upper right')
plt.tight_layout()
plt.show()
