import taichi as ti 
ti.init(arch=ti.cuda, default_fp=ti.f32)
import numpy as np
from magneticMPM.sim.mpm_class import magneticMPM
from magneticMPM.sim.loadRobot import Robot
from magneticMPM.colour_palette import generic_palette
import pickle


def load_best_fitness(file_path):
    try:
        with open(file_path, "rb") as f:
            results = pickle.load(f)
        
        generations = [data["generation"] for data in results]
        best_fitnesses = [data["best_fitness"] for data in results]
        best_solution = [data["best_solution"] for data in results]
        
        
        return generations, best_fitnesses, best_solution

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return [], []
    except Exception as e:
        print(f"An error occurred while loading or processing the file: {e}")
        return [], []

def phi_theta_to_vector(phi, theta, magnitude=1.0):
    x = magnitude * np.sin(theta) * np.cos(phi)
    y = magnitude * np.sin(theta) * np.sin(phi)
    z = magnitude * np.cos(theta)
    return np.array([x, y, z])

robotFile = "magneticMPM/SmallScaleBot/SmallScaleBot.yaml"
print("Generating Particles....")
r = Robot(robotFile, ppm3=1e13, scale=1e-3)

grid_size = 25e-3 
dx = 0.1e-3
g = np.array([0, 0, -9.81])
gamma = 180
offset = np.array([3e-3, 3e-3, 0.2e-3])
dt = 5e-7

print("Initialising Variables... (this may take a while)")
generations, best_fitnesses, best_solutions = load_best_fitness("../data_out/ex3.pkl")

b_ind = max(best_fitnesses)
b_index = best_fitnesses.index(b_ind)
print("best_fitness", b_ind)
print("b_solution", best_solutions[b_index])

reshaped = np.array(best_solutions[b_index]).reshape(-1, 2)


vectors = np.hstack([np.zeros((reshaped.shape[0], 1)), reshaped])


r.set_magnetisation_vectors(vectors)

mpm = magneticMPM(r, scale=1, grid_size=grid_size, dx=dx, g=g, gamma=gamma, offset=offset, colour_palette=generic_palette)
initial_position = mpm.get_avg_positions().to_numpy()



window = ti.ui.Window("Window", (1024, 1024))
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()
# camera.position(0, 10, 10)
# camera.lookat(5, 0, 5)
camera_x = -12
camera_y = 3.75
camera_z = 6
lookat_x = 5
lookat_y = 0
lookat_z = 9.5
camera.position(camera_x, camera_y, camera_z)
camera.lookat(lookat_x, lookat_y, lookat_z)

field = np.array([0, 0 ,0])
t = 0
p_size = 0.01
magnitude = 0
theta = np.pi
angle = 0
rotate = True
frq = 10

timesteps = 0
renderEvery = 20
frame_idx = 0
# x_visualise = ti.Vector.field(3, dtype=float, shape=(9))
# mpm.getParticles(1e3)
# x_visualise= mpm.x_visualise
# x_visualise[4][2] += 5
# initial_position[4][1] +=(3/1e3)
# for i in range(9):
#     x_visualise[i][0] = initial_position[i][0] * 1e3
#     x_visualise[i][1] = initial_position[i][2] * 1e3
#     x_visualise[i][2] = initial_position[i][1] * 1e3
# print(x_vidu)
# Create a Taichi Vector field for initial positions
# init_pos_field =  ti.Vector.field(3, dtype=float, shape=(9))

# # Transfer initial positions into Taichi field
# for i in range(9):
#     init_pos_field[i] = ti.Vector([initial_position[i, 0],  # x-coordinate
#                                     initial_position[i, 1],  # y-coordinate
#                                     initial_position[i, 2]]) # z-coordinate
print("initial_position",initial_position)
gui = window.get_gui()
while t <= 0.6:
    t += dt
    timesteps += 1
    mpm.advance(dt)
    if t == 0 or timesteps % renderEvery == 0:
        camera.track_user_inputs(window, movement_speed=0.3, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
        scene.ambient_light((0.5, 0.5, 0.5))
        mpm.getParticles(1e3)
        x_visualise= mpm.x_visualise
        # x_visualise[4][2] += 5
        # scene.particles(mpm.x_visualise,
        #                 radius=p_size,
        #                 color=(0.5, 0.42, 0.8),
        #                 per_vertex_color=mpm.colors)
        # print("x_visualise",x_visualise)
        scene.particles(x_visualise,
                radius=p_size,
                color=(0.5, 0.42, 0.8),
                per_vertex_color=mpm.colors)

        canvas.scene(scene)
        if rotate:
            alpha = (t % (1/frq))*frq*np.deg2rad(45)
            magnitude = (t % (1/frq)*frq)*8
            theta = np.pi + alpha
        rot_m = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        field_vector = rot_m@np.array([[magnitude], [0]])
        field[0] = 0.0
        field[1] = field_vector[0]
        field[2] = field_vector[1]

        mpm.setField(field * 1e-3)


        if frame_idx == 0 or  timesteps % 800 == 0:
            img = window.get_image_buffer_as_numpy()

            print("saved:",t)
            window.save_image(f"video/frame_{frame_idx:06d}_{t*100000}.png")
            frame_idx += 1


 
        window.show()

print("best_fitness", b_ind)
print("b_solution", best_solutions[b_index])
