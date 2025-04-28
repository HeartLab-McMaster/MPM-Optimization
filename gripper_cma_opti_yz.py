# Simulation setup adapted from magneticMPM, file: gripper_cma_opti_yz.py, available at: https://github.com/joshDavy1/magneticMPM (accessed April 28, 2025).
import matplotlib.pyplot as plt
import argparse
import taichi as ti
import numpy as np
from magneticMPM.sim.mpm_class import magneticMPM
from magneticMPM.sim.loadRobot import Robot
from magneticMPM.colour_palette import generic_palette
import copy
import json
import os
import multiprocessing as mp
from threading import Lock
import GPUtil
import time
import pickle
from functools import partial
import math
import cma
import numpy as np
from cma.constraints_handler import BoundTransform


def init(l):
    global lock
    lock = l

def save_progress(data, filename="progress.csv"):
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Generation", "Best Fitness"])
        writer.writerow(data)

def assign_task_to_gpu(max_tasks_per_gpu=3, wait_time=1):

    global gpu_task_count, lock_gpu, flag
    while True:

        with lock_gpu:
            available_gpus = GPUtil.getGPUs()
            if(flag): print("available Gpu:", len(available_gpus))
            flag = False
            for gpu in available_gpus:
                gpu_id = gpu.id
                if gpu_task_count[gpu_id] < max_tasks_per_gpu:
                    gpu_task_count[gpu_id] += 1  
                    print(f"Assigned task to GPU {gpu_id} (task count: {gpu_task_count[gpu_id]})")
                    return gpu_id  

        time.sleep(wait_time)

def release_gpu(gpu_id):

    global gpu_task_count, lock_gpu
    lock_gpu.acquire()
    try:
        gpu_task_count[gpu_id] -= 1 
        print(f"Released GPU {gpu_id}. Task count is now {gpu_task_count[gpu_id]}")
    finally:
        lock_gpu.release()

def calculate_distance(final_position, initial_position):
    distance = np.linalg.norm(final_position - initial_position) * 1e6
    return distance




def run_simulation(individual, duration=0.01, dt=7e-7):
    segments = [7, 8, 9, 10, 11, 12]

    print("start sim: ", individual)
    gpu_id = assign_task_to_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    ti.init(arch=ti.cuda, default_fp=ti.f32)
    print("start simulation ", time.strftime("%H:%M:%S", time.localtime()))
    robotFile = "magneticMPM/Gripper/gripper.yaml"
    print("Generating Particles....")
    r = Robot(robotFile, ppm3=1e13, scale=1e-3)
    grid_size = 25e-3 
    dx = 0.1e-3
    g = np.array([0, 0, -9.81])
    gamma = 2e1
    offset = np.array([grid_size/4, grid_size/4, 0.6e-3]) 
    print("Initialising Variables... (this may take a while)")
    
    vectors = np.array(individual).reshape(-1, 3)

    print("vectors: ", vectors)
    r.set_magnetisation_vectors(vectors)

    mpm = magneticMPM(r, scale=1, grid_size=grid_size, dx=dx, g=g, gamma=gamma, offset=offset, colour_palette=generic_palette)

    initial_position = mpm.get_avg_position_for_segment(0).to_numpy() 

    field = np.array([0, 0 ,0])
    t = 0
    theta = 0
    magnitude = 15

    timesteps = 0
    renderEvery = 10

    timesteps = 0
    renderEvery = 20
    penalty = 1000000000
    threshold = 300
    displacement = 0.0
    broke =  False
    max_distance = 0
    while t <= duration:
        t += dt
        timesteps += 1
        mpm.advance(dt)
        if timesteps % renderEvery == 0:
            mpm.getParticles(1e3)

            rot_m = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            field_vector = rot_m@np.array([[0], [magnitude]])
            field[0] = 0.0
            field[1] = field_vector[0]
            field[2] = field_vector[1]
            mpm.setField(field * 1e-3)

    points = []
    for segment in segments:
        avg_position = mpm.get_avg_position_for_segment(segment)
        points.append([avg_position[0], avg_position[1]])
    max_distance = calculate_max_distance(points)
    
    

    ti.reset()
    print("end simulation ", time.strftime("%H:%M:%S", time.localtime()))
    release_gpu(gpu_id)

    return max_distance, t


def calculate_max_distance(points):
    max_distance = 0.0

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            point1 = np.array(points[i])
            point2 = np.array(points[j])
            distance = np.linalg.norm(point1 - point2)
            max_distance = max(max_distance, distance) 

    return max_distance * 1e6

def save_results(individual, gen, translation_distance, t, filename="data_out/gripper_simulation_results.json"):
    data = {
        "generation": gen,
        "individual": individual.tolist(),
        "translation_distance": translation_distance,
        "time": t,
    }

    lock.acquire() 
    try:
      if os.path.exists(filename):
          with open(filename, "r") as file:
              file_data = json.load(file)
      else:
          file_data = []

      file_data.append(data)

      with open(filename, "w") as file:
          json.dump(file_data, file, indent=4)
    finally:
      lock.release() 


num_pairs = 13
lock = None
lock_gpu = None
gpu_task_count = None
flag = True
timestamp = time.strftime("%Y%m%d-%H%M%S")



parser = argparse.ArgumentParser(description="Setup and execute",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-s", "--seed", type=int, default=42, help="random seed")
parser.add_argument("-n", "--num_proc", type=int, default=1, help="number of parallel procs")
parser.add_argument("--population", type=int, default=30, help="size of the population")
parser.add_argument("--generation", type=int, default=10, help="number of generations")
parser.add_argument("--tag_name", type=str, default="", help="Tag to be added to output filenames")

args, unknown = parser.parse_known_args()

population_size = args.population
num_generations = args.generation
processes = args.num_proc
np.random.seed(args.seed)

man = mp.Manager()
lock = mp.Lock()
lock_gpu = mp.Lock()
gpu_task_count = man.dict({gpu.id: 0 for gpu in GPUtil.getGPUs()})

#CMA_ES


sigma = 0.1

def objective(individual, gen=0):
    translation_distance, t = run_simulation(individual)
    save_results(individual, gen, translation_distance, t, filename=f"data_out/gripper_simulation_results_{args.tag_name}-{timestamp}.json")
    return translation_distance

print("population size: ",population_size)
print("generations: ",num_generations)

bounds = [-1, 1] 
es = cma.CMAEvolutionStrategy(num_pairs * 3 * [0], sigma, {'bounds': bounds, 'popsize': population_size})

output_file = f"data_out/gripper_cmaes_results_{args.tag_name}-{timestamp}.pkl"
output_state_file = f"data_out/gripper_cmaes_state_{args.tag_name}-{timestamp}.pkl"

results = []
with mp.Pool(processes=processes, initializer=init, initargs=(lock,)) as pool:
    for generation in range(num_generations):
        solutions = np.array(es.ask())
        objective_with_gen = partial(objective, gen=generation)

        fits = pool.map(objective_with_gen, solutions)
        es.tell(solutions, fits)
        es.disp()
        generation_data = {
            "generation": generation,
            "solutions": solutions.tolist(),
            "fitnesses": fits,
            "best_solution": es.result.xbest.tolist(),
            "best_fitness": min(fits),
        }
        results.append(generation_data)
        with open(output_file, "wb") as f:
            pickle.dump(results, f)


        state = {
            'xmean': es.mean.tolist(),
            'sigma': es.sigma,
            'cov': es.C.tolist(),
            'generation': es.countiter
        }
        with open(output_state_file, 'wb') as f:
            pickle.dump(state, f)


best_solution = es.result.xbest
print(f"Optimized individual: {best_solution}")
