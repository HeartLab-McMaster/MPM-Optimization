# Simulation setup adapted from magneticMPM, file: runSmallScaleBot.py, available at: https://github.com/joshDavy1/magneticMPM (accessed April 28, 2025).
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
import csv

def init(l):
    global lock
    lock = l

# Appends the generation number and best fitness score to a CSV file for progress tracking
def save_progress(data, filename="progress.csv"):
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Generation", "Best Fitness"])
        writer.writerow(data)

# GPU Managment
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

# Simulation Helper Functions
def calculate_distance(final_position, initial_position):
    difference = [(final_position[4, j] - initial_position[4, j]) *1e6 for j in range(initial_position.shape[1])]
    y_diff = difference[1] 
    return y_diff

def calculate_max_hight(position):
    max_height = np.max(position[:, 2])
    return max_height*1e6

#Simulation Function -- objective helper function
def run_simulation(individual, duration=0.2, dt=7e-7):  # Magnetization vectors, simulation time, sim time step
    print("start sim: ", individual)
    
    # GPU setup
    gpu_id = assign_task_to_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    ti.init(arch=ti.cuda, default_fp=ti.f32)
    print("start simulation ", time.strftime("%H:%M:%S", time.localtime()))
    
    # Loading Robot File and Setting Physics Parameters
    robotFile = "magneticMPM/SmallScaleBot/SmallScaleBot.yaml"
    print("Generating Particles....")
    r = Robot(robotFile, ppm3=1e13, scale=1e-3)
    grid_size = 25e-3 
    dx = 0.1e-3
    g = np.array([0, 0, -9.81])                 # Gravity
    gamma = 180                                 # magnetic gamma parameter
    offset = np.array([3e-3, 3e-3, 0.2e-3])     # placement in simulation grid
    print("Initialising Variables... (this may take a while)")
    
    #  Apply Magnetization Vector to the Robot
    reshaped = np.array(individual).reshape(-1, 2)                      # flat list to 2D array
    vectors = np.hstack([np.zeros((reshaped.shape[0], 1)), reshaped])   # create 3D vector with x=0 (y-Z plane)
    r.set_magnetisation_vectors(vectors)                                # apply to robot


    # Creating Simulation Environment
    mpm = magneticMPM(r, scale=1, grid_size=grid_size, dx=dx, g=g, gamma=gamma, offset=offset, colour_palette=generic_palette)
    initial_position = mpm.get_avg_positions().to_numpy()       # inital center of mass for distance calc later

    # field rotation set up
    field = np.array([0, 0 ,0])
    t = 0
    magnitude = 0
    theta = np.pi
    rotate = True
    frq = 10

    # Simulation Loop
    timesteps = 0
    renderEvery = 20
    penalty = -3000
    threshold = -300
    hight_threshold = 1840
    distance = 0.0
    broke =  False
    while t <= duration:
        t += dt
        mpm.advance(dt)
        if timesteps % renderEvery == 0:
            # Compute rotated magnetic field based on time t
            if rotate:
                alpha = (t % (1/frq))*frq*np.deg2rad(45)
                magnitude = (t % (1/frq)*frq)*8
                theta = np.pi + alpha
            rot_m = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])  #rotation matrix
            field_vector = rot_m@np.array([[magnitude], [0]]) #2D vector rotation, same mag new direction in y-z plane
            field[0] = 0.0
            field[1] = field_vector[0]
            field[2] = field_vector[1]
            mpm.setField(field * 1e-3)    # apply rotated magnetic field to simulation

            if timesteps % (renderEvery * 100) == 0:
                final_position = mpm.get_avg_positions().to_numpy()
                distance = calculate_distance(final_position, initial_position)
                hight = calculate_max_hight(final_position)

                # Check termination criteria
                if(distance < threshold or hight > hight_threshold):
                    broke = True
                    break

        timesteps += 1

    # Post-Simulation Fitness Scoring
    if broke:
        distance = penalty/t
    else:
        distance = calculate_distance(mpm.get_avg_positions().to_numpy(), initial_position)

    # Clean up
    ti.reset()
    print("end simulation ", time.strftime("%H:%M:%S", time.localtime()))
    release_gpu(gpu_id)

    return distance, t

# Saving results of each individuals performance -- Objective helper function
def save_results(individual, gen, translation_distance, t, filename="data_out/simulation_results.json"):
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


# Main Script Configuration
# Constants
num_pairs = 9
lock = None
lock_gpu = None
gpu_task_count = None
flag = True
timestamp = time.strftime("%Y%m%d-%H%M%S")


# Parsing command-line Arguments
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
seed_n = args.seed
np.random.seed(seed_n)

# Initialize GPU task tracking
man = mp.Manager()
lock = mp.Lock()
lock_gpu = mp.Lock()
gpu_task_count = man.dict({gpu.id: 0 for gpu in GPUtil.getGPUs()})




#CMA_ES _____________________________________________________________________________________________________________________________________________________

sigma = 0.1  # Step Size, how much candidate solutions vary from initial point

def objective(individual, gen=0):
    translation_distance, t = run_simulation(individual)
    save_results(individual, gen, translation_distance, t, filename=f"data_out/simulation_results_{args.tag_name}-{timestamp}.json")
    return -translation_distance

print("population size: ",population_size)
print("generations: ",num_generations)

# defining intial population
bounds = [-1, 1] 
initial = np.random.uniform(low=bounds[0], high=bounds[1], size=num_pairs * 2)  # size = total vector components for 9 magnet pairs in 2D (y-z plane)
print(f"initial individual: {initial}")

# Initialize CMA-ES Optimizer
es = cma.CMAEvolutionStrategy(initial, sigma, {'bounds': bounds, 'popsize': population_size})

# Evolution Loop
output_file = f"data_out/cmaes_results_{args.tag_name}-{timestamp}.pkl"
output_state_file = f"data_out/cmaes_state_{args.tag_name}-{timestamp}.pkl"
cma_state_file = f"data_out/cma_state_{args.tag_name}-{timestamp}.pkl"
results = []
with mp.Pool(processes=processes, initializer=init, initargs=(lock,)) as pool: # setting up multiprocessing pool with shared lock
    while True:
        for generation in range(num_generations):
            solutions = np.array(es.ask())                           # generates a population of candidate solutions
            objective_with_gen = partial(objective, gen=generation)

            fits = pool.map(objective_with_gen, solutions)           # evaluate solutions in parallel using objective function
            es.tell(solutions, fits)                                 # update cma model based on fitness values
            es.disp()                                                # print out generation statistics

            # Saving results of generation (solutions + fitness score)
            best_fitness = -min(fits)
            generation_data = {
                "generation": generation,
                "solutions": solutions.tolist(),
                "fitnesses": fits,
                "best_solution": es.result.xbest.tolist(),
                "best_fitness": best_fitness,
            }
            results.append(generation_data)                         # save results locally
            with open(output_file, "wb") as f:
                pickle.dump(results, f)

            with open(cma_state_file, 'wb') as f:
                pickle.dump(es, f)
            
            # Restart Strategy
            if generation == 19 and best_fitness < 500:
                print("Restarting optimization: Best fitness did not reach the threshold. best fitness: ",best_fitness)

                # delete logs
                for file in [output_file, output_state_file, cma_state_file]:
                    if os.path.exists(file):
                        os.remove(file)

                # increase seed and new random initial population
                seed_n += 1
                np.random.seed(seed_n)
                initial = np.random.uniform(low=bounds[0], high=bounds[1], size=num_pairs * 2)
                print(f"initial individual: {initial}")

                # restart optimization and clear results
                es = cma.CMAEvolutionStrategy(initial, sigma, {'bounds': bounds, 'popsize': population_size})
                results = []
                break  # terminates generation
        else:  # completed all generations (breaks while true loop)
            break

# reporting optimal output
best_solution = es.result.xbest
print(f"Optimized individual: {best_solution}")
print("seed increased by: ", seed_n - args.seed )
