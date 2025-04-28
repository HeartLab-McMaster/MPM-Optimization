# Simulation setup adapted from magneticMPM, file: runSmallScaleBot.py, available at: https://github.com/joshDavy1/magneticMPM (accessed April 28, 2025).
from deap import base, creator, tools, algorithms
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
import random


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
    difference = [(final_position[4, j] - initial_position[4, j]) *1e6 for j in range(initial_position.shape[1])]
    y_diff = difference[1] 
    return y_diff

def calculate_max_hight(position):
    max_height = np.max(position[:, 2])
    return max_height*1e6


def run_simulation(individual, duration=7e-7*10, dt=7e-7):
    print("start sim: ", individual)
    gpu_id = assign_task_to_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    ti.init(arch=ti.cuda, default_fp=ti.f32)
    print("start simulation ", time.strftime("%H:%M:%S", time.localtime()))
    robotFile = "magneticMPM/SmallScaleBot/SmallScaleBot.yaml"
    print("Generating Particles....")
    r = Robot(robotFile, ppm3=1e13, scale=1e-3)
    grid_size = 25e-3 
    dx = 0.1e-3
    g = np.array([0, 0, -9.81])
    gamma = 180 
    offset = np.array([3e-3, 3e-3, 0.2e-3])
    
    print("Initialising Variables... (this may take a while)")
    
    reshaped = np.array(individual).reshape(-1, 2)
    vectors = np.hstack([np.zeros((reshaped.shape[0], 1)), reshaped])
    r.set_magnetisation_vectors(vectors)

    mpm = magneticMPM(r, scale=1, grid_size=grid_size, dx=dx, g=g, gamma=gamma, offset=offset, colour_palette=generic_palette)

    initial_position = mpm.get_avg_positions().to_numpy() 

    field = np.array([0, 0 ,0])
    t = 0
    magnitude = 0
    theta = np.pi
    rotate = True
    frq = 10

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
            if timesteps % (renderEvery * 100) == 0:
                final_position = mpm.get_avg_positions().to_numpy()
                distance = calculate_distance(final_position, initial_position)
                hight = calculate_max_hight(final_position)
                if(distance < threshold or hight > hight_threshold):
                    broke = True
                    break

        timesteps += 1

    if broke:
        distance = penalty/t
    else:
        distance = calculate_distance(mpm.get_avg_positions().to_numpy(), initial_position)

    ti.reset()
    print("end simulation ", time.strftime("%H:%M:%S", time.localtime()))
    release_gpu(gpu_id)

    return distance, t

def save_results(individual, gen, translation_distance, t, filename="data_out/simulation_results.json"):
    data = {
        "generation": gen,
        "individual": individual,
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


num_pairs = 9
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
seed_n = args.seed
np.random.seed(seed_n)

man = mp.Manager()
lock = mp.Lock()
lock_gpu = mp.Lock()
gpu_task_count = man.dict({gpu.id: 0 for gpu in GPUtil.getGPUs()})

sigma = 0.1

def objective(individual, gen=0):
    translation_distance, t = run_simulation(individual)
    save_results(individual, gen, translation_distance, t, filename=f"data_out/simulation_results_{args.tag_name}-{timestamp}.json")
    return (float(-translation_distance),)

POP_SIZE = args.population
GENS = args.generation
CX_PROB = 0.5  
MUT_PROB = 0.2 
TOUR_SIZE = 3  
IND_SIZE = 18 
MUT_STEP = 0.1 

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -1, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def bounded_crossover(ind1, ind2, alpha=0.5):
    tools.cxBlend(ind1, ind2, alpha)
    for i in range(len(ind1)):
        ind1[i] = max(-1, min(1, ind1[i]))  
        ind2[i] = max(-1, min(1, ind2[i])) 

def GA_mutation(ind, step=MUT_STEP, indpb=0.2):
    for i in range(len(ind)):
        if random.random() < indpb:
            ind[i] += step if random.random() < 0.5 else -step
            ind[i] = max(-1, min(1, ind[i]))
    return ind,

toolbox.register("mate", bounded_crossover)
toolbox.register("mutate", GA_mutation)
toolbox.register("select", tools.selTournament, tournsize=TOUR_SIZE)
toolbox.register("evaluate", objective)

pop = toolbox.population(n=POP_SIZE)

man = mp.Manager()
lock = mp.Lock()
lock_gpu = mp.Lock()
gpu_task_count = man.dict({gpu.id: 0 for gpu in GPUtil.getGPUs()})

output_file = f"data_out/ga_results_{args.tag_name}-{time.strftime('%Y%m%d-%H%M%S')}.pkl"
results = []

with mp.Pool(processes=args.num_proc, initializer=init, initargs=(lock,)) as pool:
    toolbox.register("map", pool.map)
    for gen in range(GENS):
        offspring = tools.selTournament(pop, len(pop), tournsize=3)
        offspring = list(map(toolbox.clone, offspring))
        
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CX_PROB:
                toolbox.mate(child1, child2)
                del child1.fitness.values, child2.fitness.values
        
        for mutant in offspring:
            if random.random() < MUT_PROB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        pop[:] = offspring
        best_solution = tools.selBest(pop, 1)[0]
        results.append({
            "generation": gen,
            "best_solution": best_solution[:],
            "best_fitness": best_solution.fitness.values[0],
        })

        with open(output_file, "wb") as f:
            pickle.dump(results, f)

print(f"Optimized individual: {best_solution}, Fitness: {best_solution.fitness.values[0]}")
