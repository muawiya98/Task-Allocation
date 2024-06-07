import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import numpy as np
import random
import pickle
import heapq
import sys
import os
import gc

np.random.seed(42)
random.seed(42)


from Codes.OptimazationUnit.Configuration import node_removal_id, node_addition_id, number_of_iterations, time_step, th_size, save_object, load_object, W1, W2, W3
from Codes.OptimazationUnit.CostFunction import CostFunction
from Codes.OptimazationUnit.Task import Task
from Codes.OptimazationUnit.Node import Node

def make_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def tasks_genration_model(number_of_tasks):
    return [Task() for i in range(number_of_tasks)]

def nodes_generation_model(number_of_nodes):
    return [Node() for i in range(number_of_nodes)]

def generate_initial_solution(nodes_set, tasks_set):
    solution = np.zeros((len(nodes_set), len(tasks_set), len(tasks_set[-1].sub_tasks)))
    for k in range(len(tasks_set[-1].sub_tasks)):
        for j, task in enumerate(tasks_set):
            sub_task = task.sub_tasks[k]
            for i, node in enumerate(nodes_set):
                if (not node.is_terminated) and (not sub_task.is_allocated):
                    if node.computational_capacities>=node.used_computational+sub_task.computation_load:
                        if node.start_time==0:
                            node.start_time=((sub_task.computation_load/node.processing_speed)*sub_task.priorty)
                        else:
                            node.start_time+=(sub_task.computation_load/node.processing_speed)
                        node.used_computational+=sub_task.computation_load
                        sub_task.is_allocated = True
                        solution[i][j][k]=1
                        node.tasks.append((j, k))
    return solution, nodes_set, tasks_set

def active_nodes(nodes_set):
  return [node for node in nodes_set if len(node.tasks)>0]

def evaluate_function(nodes_set, tasks_set, cost_function, single=False):
    the_active_nodes = active_nodes(nodes_set)
    execution_time = cost_function.totale_execution_time(the_active_nodes, tasks_set)
    execution_energy = cost_function.totale_execution_energy(the_active_nodes, tasks_set)
    energy_distribution = cost_function.calculate_energy_distribution(the_active_nodes)
    execution_time_distribution = cost_function.calculate_execution_time_distribution(the_active_nodes)
    if single:
        return ((W1*execution_time_distribution) + (W2*energy_distribution) + (W3*len(the_active_nodes)),), execution_time_distribution, energy_distribution, len(the_active_nodes)
    return execution_time_distribution, energy_distribution, len(the_active_nodes)

def task_allocation_ratio(tasks_set):
  number_of_sub_tasks, number_of_alloctaed_sub_task = len(tasks_set)*3, 0
  for i, task in enumerate(tasks_set):
    for sub_task in task.sub_tasks:
      if sub_task.is_allocated:number_of_alloctaed_sub_task+=1
  return number_of_alloctaed_sub_task/number_of_sub_tasks

def allocated_tasks(tasks_set):
    is_allocated, allocated_tasks, allocated_sub_tasks = [], [], []
    for task in tasks_set:
        for sub in task.sub_tasks:
            allocated_sub_tasks.append(sub)
            is_allocated.append(sub.is_allocated)
        if np.all(is_allocated):
            allocated_tasks.append(task)
    return allocated_tasks, allocated_sub_tasks

def identify_best_node_for_remove(nodes_set):
    nodes_speed, indeces = [], []
    for index, node in enumerate(nodes_set):
        if (not node.is_terminated) and (node.tasks!=[]):
            nodes_speed.append(node.processing_speed)
            indeces.append(index)
    if nodes_speed==[]:return 0
    return indeces[nodes_speed.index(min(nodes_speed))]

def find_best_node_for_reallocate(nodes_set, task, node_for_removal_index):
    computation_capacities, indeces = [], []
    for index, node in enumerate(nodes_set):
        if index==node_for_removal_index:continue
        c = node.computational_capacities-(node.used_computational+task.computation_load)
        if c>0 and (not node.is_terminated):
            computation_capacities.append(c)
            indeces.append(index)
    if indeces==[]:return -1
    return indeces[computation_capacities.index(min(computation_capacities))]

def node_removal(nodes_set, tasks_set, current_solution, cost_function):
    node_for_remove_index = identify_best_node_for_remove(nodes_set)
    try:
        allocated_tasks = nodes_set[node_for_remove_index].tasks
    except:
        allocated_tasks = nodes_set[node_for_remove_index].tasks
    if allocated_tasks==[]:
      for i, task in enumerate(allocated_tasks):
          new_node_index = find_best_node_for_reallocate(nodes_set, tasks_set[task[0]].sub_tasks[task[1]], node_for_remove_index)
          if new_node_index!=-1:
              nodes_set[new_node_index].start_time+=(tasks_set[task[0]].sub_tasks[task[1]].computation_load/nodes_set[new_node_index].processing_speed)
              nodes_set[node_for_remove_index].start_time-=(tasks_set[task[0]].sub_tasks[task[1]].computation_load/nodes_set[node_for_remove_index].processing_speed)
              nodes_set[new_node_index].tasks.append(task)
              current_solution[new_node_index][task[0]][task[1]]=1
              current_solution[node_for_remove_index][task[0]][task[1]]=0
              cost_function.reduce_execution_time(nodes_set[node_for_remove_index], node_for_remove_index, tasks_set[task[0]].sub_tasks[task[1]], task[0])
              del nodes_set[node_for_remove_index].tasks[i]
    if nodes_set[node_for_remove_index].tasks==[]:nodes_set[node_for_remove_index].is_terminated=False
    return nodes_set, tasks_set, current_solution

def allocate_tasks_not_allocated(nodes_set, tasks_set, new_node, current_solution):
    sub_solution, allocate = np.zeros((1, len(tasks_set), len(tasks_set[-1].sub_tasks))), False
    for k in range(len(tasks_set[-1].sub_tasks)):
        for j, task in enumerate(tasks_set):
            sub_task = task.sub_tasks[k]
            if not sub_task.is_allocated:
                if new_node.computational_capacities>=new_node.used_computational+sub_task.computation_load:
                    new_node.used_computational+=sub_task.computation_load
                    sub_task.is_allocated = True
                    sub_solution[0][j][k]=1
                    new_node.tasks.append((j, k))
                    allocate=True
                    if new_node.start_time==0:
                        new_node.start_time=((sub_task.computation_load/new_node.processing_speed)*sub_task.priorty)
                    else:
                        new_node.start_time+=(sub_task.computation_load/new_node.processing_speed)
                new_node.tasks = list(set(new_node.tasks))
    if allocate:
        nodes_set.append(new_node)
        current_solution = np.concatenate((current_solution, sub_solution), axis=0)
    return nodes_set, tasks_set, current_solution, allocate

def find_best_task_for_reallocate(nodes_set, tasks_set, new_node, current_solution, cost_function):
    sub_solution = np.zeros((1, len(tasks_set), len(tasks_set[-1].sub_tasks)))
    number_of_task = []
    for index, node in enumerate(nodes_set):
        number_of_task.append(len(node.tasks))
    max_five_indices = heapq.nlargest(5, range(len(number_of_task)), key=lambda i: number_of_task[i])
    for ind in max_five_indices:
        if number_of_task[ind]>=2 and new_node.computational_capacities-new_node.used_computational>=0:
            task_index = nodes_set[ind].tasks[0][0]
            sub_tasl_index = nodes_set[ind].tasks[0][1]
            sub_task = tasks_set[task_index].sub_tasks[sub_tasl_index]
            if new_node.computational_capacities>=new_node.used_computational+sub_task.computation_load:
                new_node.used_computational+=sub_task.computation_load
                sub_solution[0][task_index][sub_tasl_index] = 1
                current_solution[ind][task_index][sub_tasl_index] = 0
                new_node.tasks.append((task_index, sub_tasl_index))
                sub_task.is_allocated=True
                if new_node.start_time==0:
                    new_node.start_time=((sub_task.computation_load/new_node.processing_speed)*sub_task.priorty)
                else:
                    new_node.start_time+=(sub_task.computation_load/new_node.processing_speed)
                nodes_set[ind].start_time-=(sub_task.computation_load/nodes_set[ind].processing_speed)
                del nodes_set[ind].tasks[0]
                cost_function.reduce_execution_time(nodes_set[ind], ind, sub_task, task_index)
            new_node.tasks = list(set(new_node.tasks))
    if new_node.tasks!=[]:
        nodes_set.append(new_node)
        current_solution = np.concatenate((current_solution, sub_solution), axis=0)
    return nodes_set, tasks_set, current_solution

def dominates(solution1, solution2):
    return all(solution1[i] <= solution2[i] for i in range(len(solution1))) and any(solution1[i] < solution2[i] for i in range(len(solution1)))

def is_non_dominated(solution, population):
    for ind in population:
        if dominates(ind, solution):
            return False
    return True

def non_dominated_sort(population):
    fronts = []
    dominating_count = [0] * len(population)
    dominated_solutions = [[] for _ in range(len(population))]
    for i, p in enumerate(population):
        for j, q in enumerate(population):
            if dominates(p, q):
                dominating_count[i] += 1
            elif dominates(q, p):
                dominated_solutions[i].append(j)
    current_front = []
    for i, count in enumerate(dominating_count):
        if count == 0:
            current_front.append(i)
    fronts.append(current_front)
    while current_front:
        next_front = []
        for i in current_front:
            for j in dominated_solutions[i]:
                dominating_count[j] -= 1
                if dominating_count[j] == 0:
                    next_front.append(j)
        if next_front:
            fronts.append(next_front)
        current_front = next_front
    return fronts

def node_addtion(nodes_set, tasks_set, current_solution, cost_function):
    number_of_nodes = len(nodes_set)
    new_node = Node(number_of_nodes)
    cost_function.expand_matrixes()
    nodes_set, tasks_set, current_solution, allocate = allocate_tasks_not_allocated(nodes_set, tasks_set, new_node, current_solution)
    if not allocate:
        nodes_set, tasks_set, current_solution = find_best_task_for_reallocate(nodes_set, tasks_set, new_node, current_solution, cost_function)
    number_of_nodes = len(nodes_set)
    return nodes_set, tasks_set, current_solution

def generate_nieghbor_solution(nodes_set, tasks_set, current_solution, cost_function):
    operation_id = random.randint(0, 1)
    if operation_id == node_removal_id:
        nodes_set, tasks_set, current_solution = node_removal(nodes_set, tasks_set, current_solution, cost_function)
    elif operation_id == node_addition_id:
        nodes_set, tasks_set, current_solution = node_addtion(nodes_set, tasks_set, current_solution, cost_function)
    return nodes_set, tasks_set, current_solution

def applay_solution(nodes_set, tasks_set, solution, cost_function):
    cost_function.set_matrixes(len(nodes_set), len(tasks_set))
    for i, node in enumerate(solution):
        nodes_set[i].tasks = []
        nodes_set[i].used_computational = 0
        nodes_set[i].start_time = 0
        for j, task in enumerate(node):
            for k, sub_task in enumerate(task):
                if sub_task==1:
                    nodes_set[i].is_terminated = False
                    tasks_set[j].sub_tasks[k].is_allocated = True
                    nodes_set[i].tasks.append((j, k))
                    nodes_set[i].used_computational+=tasks_set[j].sub_tasks[k].computation_load
                    if nodes_set[i].start_time==0:
                        nodes_set[i].start_time=((tasks_set[j].sub_tasks[k].computation_load/nodes_set[i].processing_speed)*tasks_set[j].sub_tasks[k].priorty)
                    else:
                        nodes_set[i].start_time+=(tasks_set[j].sub_tasks[k].computation_load/nodes_set[i].processing_speed)
    return nodes_set, tasks_set, solution

def crowding_distance(solutions):
    solutions = np.array(solutions)
    num_solutions = len(solutions)
    num_objectives = len(solutions[0])
    crowding_distances = np.zeros(num_solutions)
    sorted_indices = np.argsort(solutions, axis=0)
    for obj_index in range(num_objectives):
        sorted_solutions = solutions[sorted_indices[:, obj_index]]
        crowding_distances[sorted_indices[0, obj_index]] = np.inf
        crowding_distances[sorted_indices[num_solutions - 1, obj_index]] = np.inf
        min_obj_val = sorted_solutions[0, obj_index]
        max_obj_val = sorted_solutions[num_solutions - 1, obj_index]
        if max_obj_val == min_obj_val:
            continue  # Avoid division by zero
        for i in range(1, num_solutions - 1):
            crowding_distances[sorted_indices[i, obj_index]] += \
                (sorted_solutions[i + 1, obj_index] - sorted_solutions[i - 1, obj_index]) / (max_obj_val - min_obj_val)
    return [index for index, value in enumerate(crowding_distances) if value <= 0.2]

def remove_duplicates_in_order(lst):
    seen = set()
    result = []
    for i, item in enumerate(lst):
        if item not in seen:
            seen.add(item)
            result.append(i)
    return result

def delete_items_randomly(lst, num_items):
    if num_items >= len(lst):
        return []
    indices_to_delete = random.sample(range(len(lst)), num_items)
    indices_to_delete.sort(reverse=True)
    return indices_to_delete

def simulated_annealing(tasks_set, nodes_set, cost_function, number_of_iterations, th_size=100):
    pareto_front, objective_values, tasks_sets, nodes_sets = [], [], [], [] # tasks_allocation_ratio , []
    current_solution, nodes_set, tasks_set = generate_initial_solution(nodes_set.copy(), tasks_set.copy())
    objective_values.append(evaluate_function(nodes_set.copy(), tasks_set.copy(), cost_function))
    # tasks_allocation_ratio.append(task_allocation_ratio(tasks_set.copy()))
    pareto_front.append(current_solution.copy())
    nodes_sets.append(nodes_set.copy())
    tasks_sets.append(tasks_set.copy())
    for iteration in range(number_of_iterations):
      for index, _ in enumerate(pareto_front):
          none_duplicates = remove_duplicates_in_order(objective_values)
          objective_values = [objective_values[i] for i in none_duplicates]
          pareto_front = [pareto_front[i].copy() for i in none_duplicates]
          # tasks_allocation_ratio = [tasks_allocation_ratio[i] for i in none_duplicates]
          nodes_sets = [nodes_sets[i].copy() for i in none_duplicates]
          tasks_sets = [tasks_sets[i].copy() for i in none_duplicates]
          if index>=len(pareto_front):break
          current_solution, nodes_set, tasks_set = pareto_front[index].copy(), nodes_sets[index].copy(), tasks_sets[index].copy()
          nodes_set, tasks_set, _ = applay_solution(nodes_set, tasks_set, current_solution, cost_function)
          nodes_set, tasks_set, current_solution = generate_nieghbor_solution(nodes_set.copy(), tasks_set.copy(), current_solution.copy(),cost_function)
          objective_values.append(evaluate_function(nodes_set.copy(), tasks_set.copy(), cost_function))
          # tasks_allocation_ratio.append(task_allocation_ratio(tasks_set.copy()))
          pareto_front.append(current_solution.copy())
          nodes_sets.append(nodes_set.copy())
          tasks_sets.append(tasks_set.copy())
          non_dominated_indices = non_dominated_sort(objective_values)
          objective_values = [objective_values[i] for i in non_dominated_indices[0]]
          pareto_front = [pareto_front[i].copy() for i in non_dominated_indices[0]]
          # tasks_allocation_ratio = [tasks_allocation_ratio[i] for i in non_dominated_indices[0]]
          nodes_sets = [nodes_sets[i].copy() for i in non_dominated_indices[0]]
          tasks_sets = [tasks_sets[i].copy() for i in non_dominated_indices[0]]
          if len(pareto_front)>th_size:
            crowding_distance_indices = crowding_distance(objective_values)
            objective_values = [objective_values[i] for i in crowding_distance_indices]
            # tasks_allocation_ratio = [tasks_allocation_ratio[i] for i in crowding_distance_indices]
            pareto_front = [pareto_front[i].copy() for i in crowding_distance_indices]
            nodes_sets = [nodes_sets[i].copy() for i in crowding_distance_indices]
            tasks_sets = [tasks_sets[i].copy() for i in crowding_distance_indices]
    return pareto_front, objective_values, nodes_sets, tasks_sets


def single_simulated_annealing(tasks_set, nodes_set, cost_function, number_of_iterations, th_size=100):
    pareto_front, objective_values, tasks_sets, nodes_sets = [], [], [], []
    current_solution, nodes_set, tasks_set = generate_initial_solution(nodes_set.copy(), tasks_set.copy())
    objective_values.append(evaluate_function(nodes_set.copy(), tasks_set.copy(), cost_function, True))
    pareto_front.append(current_solution.copy())
    nodes_sets.append(nodes_set.copy())
    tasks_sets.append(tasks_set.copy())
    for iteration in range(number_of_iterations):
      for index, _ in enumerate(pareto_front):
          none_duplicates = remove_duplicates_in_order(objective_values)
          objective_values = [objective_values[i] for i in none_duplicates]
          pareto_front = [pareto_front[i].copy() for i in none_duplicates]
          nodes_sets = [nodes_sets[i].copy() for i in none_duplicates]
          tasks_sets = [tasks_sets[i].copy() for i in none_duplicates]
          if index>=len(pareto_front):break
          current_solution, nodes_set, tasks_set = pareto_front[index].copy(), nodes_sets[index].copy(), tasks_sets[index].copy()
          nodes_set, tasks_set, _ = applay_solution(nodes_set, tasks_set, current_solution, cost_function)
          nodes_set, tasks_set, current_solution = generate_nieghbor_solution(nodes_set.copy(), tasks_set.copy(), current_solution.copy(), cost_function)
          objective_values.append(evaluate_function(nodes_set.copy(), tasks_set.copy(), cost_function, True))
          pareto_front.append(current_solution.copy())
          nodes_sets.append(nodes_set.copy())
          tasks_sets.append(tasks_set.copy())
          if len(pareto_front)>th_size:
            crowding_distance_indices = delete_items_randomly(objective_values, len(pareto_front)-th_size)
            objective_values = [objective_values[i] for i in crowding_distance_indices]
            pareto_front = [pareto_front[i].copy() for i in crowding_distance_indices]
            nodes_sets = [nodes_sets[i].copy() for i in crowding_distance_indices]
            tasks_sets = [tasks_sets[i].copy() for i in crowding_distance_indices]
    return pareto_front, objective_values, nodes_sets, tasks_sets

def random_method(nodes_set, tasks_set, cost_function, number_of_iterations):
    shape = (len(nodes_set), len(tasks_set), (len(tasks_set[-1].sub_tasks)))
    pareto_front, objective_values, tasks_sets, nodes_sets = [], [], [], [] # tasks_allocation_ratio , []
    for i in range(number_of_iterations):
        operation_id = random.randint(0, 1)
        if operation_id == node_removal_id:
            # del nodes_set[random.randint(0, len(nodes_set)-1)]
            shape = (len(nodes_set), len(tasks_set), (len(tasks_set[-1].sub_tasks)))
        elif operation_id == node_addition_id:
            nodes_set.append(Node(len(nodes_set)))
            cost_function.set_matrixes(len(nodes_set), len(tasks_set))
            shape = (len(nodes_set), len(tasks_set), (len(tasks_set[-1].sub_tasks)))
            number_of_nodes = len(nodes_set)
        current_solution = np.random.randint(2, size=shape)
        nodes_set, tasks_set, current_solution = applay_solution(nodes_set, tasks_set, current_solution, cost_function)
        pareto_front.append(current_solution)
        objective_values.append(evaluate_function(nodes_set, tasks_set, cost_function))
        nodes_sets.append(nodes_set.copy())
        tasks_sets.append(tasks_set.copy())
        # tasks_allocation_ratio.append(task_allocation_ratio(tasks_set))
    return pareto_front, objective_values, nodes_sets, tasks_sets


def greedy_method(nodes_set, tasks_set, cost_function, number_of_iterations):
    pareto_front, objective_values, tasks_sets, nodes_sets = [], [], [], [] # tasks_allocation_ratio , []
    for i in range(number_of_iterations):
        operation_id = random.randint(0, 1)
        if operation_id == node_removal_id:
            # del nodes_set[random.randint(0, len(nodes_set)-1)]
            number_of_nodes = len(nodes_set)
        elif operation_id == node_addition_id:
            nodes_set.append(Node(len(nodes_set)))
            number_of_nodes = len(nodes_set)
            cost_function.set_matrixes(len(nodes_set), len(tasks_set))
        
        nodes_set = list(set(nodes_set))
        nodes_set = random.sample(nodes_set, len(nodes_set))
        
        current_solution, nodes_set, tasks_set = generate_initial_solution(nodes_set, tasks_set)
        pareto_front.append(current_solution)
        objective_values.append(evaluate_function(nodes_set, tasks_set, cost_function))
        nodes_sets.append(nodes_set.copy())
        tasks_sets.append(tasks_set.copy())
        # tasks_allocation_ratio.append(task_allocation_ratio(tasks_set))
    return pareto_front, objective_values, nodes_sets, tasks_sets

def get_best_objectives(objective_values):
    obj1 = [sol[0] for sol in objective_values]
    obj2 = [sol[1] for sol in objective_values]
    obj3 = [sol[2] for sol in objective_values]
    return obj1.index(min(obj1)), obj2.index(min(obj2)), obj3.index(min(obj3))

def index_of_tuple_with_greatest_min(tuples):
    max_count = -1
    index = -1
    for i, t in enumerate(tuples):
        min_val = min(t)
        count = t.count(min_val)
        if count > max_count:
            max_count = count
            index = i
        elif count == max_count:
            if sum(t) < sum(tuples[index]):
                index = i
    return index

def select_the_final_solution(nodes_sets, tasks_sets, objective_values):
    if len(objective_values[0])>3:
        objective_values = [(y[1], y[2], y[3]) for y in objective_values]
    while True:
        index = index_of_tuple_with_greatest_min(objective_values)
        tasks = tasks_sets[index]
        nodes = nodes_sets[index]
        bool_allocate = False
        for t in tasks:
            for tt in t.sub_tasks:
                if not tt.is_allocated:bool_allocate=True
        if not bool_allocate:break
        else:
            objective_values[index] = (10000000000, 10000000000, 10000000000)
    return nodes, tasks 

def reschedulezation(nodes_set, tasks_set):
    new_tasks_set, new_nodes_set = [], []
    for task_index, task in enumerate(tasks_set):
        temp_node, temp_sub_task = [], []
        for node in nodes_set:
            for t in node.tasks:
                if t[0]==task_index:
                   temp_node.append(node)
                   temp_sub_task.append(t[1])
        time_cost = 0
        computation_cost = []
        for node, sub_task in zip(temp_node, temp_sub_task):
            time_cost+=task.sub_tasks[sub_task].computation_load/node.processing_speed
            computation_cost.append(time_step * node.processing_speed)
        if time_cost>time_step:
            task.computation_load-=sum(computation_cost)
            for i, i_s_t in enumerate(temp_sub_task):
                task.sub_tasks[i_s_t].computation_load -= computation_cost[i]
            new_tasks_set.append(task)
    new_tasks_set = new_tasks_set + tasks_genration_model(random.randint(10, 30))
    for node in nodes_set:
        if not node.is_terminated:
            node.used_computational = 0
            node.start_time = 0
            node.tasks = []
            new_nodes_set.append(node)
    return new_nodes_set, new_tasks_set

def simulated_annealing_main(nodes_set, tasks_set, step, simulated_annealing_path):
    cost_function = CostFunction(len(nodes_set), len(tasks_set))
    pareto_front, objective_values, nodes_sets, tasks_sets = simulated_annealing(tasks_set, nodes_set, cost_function, number_of_iterations)
    save_object(pareto_front, 'pareto_front_'+str(step), simulated_annealing_path)
    save_object(objective_values, 'objective_values_'+str(step), simulated_annealing_path)
    nodes_set, tasks_set = select_the_final_solution(nodes_sets, tasks_sets, objective_values)
    nodes_set, tasks_set = reschedulezation(nodes_set, tasks_set)
    gc.collect()
    return nodes_set, tasks_set

def random_main(nodes_set, tasks_set, step, random_path):
    cost_function = CostFunction(len(nodes_set), len(tasks_set))
    random_pareto_front, random_objective_values, nodes_sets, tasks_sets = random_method(nodes_set, tasks_set, cost_function, number_of_iterations)
    save_object(random_pareto_front, 'pareto_front_'+str(step), random_path)
    save_object(random_objective_values, 'objective_values_'+str(step), random_path)
    nodes_set, tasks_set = select_the_final_solution(nodes_sets, tasks_sets, random_objective_values)
    nodes_set, tasks_set = reschedulezation(nodes_set, tasks_set)
    gc.collect()
    return nodes_set, tasks_set

def greedy_main(nodes_set, tasks_set, step, greedy_path):
    cost_function = CostFunction(len(nodes_set), len(tasks_set))
    greedy_pareto_front, greedy_objective_values, nodes_sets, tasks_sets = greedy_method(nodes_set, tasks_set, cost_function, number_of_iterations)
    save_object(greedy_pareto_front, 'pareto_front_'+str(step), greedy_path)
    save_object(greedy_objective_values, 'objective_values_'+str(step), greedy_path)
    nodes_set, tasks_set = select_the_final_solution(nodes_sets, tasks_sets, greedy_objective_values)
    nodes_set, tasks_set = reschedulezation(nodes_set, tasks_set)
    gc.collect()
    return nodes_set, tasks_set

def single_simulated_annealing_main(nodes_set, tasks_set, step, single_simulated_annealing_path):
    cost_function = CostFunction(len(nodes_set), len(tasks_set))
    single_pareto_front, single_objective_values, nodes_sets, tasks_sets = single_simulated_annealing(tasks_set, nodes_set, cost_function, number_of_iterations)
    save_object(single_pareto_front, 'pareto_front_'+str(step), single_simulated_annealing_path)
    save_object(single_objective_values, 'objective_values_'+str(step), single_simulated_annealing_path)
    nodes_set, tasks_set = select_the_final_solution(nodes_sets, tasks_sets, single_objective_values)
    nodes_set, tasks_set = reschedulezation(nodes_set, tasks_set)
    gc.collect()
    return nodes_set, tasks_set

