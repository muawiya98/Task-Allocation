# JSContana: Malicious JavaScript detection using adaptable context analysis and key feature extraction and feature selection model for adaptive cross site scripting attack detection 

## 1. Introduction

	The efficient allocation of computational tasks to a set of nodes in distributed computing environments, such as cloud computing, edge computing, or sensor networks, is crucial for optimizing system performance. These environments often involve a multitude of tasks, each with distinct computational and communication requirements, as well as a variety of nodes with differing capabilities. Properly allocating tasks to nodes can significantly reduce execution time, minimize energy consumption, and ensure balanced workload distribution. In this context, it is essential to develop algorithms that can dynamically manage task allocation in a way that adapts to the changing conditions of the nodes and tasks.
	Simulated Annealing (SA) is a probabilistic optimization technique inspired by the annealing process in metallurgy. It has been widely used to solve complex optimization problems due to its ability to escape local optima and explore the solution space more thoroughly. When applied to task allocation, SA can iteratively improve the allocation of tasks to nodes by making small adjustments and accepting these changes based on a probability that decreases over time. This method can help in finding near-optimal solutions for the allocation problem.


## 2. Abstract

	In a distributed computing environment, we are given a set of nodes and a set of tasks. Each task can be divided into several sub-tasks, and each node has specific computational capabilities and energy consumption characteristics. The objective is to allocate these tasks to the nodes in such a way that multiple criteria are optimized, namely:
		1. Energy Distribution: Ensure an even distribution of energy consumption across all nodes to prevent any single node from depleting its energy resources prematurely.
		2. Execution Time Distribution: Ensure an even distribution of execution time across all nodes to balance the workload.
		3. Number of active nodes:  Minimize the number of active nodes.
	The problem is complicated by several factors:
		- Dynamic Node Availability: Nodes can be added or removed from the network dynamically.
		- Varied Task Requirements: Tasks have different computational and communication loads.
		- Node Capabilities: Nodes have varying processing speeds and energy capacities.
	Given these challenges, the goal is to develop an algorithm that efficiently allocates tasks to nodes, dynamically adjusts to changes in the network, and optimally balances the aforementioned criteria. The proposed solution involves the application of Simulated Annealing, combined with techniques to handle dynamic node addition and removal, to achieve a robust and efficient task allocation strategy.


## 3. Applied Techniques

	- Technique 1 Optimazation algorithm: [Simulated Annealing for Task Allocation (Optimazation)]
	- Technique 2 ML and Math: [A lot of mathes equation and ML algorithms to simulate the tasks and the nodes and the objective function and the relation between them (Math & ML)]