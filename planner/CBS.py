from planner.Astar import *

from queue import PriorityQueue
import numpy as np
import core.Constant as constant


def cbs(env, starts, goals):
	pq = PriorityQueue()
	root = Constraint()
	root.find_paths(env, starts, goals)
	mask = np.array(root.cost) != float('inf')
	root_cost = np.sum(np.array(root.cost)[mask])
	pq.put_nowait((root_cost, root.depth, root))
	iteration = 0

	while not pq.empty():
		if iteration > constant.CBS_MAX_ITER:
			break

		cost, _, x = pq.get_nowait()
		# print("sol", x.solution)
		conflict = find_conflict(x.solution, env)
		if conflict is None:
			return x.solution, cost
		else:
			for agent in ['agent1', 'agent2']:
				child = x.create_child()
				if not conflict['transition']:
					is_new_constraint = child.add_constraint(agent=conflict[agent], t=conflict['t'], node=conflict['node'])
				else:
					if agent == 'agent1':
						is_new_constraint = child.add_transition_constraint(agent=conflict['agent1'], t=conflict['t'], node=conflict['node'], lastnode=conflict['lastnode'])
					else:
						is_new_constraint = child.add_transition_constraint(agent=conflict['agent2'], t=conflict['t'], node=conflict['lastnode'], lastnode=conflict['node'])
				if is_new_constraint:
					child.find_paths(env, starts, goals)
					child_cost = np.sum(np.array(child.cost)[mask])
					pq.put_nowait((child_cost, child.depth+np.random.rand(1)[0], child))
		iteration += 1

	return [None]*len(starts), cost

class Constraint:

	def __init__(self):
		self.constraints = {}
		self.transition_constraints = {}
		self.children = []
		self.solution = None
		self.cost = None
		self.depth = 0
		self.T = 0

	def create_child(self):
		child = Constraint()
		child.constraints = self.constraints.copy()
		child.transition_constraints = self.transition_constraints.copy()
		child.depth = self.depth + 1
		child.T = self.T
		self.children.append(child)
		return child

	def add_constraint(self, agent, t, node):
		if agent not in self.constraints:
			self.constraints[agent] = {}
		if t not in self.constraints[agent]:
			self.constraints[agent][t] = set()
		if node in self.constraints[agent][t]:
			return False
		self.constraints[agent][t].add(node)
		self.T = max(self.T, t)
		return True

	def add_transition_constraint(self, agent, t, node, lastnode):
		if agent not in self.transition_constraints:
			self.transition_constraints[agent] = {}
		if t not in self.transition_constraints[agent]:
			self.transition_constraints[agent][t] = {}
		if node not in self.transition_constraints[agent][t]:
			self.transition_constraints[agent][t][node] = set()
		if lastnode in self.transition_constraints[agent][t][node]:
			return False
		self.transition_constraints[agent][t][node].add(lastnode)
		self.T = max(self.T, t)
		return True

	def get_constraint_fn(self, agent):
		def constraint_fn(node, lastnode, t):
			overlap = node in self.constraints.get(agent, {}).get(t, set())
			swap = lastnode in self.transition_constraints.get(agent, {}).get(t, {}).get(node, set())
			return (not overlap) and (not swap)
		return constraint_fn

	def find_paths(self, env, starts, goals):
		paths = [None] * len(starts)
		costs = [None] * len(starts) 
		# aggr_cost = [None]
		for agent in range(len(starts)):
			path, cost = astar(env=env, start=starts[agent], goal=goals[agent], constraint_fn=self.get_constraint_fn(agent), return_cost=True)
			paths[agent] = path
			costs[agent] = cost
			# aggr_cost[0] = cost

		self.T = max(self.T, max([len(path) for path in paths]))
		for agent in range(len(starts)):
			start_t = len(paths[agent]) - 1
			hold_path = stay(env=env, start=paths[agent][-1], goal=goals[agent], constraint_fn=self.get_constraint_fn(agent), start_t=start_t, T=self.T)
			paths[agent] = paths[agent] + hold_path[1:]
		# print("Final path", paths)
		self.solution = paths
		self.cost = costs
		# self.cost = env.globalcost(paths)
		# self.cost = aggr_cost


def find_conflict(paths, env):
	maxlength = max(map(lambda path: len(path) if path is not None else 0, paths))
	last_states = {}
	road_usage = dict() #keep track of road usage
	node_usage = dict()
	for t in range(maxlength):
		# Check collisions
		states = {}
		for agent in range(len(paths)):
			if paths[agent] is None:
				continue

			if t > 0 and t < (len(paths[agent])-1):
				prev_node = paths[agent][t-1]
				node = paths[agent][t]
			elif t == 0:
				node = paths[agent][t]
			else:
				prev_node = paths[agent][-1]
				node = paths[agent][-1]

			# Check Node Capacity:
			node_usage[node] = 1
			if node not in node_usage.keys():
				node_usage[node] = 1
			else:
				node_usage[node] += 1
			if node_usage[node] > env.getNodeCapacity(node) and node in states:
				other_agent = states[node][-1]
				return {'agent1': agent, 'agent2': other_agent, 't': t, 'node': node, 'transition': False}

			if t == 0:
				if node not in states.keys():
					states[node] = [agent]
				else:
					states[node].append(agent)
				continue

			# Check Edge Capacity:
			pair = frozenset((prev_node, node))
			if pair not in road_usage.keys():
				road_usage[pair] = 1
			else:
				road_usage[pair] += 1

			if road_usage[pair] > env.getEdgeCapacity(prev_node, node) and node in states:
				# print("state[node][-1]", states[node])
				other_agent = states[node][-1]
				return {'agent1': agent, 'agent2': other_agent, 't': t, 'node': node, 'transition': False}
			
			if node not in states.keys():
				states[node] = [agent]
			else:
				states[node].append(agent)

			# print('in_state', states)

		# print('state', states)
		# print('last state', last_states)
		# Edge conflicts: check swap
		for node, agents in states.items():
			if node in last_states: # if agent has moved into a spot that was just occupied, get the other agent who just moved out
				other_agents = last_states[node]
				for agent in agents:
					for other_agent in other_agents:
						# print("other", other_agent)
						# print("agent", agent)
						if other_agent != agent:
							last_node = paths[agent][t-1]
							# if the agent's last spot is now occupied by that other agent, it's a swap
							if paths[other_agent][t] == last_node and \
								frozenset((last_node, node)) in road_usage and \
								road_usage[frozenset((last_node, node))] > env.getEdgeCapacity(last_node, node):
								return {'agent1': agent, 'agent2': other_agent, 't': t, 'node': node, 'lastnode': last_node, 'transition': True}
		last_states = states
	return None

