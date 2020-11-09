import random
import plot


class Edge:
    def __init__(self, energy_limit):
        self.energy_limit = energy_limit
        self.energy = 0

    def resetEnergy(self):
        self.energy = 0

    def validateEnergy(self, potential):
        return self.energy + potential <= self.energy_limit


class Node:
    def __init__(self, demand, production, position):
        self.neighbours = {}
        self.default_demand = demand
        self.default_production = production
        self.position = position

        self.demand = demand
        self.production = production

    def addEdge(self, edge, neighbour):
        self.neighbours[neighbour] = edge

    def getEnergy(self):
        return self.production - self.demand

    def resetEnergy(self):
        self.demand = self.default_demand
        self.production = self.default_production


def euclideanDistanceSquared(node1: Node, node2: Node):
    return sum((x1-x2) ** 2 for x1, x2 in zip(node1.position, node2.position))


class Graph:
    def __init__(self, energy_heuristic, neighbour_radius, energy_kwargs=None):
        self.nodes = {}
        self.edges = []
        self.rank = 0
        self.energy_heuristic = energy_heuristic
        self.energy_kwargs = energy_kwargs
        self.neighbour_radius = neighbour_radius

    def _compute_energy(self, node1, node2):
        try:
            return self.energy_heuristic(node1, node2, **self.energy_kwargs)
        except TypeError:
            return self.energy_heuristic(node1, node2)

    def addNode(self, node: Node):
        self.nodes[node] = self.rank
        self.edges.append([[] for _ in range(self.rank)])
        self.rank += 1
        for i in range(self.rank):
            self.edges[i].append([])

        for potential_neighbour in list(self.nodes.keys())[:-1]:
            distance = euclideanDistanceSquared(node, potential_neighbour)
            if distance <= self.neighbour_radius ** 2:
                self._addEdge(self._compute_energy(node, potential_neighbour), node, potential_neighbour)

    def _addEdge(self, energy_limit, node1: Node, node2: Node):
        edge = Edge(energy_limit)
        self.edges[self.nodes[node1]][self.nodes[node2]] = edge
        self.edges[self.nodes[node2]][self.nodes[node1]] = edge
        node1.addEdge(edge, node2)
        node2.addEdge(edge, node1)

    def getEdge(self, node1: Node, node2: Node):
        try:
            return self.edges[self.nodes[node1]][self.nodes[node2]]
        except IndexError:
            return False

    def getIndex(self, node):
        return self.nodes[node]


class AntColony:
    def __init__(self, neighbour_radius, energy_heuristic, ant_count, cycles, cost_heuristic, cost_kwargs=None,
                 energy_kwargs=None, alpha=1, beta=3, rho=0.5, Q=1):

        # Parameters:
        #  Alpha: Importance of Pheromones
        #  Beta: Importance of Heuristics
        #  Rho: Pheromone Evaporation Coefficient
        #  Q: Pheromone Update Coefficient
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q

        self.limit = 0.25

        self.graph = Graph(energy_heuristic=energy_heuristic, neighbour_radius=neighbour_radius,
                           energy_kwargs=energy_kwargs)
        self.producers = []
        self.consumers = []
        self.pheromones = []
        self.has_producers = True

        self._heuristic_method = cost_heuristic
        self._heuristic_kwargs = cost_kwargs
        self.heuristics = self._computeHeuristics()

        self.ants = []
        self.ant_count = ant_count
        self.cycles = cycles

    def addAnt(self):
        ant = Ant(self)
        ant.reset()
        self.ants.append(ant)
        self.ant_count += 1

    def addNode(self, node: Node):
        self.graph.addNode(node)
        self.pheromones.append(0)
        self.heuristics = self._computeHeuristics()
        if node.default_production > 0:
            self.producers.append(node)
        if node.default_demand > 0:
            self.consumers.append(node)

    def _computeHeuristics(self):
        try:
            return self._heuristic_method(self.graph, **self._heuristic_kwargs)
        except TypeError:
            return self._heuristic_method(self.graph)

    def _updatePheromone(self):
        for i in range(len(self.pheromones)):
            for j in range(len(self.pheromones)):
                self.pheromones[i][j] *= self.rho
                for ant in self.ants:
                    self.pheromones[i] += ant.deltaPheromones[i]

    def solve(self):
        self.ants = [Ant(self) for _ in range(self.ant_count)]
        self.pheromones = [[0 for _ in self.graph.nodes] for _ in self.graph.nodes]
        for ant in self.ants:
            ant.reset()
        cost = float('inf')
        solution = []
        for cycle in range(self.cycles):
            for ant in self.ants:
                for _ in self.graph.nodes: #[:-1]:
                    ant.move()
                if 0 != ant.total_cost < cost:
                    solution = ant.tabulist
                    cost = ant.total_cost
                ant.updatePheromoneDelta()
                self._updatePheromone()
                ant.reset()
            for node in self.graph.nodes:
                node.resetEnergy()
            for es in self.graph.edges:
                for edge in es:
                    if edge:
                        edge.resetEnergy()
            self.has_producers = True
        return cost, solution


class Ant:
    def __init__(self, colony: AntColony):
        self.colony = colony
        self.deltaPheromones = [[0 for _ in self.colony.graph.nodes] for _ in self.colony.graph.nodes]
        self.node = None
        self.previous = 0
        self.energy = 0
        self.tabulist = []
        self.total_cost = 0
        self.allowed = []

    def reset(self):
        self.node = random.choice(self.colony.producers)
        self.previous = self.node
        self.energy = 0
        self.tabulist = [self.colony.graph.getIndex(self.node)]
        self.total_cost = 0
        self._computeEnergy()

    def updatePheromoneDelta(self):
        for i in self.tabulist:
            for j in self.tabulist:
                if i != j:
                    try:
                        self.deltaPheromones[i][j] = self.colony.Q / self.total_cost
                    except ZeroDivisionError:
                        self.deltaPheromones[i][j] = 0

    def _getAllowed(self):
        allowed = self.node.neighbours
        return [x for x in allowed if (self.colony.graph.getIndex(x) not in self.tabulist
                                       and self.colony.graph.getEdge(self.node, x).validateEnergy(self.energy))]

    def _computeEnergy(self):
        charge = min(self.node.demand, self.energy)
        self.energy -= charge
        self.node.demand -= charge

        self.energy += self.node.production
        self.node.production = 0

    def _movetoClosestProducer(self):
        choice = [x for x in self.colony.producers if x.production > 0]
        if not choice:
            self.colony.has_producers = False
            return
        choice.sort(key=lambda x: euclideanDistanceSquared(self.node, x))
        choice = choice[0]
        choice_index = self.colony.graph.getIndex(choice)
        #self.tabulist.append(choice_index)
        self.previous = self.node
        self.node = choice

    def _stochasticMove(self):
        # Probabilities of moving into a specific cell k
        probabilities = []
        choice = 0
        choice_index = 0
        start = self.tabulist[-1]
        self.allowed = self._getAllowed()
        if not self.allowed:
            return
        for a in self.allowed:
            edge = self.colony.graph.getIndex(a)
            try:
                p = (self.colony.pheromones[start][edge] ** self.colony.alpha) * (
                        self.colony.heuristics[start][edge] ** self.colony.beta)
                h = sum([(self.colony.pheromones[start][self.colony.graph.getIndex(x)] ** self.colony.alpha) *
                         (self.colony.heuristics[start][self.colony.graph.getIndex(x)] ** self.colony.beta) for x in self.allowed])
                probabilities.append(p / h)
            except ZeroDivisionError:
                probabilities.append(0)

        rand = random.random()
        for index, probability in enumerate(probabilities):
            rand -= probability
            if rand <= 0:
                choice = self.allowed[index]
                break
        else:
            choice = random.choice(self.allowed)
        choice_index = self.colony.graph.getIndex(choice)
        self.tabulist.append(choice_index)
        self.total_cost += self.colony.heuristics[start][choice_index]
        self.colony.graph.getEdge(self.node, choice).energy += self.energy
        self.previous = self.node
        self.node = choice

    def move(self):
        if self.energy == 0:
            if self.colony.has_producers:
                self._movetoClosestProducer()
            else:
                return
        else:
            self._stochasticMove()
        self._computeEnergy()

