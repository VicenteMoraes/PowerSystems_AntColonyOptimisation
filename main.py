import ACO
from heuristics import cost, energy
import random
import plot

colony = ACO.AntColony(neighbour_radius=10000, energy_heuristic=energy.euclideanEnergy, energy_kwargs={"weight": float('inf')},
                       ant_count=10, cycles=10, cost_heuristic=cost.euclideanCost)
#Producers
for i in range(20):
    n = ACO.Node(demand=0, production=10, position=(random.randint(0, 50), random.randint(0, 50)))
    colony.addNode(n)

#Consumers
for i in range(5):
    d = ACO.Node(demand=2, production=0, position=(random.randint(0, 50), random.randint(0, 50)))
    colony.addNode(d)

cost, solution = colony.solve()
print(cost)
plot.plot_solution(solution, colony.graph)