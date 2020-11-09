import matplotlib.pyplot as plt


def get_xy(solution):
    return [[node.position[0] for node in solution], [node.position[1] for node in solution]]


def plot_solution(solution, graph):
    nodes = graph.nodes
    keys = list(nodes.keys())
    values = list(nodes.values())
    solution = [keys[values.index(x)] for x in solution]
    producers = [x for x in nodes if x.default_production > 0]
    consumers = [x for x in nodes if x.default_demand > 0]
    x, y = get_xy(solution)
    x_prod, y_prod = get_xy(producers)
    x_con, y_con = get_xy(consumers)
    plt.plot(x, y, zorder=1, label="links", color="black")
    plt.scatter(x_prod, y_prod, color="r", marker="o", zorder=2, label="Producers")
    plt.scatter(x_con, y_con, color="g", marker="o", zorder=2, label="Consumers")
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()