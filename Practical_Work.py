import numpy as np
import time 

def north_east_method(cost_matrix, supply, demand, basis_matrix):
    
    num_supply = len(supply)
    num_demand = len(demand)

    m, n = cost_matrix.shape
    transport_plan = np.zeros((m, n))

    supply_copy = supply.copy()
    demand_copy = demand.copy()
    
    i, j = 0, 0
    while i < num_supply and j < num_demand:
        allocation = min(supply_copy[i], demand_copy[j])
        basis_matrix [i,j] = True
        transport_plan[i, j] = allocation
        supply_copy[i] -= allocation
        demand_copy[j] -= allocation
        
        if supply_copy[i] == 0:
            i += 1
        elif demand_copy[j] == 0:
            j += 1

    return transport_plan

def min_element_method(cost_matrix, supply, demand, basis_matrix):

    m, n = cost_matrix.shape
    copy_cost_matrix = cost_matrix.astype(float)

    transport_plan = np.zeros((m, n))
    supply_copy = supply.copy()
    demand_copy = demand.copy()
    
    count_matrix = np.zeros((m, n))

    while np.count_nonzero(count_matrix) != m + n - 1:
        i, j = np.unravel_index(np.argmin(copy_cost_matrix), copy_cost_matrix.shape)

        x = min(supply_copy[i], demand_copy[j])
        transport_plan[i, j] = x
        count_matrix[i, j] = 1
        supply_copy[i] -= x
        demand_copy[j] -= x


        if supply_copy[i] == 0 and np.sum(copy_cost_matrix[i, :] == np.inf) > np.sum(copy_cost_matrix[:, j] == np.inf):
            copy_cost_matrix[i, :] = np.inf
        elif demand_copy[j] == 0:
            copy_cost_matrix[:, j] = np.inf
        elif supply_copy[i] == 0:
            copy_cost_matrix[i, :] = np.inf

    basis_matrix[:, :] = count_matrix != 0
    return transport_plan

def vogels_approximation_method(cost_matrix, supply, demand, basis_matrix):
    m, n = cost_matrix.shape
    transport_plan = np.zeros((m, n))
    supply_copy = supply.copy()
    demand_copy = demand.copy()
    copy_cost_matrix = cost_matrix.astype(float)
    count_matrix = np.zeros((m, n))

    excluded_rows = set()
    excluded_cols = set()

    while np.count_nonzero(count_matrix) != m + n - 1:
        penalties = []

        for i in range(m):
            if i not in excluded_rows:
                row = sorted(cost_matrix[i, :])
                penalties.append((row[1] - row[0], i, 'row'))
        for j in range(n):
            if j not in excluded_cols:
                col = sorted(cost_matrix[:, j])
                penalties.append((col[1] - col[0], j, 'col'))

        penalties.sort(reverse=True, key=lambda x: x[0])

        _, index, type_ = penalties[0]

        if type_ == 'row':
            i = index
            j = np.argmin(copy_cost_matrix[i, :])
        else:
            j = index
            i = np.argmin(copy_cost_matrix[:, j])

        x = min(supply_copy[i], demand_copy[j])
        transport_plan[i, j] = x
        count_matrix[i, j] = 1
        supply_copy[i] -= x
        demand_copy[j] -= x

        if supply_copy[i] == 0 and np.sum(copy_cost_matrix[i, :] == np.inf) > np.sum(copy_cost_matrix[:, j] == np.inf):
            copy_cost_matrix[i, :] = np.inf
            excluded_rows.add(i)
        elif demand_copy[j] == 0:
            copy_cost_matrix[:, j] = np.inf
            excluded_cols.add(j)
        elif supply_copy[i] == 0:
            copy_cost_matrix[i, :] = np.inf
            excluded_rows.add(i)

    basis_matrix[:, :] = count_matrix != 0
    return transport_plan

def double_preference_method(cost_matrix, supply, demand, basis_matrix):
    m, n = cost_matrix.shape
    transport_plan = np.zeros((m, n))
    supply_copy = supply.copy()
    demand_copy = demand.copy()
    count_matrix = np.zeros((m, n))
    copy_cost_matrix = cost_matrix.astype(float)

    while np.count_nonzero(count_matrix) != m + n - 1:
        row_mins = np.min(copy_cost_matrix, axis=1)
        col_mins = np.min(copy_cost_matrix, axis=0)

        min_row_value = np.min(row_mins)
        min_col_value = np.min(col_mins)

        if min_row_value <= min_col_value:
            i = np.argmin(row_mins)
            j = np.argmin(copy_cost_matrix[i, :])
        else:
            j = np.argmin(col_mins)
            i = np.argmin(copy_cost_matrix[:, j])

        x = min(supply_copy[i], demand_copy[j])
        transport_plan[i, j] = x
        count_matrix[i, j] = 1
        supply_copy[i] -= x
        demand_copy[j] -= x

        if supply_copy[i] == 0 and np.sum(copy_cost_matrix[i, :] == np.inf) > np.sum(copy_cost_matrix[:, j] == np.inf):
            copy_cost_matrix[i, :] = np.inf
        elif demand_copy[j] == 0:
            copy_cost_matrix[:, j] = np.inf
        elif supply_copy[i] == 0:
            copy_cost_matrix[i, :] = np.inf
    basis_matrix[:, :] = count_matrix != 0
    return transport_plan

def find_cycle_path(x: np.ndarray, start_pos, basis_matrix) :
    def get_possible_moves(bool_table: np.ndarray, path):
        possible_moves = np.full(bool_table.shape, False)
        current_pos = path[-1]
        prev_pos = path[-2] if len(path) > 1 else (np.nan, np.nan)

        if current_pos[0] != prev_pos[0]:
            possible_moves[current_pos[0]] = True
        if current_pos[1] != prev_pos[1]:
            possible_moves[:, current_pos[1]] = True

        return list(zip(*np.nonzero(possible_moves * bool_table)))

    res = [start_pos]
    bool_table = basis_matrix.copy()

    while True:
        current_pos = res[-1]

        bool_table[current_pos[0]][current_pos[1]] = False

        if len(res) > 3:
            bool_table[start_pos[0]][start_pos[1]] = True

        possible_moves = get_possible_moves(bool_table, res)

        if start_pos in possible_moves:
            res.append(start_pos)
            return res

        if not possible_moves:
            for i, j in res[1:-1]:
                bool_table[i][j] = True
            res = [start_pos]
            continue

        res.append(possible_moves[0])

def recalculate_plan(x: np.ndarray, cycle_path, basis_matrix, excluded_cells) -> int:
    o = np.min([x[i][j] for i, j in cycle_path[1:-1:2]])
    flag = True
    for k, (i, j) in enumerate(cycle_path[:-1]):
        if k % 2 == 0:
            x[i][j] += o
        else:
            x[i][j] -= o

        if x[i][j] == 0 and flag == True and basis_matrix[i, j] == True and (i, j) not in excluded_cells:
            flag = False
            basis_matrix[i, j] = False

def generate_transport_problem(n, min_cost=1, max_cost=100, total_supply=100):
    
    cost_matrix = np.random.randint(min_cost, max_cost + 1, size=(n, n))

    supply = np.random.randint(1, total_supply, size=n)
    supply = (supply / supply.sum() * total_supply).astype(int)
    supply[-1] += total_supply - supply.sum()

    demand = np.random.randint(1, total_supply, size=n)
    demand = (demand / demand.sum() * total_supply).astype(int)
    demand[-1] += total_supply - demand.sum()
    return cost_matrix, supply, demand

def potentials_method(cost_matrix, supply, demand , method):
    counter = 0
    m, n = cost_matrix.shape
    basis_matrix = np.zeros((m, n))
    basis_matrix = basis_matrix != 0

    if method == 1:
        transport_plan = north_east_method(cost_matrix, supply, demand, basis_matrix)
    elif method == 2:
        transport_plan = min_element_method(cost_matrix, supply, demand, basis_matrix)
    elif method == 3:
        transport_plan = vogels_approximation_method(cost_matrix, supply, demand, basis_matrix)
    else:
        transport_plan = double_preference_method(cost_matrix, supply, demand, basis_matrix)
    
    num_supply = len(supply)
    num_demand = len(demand)

    u = [None] * num_supply
    v = [None] * num_demand
    u[0] = 0 

    while None in u or None in v:
        for i in range(num_supply):
            for j in range(num_demand):
                if basis_matrix[i,j] == True:
                    if u[i] is not None and v[j] is None:
                        v[j] = cost_matrix[i, j] - u[i]
                    elif u[i] is None and v[j] is not None:
                        u[i] = cost_matrix[i, j] - v[j]
                        
    excluded_cells = set()
    while True:
        delta = np.full((num_supply, num_demand), float('inf'))
        flag = False
        for i in range(num_supply):
            for j in range(num_demand):
                if basis_matrix[i, j] == False:
                    delta[i, j] = cost_matrix[i, j] - (u[i] + v[j])
                    if delta[i, j] < 0:
                        flag = True
        if flag == False:
            break

        i_new, j_new = np.unravel_index(np.argmin(delta), delta.shape)
        
        cycle = find_cycle_path(transport_plan, (i_new, j_new), basis_matrix)
        temp_plan = transport_plan
        recalculate_plan(transport_plan, cycle, basis_matrix, excluded_cells)
        
        if (np.sum(temp_plan * cost_matrix) >= np.sum(transport_plan * cost_matrix)):
            excluded_cells.add((i_new, j_new))
        else:
            excluded_cells.clear()


        basis_matrix[i_new, j_new] = True
        
        counter += 1
        u = [None] * num_supply
        v = [None] * num_demand
        u[0] = 0
        while None in u or None in v:
            for i in range(num_supply):
                for j in range(num_demand):
                    if basis_matrix[i, j] == True:
                        if u[i] is not None and v[j] is None:
                            v[j] = cost_matrix[i, j] - u[i]
                        elif u[i] is None and v[j] is not None:
                            u[i] = cost_matrix[i, j] - v[j]

    
    total_cost = np.sum(transport_plan * cost_matrix)
    return transport_plan, total_cost

n = 3
min_cost = 1
max_cost = 100
total_supply = 1000

time_spend_methods = [0, 0, 0, 0]
cost_matrix, supply, demand = generate_transport_problem(n, min_cost, max_cost, total_supply)
previous_cost = 0
for i in range(20000):
    
    cost_matrix, supply, demand = generate_transport_problem(n, min_cost, max_cost, total_supply)

    print("Матрица стоимости (cost_matrix):")   
    print("\n".join(["\t".join(map(str, row)) for row in cost_matrix]))
    print("\nВектор предложения (supply):", supply)
    print("\nВектор спроса (demand):", demand)

    with open("output.txt", mode='a') as file:
        file.write("\nМатрица стоимости (cost_matrix):\n")
        file.write("\n".join(["\t".join(map(str, row)) for row in cost_matrix]) + "\n")
        file.write("Вектор предложения (supply): " + "\t".join(map(str, supply)) + "\n")
        file.write("Вектор спроса (demand): " + "\t".join(map(str, demand)) + "\n")
        
    flag = True
    for i in range(1, 5):
        start_time = time.time()    
        plan, cost = potentials_method(cost_matrix, supply, demand, i)
        end_time = time.time()
        time_spend_methods[i-1] += end_time - start_time
        print("Оптимальный план перевозок:")
        print(plan)
        print(f"Минимальная стоимость: {cost}")
        print(f"Базисные клетки {np.count_nonzero(plan)}" )
        previous_cost = cost
        flag = False

print(*time_spend_methods)