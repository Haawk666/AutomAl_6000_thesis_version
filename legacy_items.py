from copy import deepcopy
import numpy as np

'''

def precipitate_controller(self, i):

    self.boarder_size = 0
    self.precipitate_boarder = np.ndarray([1], dtype=int)

    self.reset_all_flags()

    self.precipitate_finder(i)

    counter = 0

    for x in range(0, self.num_columns):

        if self.columns[x].flag_1 or self.columns[x].h_index == 6:
            self.columns[x].is_in_precipitate = False
        else:
            self.columns[x].is_in_precipitate = True

        if self.columns[x].flag_2:

            if counter == 0:
                self.precipitate_boarder[0] = x
            else:
                self.precipitate_boarder = np.append(self.precipitate_boarder, x)

            counter = counter + 1

    self.boarder_size = counter
    self.reset_all_flags()
    self.sort_boarder()

def sort_boarder(self):

    temp_boarder = deepcopy(self.precipitate_boarder)
    selected = np.ndarray([self.boarder_size], dtype=bool)
    for y in range(0, self.boarder_size):
        selected[y] = False
    next_index = 0
    index = 0
    cont_var = True
    selected[0] = True

    while cont_var:

        distance = self.N * self.M

        for x in range(0, self.boarder_size):

            current_distance = np.sqrt((self.columns[self.precipitate_boarder[x]].x -
                                        self.columns[temp_boarder[index]].x )* *2 +
                                       (self.columns[self.precipitate_boarder[x]].y -
                                        self.columns[temp_boarder[index]].y )* *2)

            if current_distance < distance and not temp_boarder[index] == self.precipitate_boarder[x] and not selected
                [x]:
                distance = current_distance
                next_index = x

        selected[next_index] = True
        index = index + 1

        temp_boarder[index] = self.precipitate_boarder[next_index]

        if index == self.boarder_size - 1:
            cont_var = False

    self.precipitate_boarder = deepcopy(temp_boarder)

def precipitate_finder(self, i):

    indices, distances, n = self.find_nearest(i, True)

    self.columns[i].flag_1 = True

    for x in range(0, n):

        if not self.columns[indices[x]].h_index == 3:

            if not self.columns[indices[x]].h_index == 6:
                self.columns[i].flag_2 = True

        else:

            if not self.columns[indices[x]].flag_1:
                self.precipitate_finder(indices[x])
                
def define_levels(self, i, level=0):

    self.reset_all_flags()

    self.mesh_levels(i, level)

    complete = False
    emer_abort = False
    overcounter = 0
    neighbour_level = 0

    while not complete and not emer_abort:

        found = False
        counter = 0

        while counter < self.num_columns:

            if self.columns[counter].is_in_precipitate and not self.columns[counter].flag_1:

                x = 0

                while x <= neighbour_level:

                    if self.columns[self.columns[counter].neighbour_indices[x]].is_in_precipitate and\
                                    self.columns[self.columns[counter].neighbour_indices[x]].flag_1:

                        neighbour = self.columns[counter].neighbour_indices[x]
                        if self.columns[neighbour].level == 0:
                            self.columns[counter].level = 1
                        else:
                            self.columns[counter].level = 0
                        self.columns[counter].flag_1 = True
                        found = True

                    x = x + 1

            counter = counter + 1

        complete = True

        for y in range(0, self.num_columns):

            if self.columns[y].is_in_precipitate and not self.columns[y].flag_1:

                complete = False

        if found and neighbour_level > 0:

            neighbour_level = neighbour_level - 1

        if not found and neighbour_level < 2:

            neighbour_level = neighbour_level + 1

        overcounter = overcounter + 1
        if overcounter > 100:

            emer_abort = True
            print('Emergency abort')

        print(neighbour_level)

    self.reset_all_flags()

def mesh_levels(self, i, level):

    if self.columns[i].is_in_precipitate:

        self.columns[i].flag_1 = True
        self.set_level(i, level)

    else:

        self.columns[i].flag_1 = True

        next_level = 0
        if level == 0:
            next_level = 1
        elif level == 1:
            next_level = 0
        else:
            print('Disaster!')

        self.set_level(i, level)

        indices = self.columns[i].neighbour_indices

        for x in range(0, self.columns[i].n()):

            reciprocal = self.test_reciprocality(i, indices[x])

            if not self.columns[indices[x]].flag_1 and not self.columns[i].is_edge_column and reciprocal:

                self.mesh_levels(indices[x], next_level)

def precipitate_levels(self, i, level):

    if not self.columns[i].is_in_precipitate:

        self.columns[i].flag_1 = True

    else:

        self.columns[i].flag_1 = True

        next_level = 0
        if level == 0:
            next_level = 1
        elif level == 1:
            next_level = 0
        else:
            print('Disaster!')

        self.set_level(i, level)

        indices = self.columns[i].neighbour_indices

        complete = False
        counter_1 = 0
        counter_2 = 0

        while not complete:

            if not self.columns[indices[counter_1]].flag_1:

                if self.test_reciprocality(i, indices[counter_1]):

                    self.precipitate_levels(indices[counter_1], next_level)
                    counter_1 = counter_1 + 1
                    counter_2 = counter_2 + 1

                else:

                    counter_1 = counter_1 + 1

            else:

                counter_1 = counter_1 + 1

            if counter_2 == self.columns[i].n() - 2 or counter_1 == self.columns[i].n() - 2:

                complete = True

def set_level(self, i, level):

    previous_level = self.columns[i].level
    self.columns[i].level = level

    if level == previous_level:
        return False
    else:
        return True

def classify_pair(self, i, j):

    neighbour_type = 0
    partner_type = 0
    intersects = False

    i_neighbour_to_j = False
    j_neighbour_to_i = False
    i_partner_to_j = False
    j_partner_to_i = False

    for x in range(0, 8):

        if self.columns[i].neighbour_indices[x] == j:
            j_neighbour_to_i = True
            if x < self.columns[i].n():
                j_partner_to_i = True

        if self.columns[j].neighbour_indices[x] == i:
            i_neighbour_to_j = True
            if x < self.columns[j].n():
                i_partner_to_j = True

    if not i_neighbour_to_j and not j_neighbour_to_i:
        neighbour_type = 0
    elif not i_neighbour_to_j and j_neighbour_to_i:
        neighbour_type = 1
    elif i_neighbour_to_j and not j_neighbour_to_i:
        neighbour_type = 2
    elif i_neighbour_to_j and j_neighbour_to_i:
        neighbour_type = 3

    if not i_partner_to_j and not j_partner_to_i:
        partner_type = 0
    elif not i_partner_to_j and j_partner_to_i:
        partner_type = 1
    elif i_partner_to_j and not j_partner_to_i:
        partner_type = 2
    elif i_partner_to_j and j_partner_to_i:
        partner_type = 3

    if self.columns[i].level == self.columns[j].level:
        level_type = 0
    else:
        level_type = 1

    if partner_type == 0:

        geometry_type_clockwise = -1
        geometry_type_anticlockwise = -1
        geo_type_symmetry = -1

    else:

        indices, num_edges_clockwise_right = self.find_shape(i, j, clockwise=True)
        indices, num_edges_clockwise_left = self.find_shape(j, i, clockwise=True)
        indices, num_edges_anticlockwise_right = self.find_shape(j, i, clockwise=False)
        indices, num_edges_anticlockwise_left = self.find_shape(i, j, clockwise=False)

        if num_edges_clockwise_right == 3 and num_edges_clockwise_left == 3:
            geometry_type_clockwise = 1
        elif num_edges_clockwise_right == 5 and num_edges_clockwise_left == 3:
            geometry_type_clockwise = 2
        elif num_edges_clockwise_right == 3 and num_edges_clockwise_left == 5:
            geometry_type_clockwise = 3
        elif num_edges_clockwise_right == 4 and num_edges_clockwise_left == 3:
            geometry_type_clockwise = 4
        elif num_edges_clockwise_right == 3 and num_edges_clockwise_left == 4:
            geometry_type_clockwise = 5
        elif num_edges_clockwise_right == 4 and num_edges_clockwise_left == 4:
            geometry_type_clockwise = 6
        elif num_edges_clockwise_right == 5 and num_edges_clockwise_left == 5:
            geometry_type_clockwise = 7
        else:
            geometry_type_clockwise = 0

        if num_edges_anticlockwise_right == 3 and num_edges_anticlockwise_left == 3:
            geometry_type_anticlockwise = 1
        elif num_edges_anticlockwise_right == 5 and num_edges_anticlockwise_left == 3:
            geometry_type_anticlockwise = 2
        elif num_edges_anticlockwise_right == 3 and num_edges_anticlockwise_left == 5:
            geometry_type_anticlockwise = 3
        elif num_edges_anticlockwise_right == 4 and num_edges_anticlockwise_left == 3:
            geometry_type_anticlockwise = 4
        elif num_edges_anticlockwise_right == 3 and num_edges_anticlockwise_left == 4:
            geometry_type_anticlockwise = 5
        elif num_edges_anticlockwise_right == 4 and num_edges_anticlockwise_left == 4:
            geometry_type_anticlockwise = 6
        elif num_edges_anticlockwise_right == 5 and num_edges_anticlockwise_left == 5:
            geometry_type_anticlockwise = 7
        else:
            geometry_type_anticlockwise = 0

        if geometry_type_clockwise == geometry_type_anticlockwise:
            geo_type_symmetry = 0
        else:
            geo_type_symmetry = 1

    # Implement method to find intersections

    return neighbour_type, partner_type, level_type, geometry_type_clockwise, geometry_type_anticlockwise,\
        geo_type_symmetry, intersects

def resolve_edge_inconsistency(self, i, j, clockwise=True):

    neighbour_type, partner_type, level_type, geometry_type_clockwise, geometry_type_anticlockwise, \
        geo_type_symmetry, intersects = self.classify_pair(i, j)

    if geo_type_symmetry == 0:
        geometry_type = geometry_type_clockwise
    else:
        geometry_type = 0

    if neighbour_type == 0:
        i_neighbour_to_j = False
        j_neighbour_to_i = False
    elif neighbour_type == 1:
        i_neighbour_to_j = False
        j_neighbour_to_i = True
    elif neighbour_type == 2:
        i_neighbour_to_j = True
        j_neighbour_to_i = False
    elif neighbour_type == 3:
        i_neighbour_to_j = True
        j_neighbour_to_i = True
    else:
        i_neighbour_to_j = False
        j_neighbour_to_i = False

    if partner_type == 0:
        i_partner_to_j = False
        j_partner_to_i = False
    elif partner_type == 1:
        i_partner_to_j = False
        j_partner_to_i = True
    elif partner_type == 2:
        i_partner_to_j = True
        j_partner_to_i = False
    elif partner_type == 3:
        i_partner_to_j = True
        j_partner_to_i = True
    else:
        i_partner_to_j = False
        j_partner_to_i = False

    i_index_in_j = -1
    j_index_in_i = -1

    for x in range(0, 8):
        if self.columns[i].neighbour_indices[x] == j:
            j_index_in_i = x

    if j_index_in_i == -1:
        self.columns[i].neighbour_indices[7] = j
        j_index_in_i = 7

    for x in range(0, 8):
        if self.columns[j].neighbour_indices[x] == i:
            i_index_in_j = x

    if i_index_in_j == -1:
        self.columns[j].neighbour_indices[7] = i
        i_index_in_j = 7

    if i_partner_to_j:
        # Perturb neighbours of j such that i is last element in k^j
        self.perturbator(j, i_index_in_j, self.columns[j].n() - 1)
    else:
        # Perturb neighbours of j such that i is last element in k^j
        self.perturbator(j, i_index_in_j, self.columns[j].n())

    if j_partner_to_i:
        # Perturb neighbours of i such that j is last k
        self.perturbator(i, j_index_in_i, self.columns[i].n() - 1)
    else:
        # Perturb neighbours of i such that j is last k
        self.perturbator(i, j_index_in_i, self.columns[i].n())

    if clockwise:

        shape_1_indices, num_edge_1 = self.find_shape(i, j, clockwise=clockwise)
        shape_2_indices, num_edge_2 = self.find_shape(j, i, clockwise=clockwise)

    else:

        shape_1_indices, num_edge_1 = self.find_shape(j, i, clockwise=clockwise)
        shape_2_indices, num_edge_2 = self.find_shape(i, j, clockwise=clockwise)

    print(str(i) + ', ' + str(j) + ': ')

    if geometry_type == 1:
        # This means we want to break the connection!

        if partner_type == 1:
            if not self.try_connect(i, j_index_in_i):
                if not self.decrease_h_value(i):
                    print('Could not reconnect!')

        elif partner_type == 2:
            if not self.try_connect(j, i_index_in_j):
                if not self.decrease_h_value(j):
                    print('Could not reconnect!')

        elif partner_type == 3:
            if not self.try_connect(j, i_index_in_j):
                if not self.decrease_h_value(j):
                    print('Could not reconnect!')
            if not self.try_connect(i, j_index_in_i):
                if not self.decrease_h_value(i):
                    print('Could not reconnect!')

    if geometry_type == 2 or geometry_type == 3:
        # This means we want to switch connections to make geometry type 6

        loser_index_in_stayer = -1
        loser_connected_to_stayer = False
        stayer_connected_to_loser = False
        new_index_in_stayer = -1

        if geometry_type == 2:

            ind_1 = shape_1_indices[2]
            ind_2 = shape_1_indices[4]

        else:

            ind_1 = shape_2_indices[4]
            ind_2 = shape_2_indices[2]

        distance_1 = np.sqrt((self.columns[ind_1].x - self.columns[i].x) ** 2 + (
                self.columns[ind_1].y - self.columns[i].y) ** 2)
        distance_2 = np.sqrt((self.columns[j].x - self.columns[ind_2].x) ** 2 + (
                self.columns[j].y - self.columns[ind_2].y) ** 2)

        if distance_1 < distance_2:
            index_stayer = i
            index_loser = j
            stayer_index_in_loser = i_index_in_j
            index_new = ind_1
            if i_partner_to_j:
                loser_connected_to_stayer = True
            if j_partner_to_i:
                stayer_connected_to_loser = True
        else:
            index_stayer = j
            index_loser = i
            stayer_index_in_loser = j_index_in_i
            index_new = ind_2
            if j_partner_to_i:
                loser_connected_to_stayer = True
            if i_partner_to_j:
                stayer_connected_to_loser = True

        for x in range(self.columns[index_stayer].n(), 8):
            if self.columns[index_stayer].neighbour_indices[x] == index_new:
                new_index_in_stayer = x

        if new_index_in_stayer == -1:
            self.columns[index_stayer].neighbour_indices[7] = index_new
            new_index_in_stayer = 7

        self.perturbator(index_stayer, self.columns[index_stayer].n(), new_index_in_stayer)

        if loser_connected_to_stayer:
            if not self.try_connect(index_loser, stayer_index_in_loser):
                if not self.decrease_h_value(index_loser):
                    print('Could not reconnect!')

        if stayer_connected_to_loser:
            self.perturbator(index_stayer, loser_index_in_stayer, new_index_in_stayer)
        else:
            if not self.increase_h_value(index_stayer):
                print('Could not reconnect!')

    if geometry_type == 4 or geometry_type == 5:

        pass

    if geometry_type == 6:
        # This means we want to keep the connection

        if partner_type == 1:
            if not self.increase_h_value(j):
                print('Could not reconnect!')

        elif partner_type == 2:
            if not self.increase_h_value(i):
                print('Could not reconnect!')

    if geometry_type == 0:

        print(str(num_edge_1) + ', ' + str(num_edge_2))
        print(shape_1_indices)
        print(shape_2_indices)

def try_connect(self, i, j_index_in_i):

    changed = False
    better_friend = False
    friend_index_in_i = -1

    for x in range(self.columns[i].n(), 8):

        if self.test_reciprocality(i, self.columns[i].neighbour_indices[x]):

            if not self.columns[i].level == self.columns[self.columns[i].neighbour_indices[x]].level:

                better_friend = True
                friend_index_in_i = x

            else:

                print('Maybe should have?')

    if better_friend:

        self.perturbator(i, j_index_in_i, friend_index_in_i)
        changed = True

    return changed

def perturbator(self, i, a, b):

    val_a = self.columns[i].neighbour_indices[a]
    val_b = self.columns[i].neighbour_indices[b]

    self.columns[i].neighbour_indices[a] = val_b
    self.columns[i].neighbour_indices[b] = val_a

def find_edge_columns(self):

    for y in range(0, self.num_columns):

        x_coor = self.columns[y].x
        y_coor = self.columns[y].y
        margin = 6 * self.r

        if x_coor < margin or x_coor > self.M - margin - 1 or y_coor < margin or y_coor > self.N - margin - 1:

            self.columns[y].is_edge_column = True
            self.reset_prop_vector(y, bias=3)

        else:

            self.columns[y].is_edge_column = False

def find_consistent_perturbations_simple(self, y, sub=False):

    if not self.columns[y].is_edge_column:

        n = self.columns[y].n()

        if sub:

            n = 3

        indices = self.columns[y].neighbour_indices
        new_indices = np.zeros([indices.shape[0]], dtype=int)
        found = 0

        for x in range(0, indices.shape[0]):

            n2 = 3

            if self.columns[indices[x]].h_index == 0 or self.columns[indices[x]].h_index == 1:
                n2 = 3
            elif self.columns[indices[x]].h_index == 3:
                n2 = 4
            elif self.columns[indices[x]].h_index == 5:
                n2 = 5
            else:
                print('Problem in find_consistent_perturbations_simple!')

            neighbour_indices = self.columns[indices[x]].neighbour_indices

            for z in range(0, n2):

                if neighbour_indices[z] == y:
                    new_indices[found] = indices[x]
                    found = found + 1

        if found == n:

            index_positions = np.zeros([found], dtype=int)

            for k in range(0, found):

                for z in range(0, indices.shape[0]):

                    if indices[z] == new_indices[k]:
                        index_positions[k] = z

            counter = found - 1

            for i in range(0, indices.shape[0]):

                are_used = False

                for z in range(0, found):

                    if i == index_positions[z]:

                        are_used = True

                if not are_used:

                    counter = counter + 1
                    new_indices[counter] = indices[i]

            self.columns[y].neighbour_indices = new_indices
            self.columns[y].is_popular = False
            self.columns[y].is_unpopular = False

        elif found > n:

            index_positions = np.zeros([found], dtype=int)

            for k in range(0, found):

                for z in range(0, indices.shape[0]):

                    if indices[z] == new_indices[k]:
                        index_positions[k] = z

            counter = found - 1

            for i in range(0, indices.shape[0]):

                are_used = False

                for z in range(0, found):

                    if i == index_positions[z]:
                        are_used = True

                if not are_used:
                    counter = counter + 1
                    new_indices[counter] = indices[i]

            self.columns[y].neighbour_indices = new_indices
            self.columns[y].is_unpopular = False
            self.columns[y].is_popular = True

        else:

            index_positions = np.zeros([found], dtype=int)

            for k in range(0, found):

                for z in range(0, indices.shape[0]):

                    if indices[z] == new_indices[k]:
                        index_positions[k] = z

            counter = found - 1

            for i in range(0, indices.shape[0]):

                are_used = False

                for z in range(0, found):

                    if i == index_positions[z]:
                        are_used = True

                if not are_used:
                    counter = counter + 1
                    new_indices[counter] = indices[i]

            self.columns[y].neighbour_indices = new_indices
            self.columns[y].is_unpopular = True
            self.columns[y].is_popular = False

def sort_neighbours_by_level(self, y):

    n = self.columns[y].n()

    num_wrong_flags = 0

    for x in range(0, n):

        if self.columns[self.columns[y].neighbour_indices[x]].level == self.columns[y].level:
            num_wrong_flags = num_wrong_flags + 1

    if num_wrong_flags >= n - 1:

        if self.columns[y].level == 0:
            self.columns[y].level = 1
        else:
            self.columns[y].level = 0

        num_wrong_flags = 0

        for x in range(0, n):

            if self.columns[self.columns[y].neighbour_indices[x]].level == self.columns[y].level:
                num_wrong_flags = num_wrong_flags + 1

    finished = False
    debug_counter = 0

    while not finished:

        print(debug_counter)

        num_perturbations = 0

        for x in range(0, n - 1):

            if not self.columns[self.columns[y].neighbour_indices[x]].level == self.columns[y].level:
                pass
            else:
                self.perturbator(y, x, x + 1)
                num_perturbations = num_perturbations + 1

            if x == n - num_wrong_flags - 2 and num_perturbations == 0:

                finished = True

        debug_counter = debug_counter + 1

def find_consistent_perturbations_advanced(self, y, experimental=False):

    if not self.columns[y].is_edge_column:

        n = self.columns[y].n()

        indices = deepcopy(self.columns[y].neighbour_indices)
        new_indices = np.zeros([indices.shape[0]], dtype=int)
        index_of_unpopular_neighbours = np.zeros([indices.shape[0]], dtype=int)
        found = 0

        for x in range(0, indices.shape[0]):

            if self.columns[indices[x]].is_unpopular:
                index_of_unpopular_neighbours[x] = indices[x]
            else:
                index_of_unpopular_neighbours[x] = -1

            n2 = 3

            if self.columns[indices[x]].h_index == 0 or self.columns[indices[x]].h_index == 1:
                n2 = 3
            elif self.columns[indices[x]].h_index == 3:
                n2 = 4
            elif self.columns[indices[x]].h_index == 5:
                n2 = 5
            else:
                print('Problem in find_consistent_perturbations_simple!')

            neighbour_indices = self.columns[indices[x]].neighbour_indices

            for z in range(0, n2):

                if neighbour_indices[z] == y:
                    new_indices[found] = indices[x]
                    found = found + 1
                    index_of_unpopular_neighbours[x] = -1

        if found == n:

            index_positions = np.zeros([found], dtype=int)

            for k in range(0, found):

                for z in range(0, indices.shape[0]):

                    if indices[z] == new_indices[k]:
                        index_positions[k] = z

            counter = found - 1

            for i in range(0, indices.shape[0]):

                are_used = False

                for z in range(0, found):

                    if i == index_positions[z]:

                        are_used = True

                if not are_used:

                    counter = counter + 1
                    new_indices[counter] = indices[i]

            self.columns[y].neighbour_indices = new_indices
            self.columns[y].is_unpopular = False
            self.columns[y].is_popular = False

            if experimental:
                self.sort_neighbours_by_level(y)

        elif found > n:

            index_positions = np.zeros([found], dtype=int)

            for k in range(0, found):

                for z in range(0, indices.shape[0]):

                    if indices[z] == new_indices[k]:
                        index_positions[k] = z

            counter = found - 1

            for i in range(0, indices.shape[0]):

                are_used = False

                for z in range(0, found):

                    if i == index_positions[z]:
                        are_used = True

                if not are_used:
                    counter = counter + 1
                    new_indices[counter] = indices[i]

            self.columns[y].neighbour_indices = new_indices
            self.columns[y].is_popular = True
            self.columns[y].is_unpopular = False

            if experimental:
                self.sort_neighbours_by_level(y)

        else:

            print(index_of_unpopular_neighbours)

            index_positions = np.zeros([found], dtype=int)

            for k in range(0, found):

                for z in range(0, indices.shape[0]):

                    if indices[z] == new_indices[k]:
                        index_positions[k] = z

            counter = found - 1

            for i in range(0, indices.shape[0]):

                are_used = False

                for z in range(0, found):

                    if i == index_positions[z]:
                        are_used = True

                if not are_used:
                    counter = counter + 1
                    new_indices[counter] = indices[i]

            self.columns[y].neighbour_indices = new_indices
            self.columns[y].is_unpopular = True
            self.columns[y].is_popular = False

            friend_index = -1
            distance = self.N

            for x in range(0, indices.shape[0]):

                if not index_of_unpopular_neighbours[x] == -1:

                    temp_distance = np.sqrt((self.columns[y].x -
                                             self.columns[index_of_unpopular_neighbours[x]].x)**2 +
                                            (self.columns[y].y -
                                             self.columns[index_of_unpopular_neighbours[x]].y)**2)

                    if temp_distance < distance:
                        distance = temp_distance
                        friend_index = index_of_unpopular_neighbours[x]

            if not friend_index == -1:

                i_1 = -1

                for j in range(0, indices.shape[0]):

                    if new_indices[j] == friend_index:
                        i_1 = j

                print('y: ' + str(y) + ', found: ' + str(found) + ', i_1: ' + str(i_1) + ', friend_index: ' + str(friend_index))

                self.columns[y].neighbour_indices[i_1] = self.columns[y].neighbour_indices[found]
                self.columns[y].neighbour_indices[found] = friend_index

                self.find_consistent_perturbations_simple(friend_index)
                self.find_consistent_perturbations_simple(y)

            else:

                distance = self.N
                friend_index = -1

                for x in range(found, indices.shape[0]):

                    if not self.columns[self.columns[y].neighbour_indices[x]].level == self.columns[y].level:

                        temp_distance = np.sqrt((self.columns[y].x -
                                                 self.columns[self.columns[y].neighbour_indices[x]].x) ** 2 +
                                                (self.columns[y].y -
                                                 self.columns[self.columns[y].neighbour_indices[x]].y) ** 2)

                        if temp_distance < distance:
                            distance = temp_distance
                            friend_index = self.columns[y].neighbour_indices[x]

                if not friend_index == -1:

                    i_1 = -1

                    for j in range(0, indices.shape[0]):

                        if new_indices[j] == friend_index:
                            i_1 = j

                    self.columns[y].neighbour_indices[i_1] = self.columns[y].neighbour_indices[found]
                    self.columns[y].neighbour_indices[found] = friend_index

                    self.find_consistent_perturbations_simple(friend_index)
                    self.find_consistent_perturbations_simple(y)

            if experimental:
                self.sort_neighbours_by_level(y)

def connection_shift_on_level(self, i, experimental=False):

    n = self.columns[i].n()
    indices = self.columns[i].neighbour_indices

    bad_index = -1
    good_index = -1

    print(str(i) + ': n = ' + str(n) + '\n----------------')

    for x in range(0, n):

        print('k: ' + str(x))

        if self.columns[indices[x]].level == self.columns[i].level:

            bad_index = x

    if experimental:

        high = n + 1

    else:

        high = 8

    for x in range(n, high):

        print('j: ' + str(n + high - 1 - x))

        if not self.columns[indices[n + high - 1 - x]].level == self.columns[i].level:

            good_index = n + high - 1 - x

    if not bad_index == -1 and not good_index == -1:
        print(
            str(i) + ' | ' + str(bad_index) + ': ' + str(indices[bad_index]) + ' | ' + str(good_index) + ': ' + str(
                indices[good_index]))
        self.perturbator(i, bad_index, good_index)

def reset_popularity_flags(self):

    for x in range(0, self.num_columns):
        self.columns[x].is_popular = False
        self.columns[x].is_unpopular = False

    self.num_unpopular = 0
    self.num_popular = 0
    self.num_inconsistencies = 0

def find_shape(self, i, j, clockwise=True):

    closed = False
    start_index = i
    shape_indices = np.ndarray([2], dtype=int)
    shape_indices[0] = i
    shape_indices[1] = j

    while not closed:

        i = shape_indices[shape_indices.shape[0] - 2]
        j = shape_indices[shape_indices.shape[0] - 1]

        if j == start_index or shape_indices.shape[0] > 7:

            closed = True

        else:

            next_index = -1

            if not self.test_reciprocality(i, j):

                if clockwise:
                    sorted_indices, alpha = self.clockwise_neighbour_sort(j, j=i)
                else:
                    sorted_indices, alpha = self.anticlockwise_neighbour_sort(j, j=i)

                next_index = self.columns[j].n()

            else:

                if clockwise:
                    sorted_indices, alpha = self.clockwise_neighbour_sort(j)
                else:
                    sorted_indices, alpha = self.anticlockwise_neighbour_sort(j)

                for x in range(0, self.columns[j].n()):

                    if sorted_indices[x] == i:

                        if x == 0:
                            next_index = self.columns[j].n() - 1
                        else:
                            next_index = x - 1

            next_index = sorted_indices[next_index]

            shape_indices = np.append(shape_indices, next_index)

    return shape_indices, shape_indices.shape[0] - 1

def clockwise_neighbour_sort(self, i, j=-1):

    n = self.columns[i].n()
    print('n: ' + str(n))

    if not j == -1:
        n = self.columns[i].n() + 1
        print('changed n: ' + str(n))

    a = np.ndarray([n], dtype=np.int)
    b = np.ndarray([n], dtype=np.int)
    alpha = np.ndarray([n - 1], dtype=np.float64)
    indices = np.ndarray([n - 1], dtype=int)
    sorted_indices = np.ndarray([n], dtype=int)

    if not j == -1:

        sorted_indices[0] = j

        a[0] = self.columns[j].x - self.columns[i].x
        b[0] = self.columns[j].y - self.columns[i].y

        for x in range(1, n):
            a[x] = self.columns[self.columns[i].neighbour_indices[x - 1]].x - self.columns[i].x
            b[x] = self.columns[self.columns[i].neighbour_indices[x - 1]].y - self.columns[i].y

        for x in range(0, n - 1):
            indices[x] = self.columns[i].neighbour_indices[x]

        for x in range(1, n):

            alpha[x - 1] = utils.find_angle(a[0], a[x], b[0], b[x])

            if utils.vector_cross_product_magnitude(a[0], a[x], b[0], b[x]) < 0:
                alpha[x - 1] = 2 * np.pi - alpha[x - 1]

        alpha, indices = utils.dual_sort(alpha, indices)

        for x in range(0, n - 1):
            sorted_indices[x + 1] = indices[x]

        alpha = np.append(alpha, 2 * np.pi)

    else:

        sorted_indices[0] = self.columns[i].neighbour_indices[0]

        for x in range(1, n):
            indices[x - 1] = self.columns[i].neighbour_indices[x]

        for x in range(0, n):
            a[x] = self.columns[self.columns[i].neighbour_indices[x]].x - self.columns[i].x
            b[x] = self.columns[self.columns[i].neighbour_indices[x]].y - self.columns[i].y

        for x in range(1, n):

            indices[x - 1] = self.columns[i].neighbour_indices[x]

            alpha[x - 1] = utils.find_angle(a[0], a[x], b[0], b[x])

            if utils.vector_cross_product_magnitude(a[0], a[x], b[0], b[x]) < 0:

                alpha[x - 1] = 2 * np.pi - alpha[x - 1]

        alpha, indices = utils.dual_sort(alpha, indices)

        for x in range(0, n - 1):

            sorted_indices[x + 1] = indices[x]

        alpha = np.append(alpha, 2 * np.pi)

    return sorted_indices, alpha

def anticlockwise_neighbour_sort(self, i, j=-1):

    n = self.columns[i].n()
    print('n: ' + str(n))

    if not j == -1:
        n = self.columns[i].n() + 1
        print('changed n: ' + str(n))

    a = np.ndarray([n], dtype=np.int)
    b = np.ndarray([n], dtype=np.int)
    alpha = np.ndarray([n - 1], dtype=np.float64)
    indices = np.ndarray([n - 1], dtype=int)
    sorted_indices = np.ndarray([n], dtype=int)

    if not j == -1:

        sorted_indices[0] = j

        a[0] = self.columns[j].x - self.columns[i].x
        b[0] = self.columns[j].y - self.columns[i].y

        for x in range(1, n):
            a[x] = self.columns[self.columns[i].neighbour_indices[x - 1]].x - self.columns[i].x
            b[x] = self.columns[self.columns[i].neighbour_indices[x - 1]].y - self.columns[i].y

        for x in range(0, n - 1):
            indices[x] = self.columns[i].neighbour_indices[x]

        for x in range(1, n):

            alpha[x - 1] = utils.find_angle(a[0], a[x], b[0], b[x])

            if utils.vector_cross_product_magnitude(a[0], a[x], b[0], b[x]) > 0:
                alpha[x - 1] = 2 * np.pi - alpha[x - 1]

        alpha, indices = utils.dual_sort(alpha, indices)

        for x in range(0, n - 1):
            sorted_indices[x + 1] = indices[x]

        alpha = np.append(alpha, 2 * np.pi)

    else:

        sorted_indices[0] = self.columns[i].neighbour_indices[0]

        for x in range(1, n):
            indices[x - 1] = self.columns[i].neighbour_indices[x]

        for x in range(0, n):
            a[x] = self.columns[self.columns[i].neighbour_indices[x]].x - self.columns[i].x
            b[x] = self.columns[self.columns[i].neighbour_indices[x]].y - self.columns[i].y

        for x in range(1, n):

            indices[x - 1] = self.columns[i].neighbour_indices[x]

            alpha[x - 1] = utils.find_angle(a[0], a[x], b[0], b[x])

            if utils.vector_cross_product_magnitude(a[0], a[x], b[0], b[x]) > 0:

                alpha[x - 1] = 2 * np.pi - alpha[x - 1]

        alpha, indices = utils.dual_sort(alpha, indices)

        for x in range(0, n - 1):

            sorted_indices[x + 1] = indices[x]

        alpha = np.append(alpha, 2 * np.pi)

    return sorted_indices, alpha

'''

