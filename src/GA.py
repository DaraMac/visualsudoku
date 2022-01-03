import numpy as np
import random
import operator
from past.builtins import range

random.seed()

# Nd x Nd sudoku
Nd = 9  

# A set of candidate solutions to the Sudoku puzzle(chromosomes in the population).
class Population(object):

    def __init__(self):
        self.candidates = []
        return

    def seed(self, Nc, given):
        self.candidates = []

        # Determine the legal values that each square can take.
        helper = Candidate()
        helper.values = [[[] for j in range(0, Nd)] for i in range(0, Nd)]
        for row in range(0, Nd):
            for column in range(0, Nd):
                for value in range(1, 10):
                    if ((given.values[row][column] == 0) and not (given.is_column_duplicate(column, value) or given.is_block_duplicate(row, column, value) or given.is_row_duplicate(row, value))):
                        # Value is available.
                        helper.values[row][column].append(value)
                    elif given.values[row][column] != 0:
                        # Given/known value from file.
                        helper.values[row][column].append(given.values[row][column])
                        break

# =============================================================================
#         for i in range(0,len(helper.values)):
#             print("possiable digit for row", i+1 ,"are: ", helper.values[i])
# =============================================================================

        # Seed a new population.
        for p in range(0, Nc):
            g = Candidate()
            for i in range(0, Nd):  # New row in candidate.
                row = np.zeros(Nd)

                # Fill in the givens.
                for j in range(0, Nd):  # New column j value in row i.

                    # If value is already given, don't change it.
                    if given.values[i][j] != 0:
                        row[j] = given.values[i][j]
                    # Fill in the gaps using the helper board.
                    elif given.values[i][j] == 0:
                        row[j] = helper.values[i][j][random.randint(0, len(helper.values[i][j]) - 1)]

                # If we don't have a valid board, then try again. max iteration 500,000
                # There must be no duplicates in the row.
                ii = 0
                while len(list(set(row))) != Nd:
                    ii += 1
                    if ii > 500000:
                        return 0
                    for j in range(0, Nd):
                        if given.values[i][j] == 0:
                            row[j] = helper.values[i][j][random.randint(0, len(helper.values[i][j]) - 1)]

                g.values[i] = row
            # print(g.values)
            self.candidates.append(g)

# =============================================================================
#         print("Number of candidates created: ", len(self.candidates))
# =============================================================================

        # Compute the fitness of all candidates in the population.
        self.update_fitness()

        # print("Seeding complete.")

        return 1

    def update_fitness(self):
        # Update fitness of every candidate. 
        for candidate in self.candidates:

            candidate.update_fitness()
        return

    def sort(self):
        # Sort the population based on fitness.
        self.candidates = sorted(self.candidates, key=operator.attrgetter('fitness'))
        return

# A candidate solutions for Sudoku
class Candidate(object):


    def __init__(self):
        self.values = np.zeros((Nd, Nd))
        self.fitness = None
        return

      # The fitness of a candidate solution is determined by how close it is to being the actual solution to the puzzle.
      # The actual solution is defined as a 9x9 grid of numbers in the range 1 to 9, where each row & column & 3x3 block contains the numbers 1 to 9 without any duplicates
      # Any duplicates will lower down the fitness. 
    def update_fitness(self):


        column_count = np.zeros(Nd)
        block_count = np.zeros(Nd)
        column_sum = 0
        block_sum = 0

        self.values = self.values.astype(int)
# =============================================================================
#         print("Candidate\n", self.values)
# =============================================================================
        # For each column....
        for j in range(0, Nd):
            for i in range(0, Nd):
                column_count[self.values[i][j] - 1] += 1
# =============================================================================
#                 print("column",j," after the count the ",i ,"digit, the column count become", column_count)
# =============================================================================

            # unique
            # column_sum += (1.0 / len(set(column_count))) / Nd
            # set
            # for k in range(len(column_count)):
            #     if column_count[k] != 0:
            #         column_sum += (1/Nd)/Nd
            # duplicate
            for k in range(len(column_count)):
                if column_count[k] == 1:
                    column_sum += (1/Nd)/Nd
# =============================================================================
#                     print("At",k,"column_sum",column_sum)
#             print("column_sum!!!",column_sum)
# =============================================================================
            column_count = np.zeros(Nd)

        # For each block...
        for i in range(0, Nd, 3):
            for j in range(0, Nd, 3):
                block_count[self.values[i][j] - 1] += 1
                block_count[self.values[i][j + 1] - 1] += 1
                block_count[self.values[i][j + 2] - 1] += 1

                block_count[self.values[i + 1][j] - 1] += 1
                block_count[self.values[i + 1][j + 1] - 1] += 1
                block_count[self.values[i + 1][j + 2] - 1] += 1

                block_count[self.values[i + 2][j] - 1] += 1
                block_count[self.values[i + 2][j + 1] - 1] += 1
                block_count[self.values[i + 2][j + 2] - 1] += 1

                # unique
                # block_sum += (1.0 / len(set(block_count))) / Nd
                # set
                # for k in range(len(block_count)):
                #     if block_count[k] != 0:
                #         block_sum += (1/Nd)/Nd
                # duplicate
                for k in range(len(block_count)):
                    if block_count[k] == 1:
                        block_sum += (1/Nd)/Nd
                block_count = np.zeros(Nd)

        # Calculate overall fitness.
        if int(column_sum) == 1 and int(block_sum) == 1:
            fitness = 1.0
        else:
            fitness = column_sum * block_sum

        self.fitness = fitness
        return

    #  Mutate a candidate by picking a row, and then picking two values within that row to swap. 
    def mutate(self, mutation_rate, given):
        

        r = random.uniform(0, 1.1)
        while r > 1:  # Outside [0, 1] boundary - choose another
            r = random.uniform(0, 1.1)

        success = False
        if r < mutation_rate:  # Mutate.
            while not success:
                row1 = random.randint(0, 8)
                row2 = random.randint(0, 8)
                row2 = row1

                from_column = random.randint(0, 8)
                to_column = random.randint(0, 8)
                while from_column == to_column:
                    from_column = random.randint(0, 8)
                    to_column = random.randint(0, 8)

                    # Check if the two places are free to swap
                if given.values[row1][from_column] == 0 and given.values[row1][to_column] == 0:
                    # ...and that we are not causing a duplicate in the rows' columns.
                    if not given.is_column_duplicate(to_column, self.values[row1][from_column]) and not given.is_column_duplicate(from_column, self.values[row2][to_column]) and not given.is_block_duplicate(row2, to_column, self.values[row1][from_column]) and not given.is_block_duplicate(row1, from_column, self.values[row2][to_column]):
                        # Swap values.
                        temp = self.values[row2][to_column]
                        self.values[row2][to_column] = self.values[row1][from_column]
                        self.values[row1][from_column] = temp
                        success = True

        return success

# fixed/given values
class Fixed(Candidate):


    def __init__(self, values):
        self.values = values
        return

    # Check duplicate in a row.
    def is_row_duplicate(self, row, value):
        for column in range(0, Nd):
            if self.values[row][column] == value:
                return True
        return False

    # Check duplicate in a column. 
    def is_column_duplicate(self, column, value):
        
        for row in range(0, Nd):
            if self.values[row][column] == value:
                return True
        return False

   # Check duplicate in a block.
    def is_block_duplicate(self, row, column, value):

        i = 3 * (int(row / 3))
        j = 3 * (int(column / 3))

        if ((self.values[i][j] == value)
            or (self.values[i][j + 1] == value)
            or (self.values[i][j + 2] == value)
            or (self.values[i + 1][j] == value)
            or (self.values[i + 1][j + 1] == value)
            or (self.values[i + 1][j + 2] == value)
            or (self.values[i + 2][j] == value)
            or (self.values[i + 2][j + 1] == value)
            or (self.values[i + 2][j + 2] == value)):
            return True
        else:
            return False

    def make_index(self, v):
        if v <= 2:
            return 0
        elif v <= 5:
            return 3
        else:
            return 6

    def no_duplicates(self):
        for row in range(0, Nd):
            for col in range(0, Nd):
                if self.values[row][col] != 0:

                    cnt1 = list(self.values[row]).count(self.values[row][col])
                    cnt2 = list(self.values[:,col]).count(self.values[row][col])

                    block_values = [y[self.make_index(col):self.make_index(col)+3] for y in
                                    self.values[self.make_index(row):self.make_index(row)+3]]
                    block_values_ = [int(x) for y in block_values for x in y]
                    cnt3 = block_values_.count(self.values[row][col])

                    if cnt1 > 1 or cnt2 > 1 or cnt3 > 1:
                        return False
        return True


# The crossover function requires two parents to be selected from the population pool. The Tournament class is used to do this.
# Two individuals are selected from the population pool and a random number in [0, 1] is chosen.
# If this number is less than the 'selection rate' (e.g. 0.80), then the fitter individual is selected; otherwise, the weaker one is selected.

class Tournament(object):


    def __init__(self):
        return

    # Pick 2 random candidates from the population and get them to compete against each other.
    def compete(self, candidates):

        c1 = candidates[random.randint(0, len(candidates) - 1)]
        c2 = candidates[random.randint(0, len(candidates) - 1)]
        f1 = c1.fitness
        f2 = c2.fitness

        # Find the fittest and the weakest.
        if (f1 > f2):
            fittest = c1
            weakest = c2
        else:
            fittest = c2
            weakest = c1

        # selection_rate = 0.85
        selection_rate = 0.80
        r = random.uniform(0, 1.1)
        while (r > 1):  # Outside [0, 1] boundary. Choose another.
            r = random.uniform(0, 1.1)
        if (r < selection_rate):
            return fittest
        else:
            return weakest

# Crossover relates to the analogy of genes within each parent candidate mixing together in the hopes of creating a fitter child candidate.
# REF: A. E. Eiben, J. E. Smith. Introduction to Evolutionary Computing. Springer, 2007
class CycleCrossover(object):

    def __init__(self):
        return

    def crossover(self, parent1, parent2, crossover_rate):    
        # Create two new child candidates by crossing over parent genes. 
        child1 = Candidate()
        child2 = Candidate()

        # Make a copy of the parent genes.
        child1.values = np.copy(parent1.values)
        child2.values = np.copy(parent2.values)

        r = random.uniform(0, 1.1)
        while (r > 1):  # Outside [0, 1] boundary. Choose another.
            r = random.uniform(0, 1.1)

        # Perform crossover.
        if (r < crossover_rate):
            # Pick a crossover point. Crossover must have at least 1 row (and at most Nd-1) rows.
            crossover_point1 = random.randint(0, 8)
            crossover_point2 = random.randint(1, 9)
            while (crossover_point1 == crossover_point2):
                crossover_point1 = random.randint(0, 8)
                crossover_point2 = random.randint(1, 9)

            if (crossover_point1 > crossover_point2):
                temp = crossover_point1
                crossover_point1 = crossover_point2
                crossover_point2 = temp

            for i in range(crossover_point1, crossover_point2):
                child1.values[i], child2.values[i] = self.crossover_rows(child1.values[i], child2.values[i])

        return child1, child2

    def crossover_rows(self, row1, row2):
# =============================================================================
#         print("P1row:", row1, "P2row", row2)
# =============================================================================
        child_row1 = np.zeros(Nd)
        child_row2 = np.zeros(Nd)

        remaining = range(1, Nd + 1)
        cycle = 0

        while ((0 in child_row1) and (0 in child_row2)):  # While child rows not complete...
            if (cycle % 2 == 0):  # Even cycles.
                # Assign next unused value.
                index = self.find_unused(row1, remaining)
                start = row1[index]
                remaining.remove(row1[index])
                child_row1[index] = row1[index]
                child_row2[index] = row2[index]
                next = row2[index]

                while (next != start):  # While cycle not done...
                    index = self.find_value(row1, next)
                    child_row1[index] = row1[index]
                    remaining.remove(row1[index])
                    child_row2[index] = row2[index]
                    next = row2[index]

                cycle += 1

            else:  # Odd cycle - flip values.
                index = self.find_unused(row1, remaining)
                start = row1[index]
                remaining.remove(row1[index])
                child_row1[index] = row2[index]
                child_row2[index] = row1[index]
                next = row2[index]

                while (next != start):  # While cycle not done...
                    index = self.find_value(row1, next)
                    child_row1[index] = row2[index]
                    remaining.remove(row1[index])
                    child_row2[index] = row1[index]
                    next = row2[index]

                cycle += 1
# =============================================================================
#         print("child1_row:", child_row1, "child2_row", child_row2)
# =============================================================================
        return child_row1, child_row2

    def find_unused(self, parent_row, remaining):
        for i in range(0, len(parent_row)):
            if (parent_row[i] in remaining):
# =============================================================================
#                 print("parent_row", parent_row,"remaining",remaining)
# =============================================================================
                return i

    def find_value(self, parent_row, value):
        for i in range(0, len(parent_row)):
            if (parent_row[i] == value):
                return i

# Solves a given Sudoku using GA
class Sudoku(object):


    def __init__(self):
        self.given = None
        return

    def load(self, p):
        #values = np.array(list(p.replace(".","0"))).reshape((Nd, Nd)).astype(int)
        self.given = Fixed(p)
        return

    def solve(self):

        Nc = 1000  # Number of candidates (i.e. population size).
        Ne = int(0.05 * Nc)  # Number of elites.
        Ng = 10000  # Number of generations.
        Nm = 0  # Number of mutations.

        # Mutation parameters.
        phi = 0
        sigma = 1
        mutation_rate = 0.06

        # Check given one first
        if self.given.no_duplicates() == False:
            return (-1, 1)

        # Create an initial population.
        self.population = Population()
        print("create an initial population.")
        if self.population.seed(Nc, self.given) ==  1:
            pass
        else:
            return (-1, 1)

        # For up to 10000 generations...
        stale = 0
        for generation in range(0, Ng):

            # Check for a solution.
            best_fitness = 0.0
            #best_fitness_population_values = self.population.candidates[0].values
            for c in range(0, Nc):
                fitness = self.population.candidates[c].fitness
                if (fitness == 1):
                    print("At generation %d!" % generation, "the fitness is 1.0(solution) " )
                    return (generation, self.population.candidates[c])

                # Find the best fitness and corresponding chromosome
                if (fitness > best_fitness):
                    best_fitness = fitness
                    #best_fitness_population_values = self.population.candidates[c].values

            print("The best fitness at", generation,"'th generation is", best_fitness)
            #print(best_fitness_population_values)

            # Create the next population.
            next_population = []

            # Select elites (the fittest candidates) and preserve them for the next generation.
            self.population.sort()
            elites = []
            for e in range(0, Ne):
                elite = Candidate()
                elite.values = np.copy(self.population.candidates[e].values)
                elites.append(elite)

            # Create the rest of the candidates.
            for count in range(Ne, Nc, 2):
                # Select parents from population via a tournament.
                t = Tournament()
                parent1 = t.compete(self.population.candidates)
                parent2 = t.compete(self.population.candidates)

                ## Cross-over.
                cc = CycleCrossover()
                child1, child2 = cc.crossover(parent1, parent2, crossover_rate=1.0)

                # Mutate child1.
                child1.update_fitness()
                old_fitness = child1.fitness
                success = child1.mutate(mutation_rate, self.given)
                child1.update_fitness()
                if (success):
                    Nm += 1
                    if (child1.fitness > old_fitness):  # Used to calculate the relative success rate of mutations.
                        phi = phi + 1

                # Mutate child2.
                child2.update_fitness()
                old_fitness = child2.fitness
                success = child2.mutate(mutation_rate, self.given)
                child2.update_fitness()
                if (success):
                    Nm += 1
                    if (child2.fitness > old_fitness):  # Used to calculate the relative success rate of mutations.
                        phi = phi + 1

                # Add children to new population.
                next_population.append(child1)
                next_population.append(child2)

            # Append elites onto the end of the population. These will not have been affected by crossover or mutation.
            for e in range(0, Ne):
                next_population.append(elites[e])

            # Select next generation.
            self.population.candidates = next_population
            self.population.update_fitness()

            # Calculate new adaptive mutation rate (based on Rechenberg's 1/5 success rule).
            # This is to stop too much mutation as the fitness progresses towards unity.
            if (Nm == 0):
                phi = 0  # Avoid divide by zero.
            else:
                phi = phi / Nm

            if (phi > 0.2):
                sigma = sigma / 0.998
                print("success rate > 20%")
            elif (phi < 0.2):
                sigma = sigma * 0.998
                print("success rate < 20%")

            print("mutation_rate ", sigma)
            mutation_rate = abs(np.random.normal(loc=0.0, scale=sigma, size=None))
            print("mutation_rate after update: ", mutation_rate)

            # Check for stale population.
            self.population.sort()
            if (self.population.candidates[0].fitness != self.population.candidates[1].fitness):
                stale = 0
            else:
                stale += 1

            # Re-seed the population if 100 generations have passed
            # with the fittest two candidates always having the same fitness.
            if (stale >= 100):
                print("The population has gone stale. Re-seeding...")
                self.population.seed(Nc, self.given)
                stale = 0
                sigma = 1
                phi = 0
                mutation_rate = 0.06

        print("No solution found.")
        return (-2, 1)