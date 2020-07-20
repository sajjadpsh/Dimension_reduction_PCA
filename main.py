import numpy as np
import matplotlib.pyplot as plt
import csv
from statistics import variance
import random
import math

generations = 70
score = 0
Mu = 80
Lambda = 2 * Mu
crossover_probability = 0.4
TOURNAMENT_SIZE = 4
gen = []

def read_from_file():
    x_raw = []
    y_raw = []

    with open('Dataset2.csv', 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for row in plots:
            x_raw.append(row[0])
            y_raw.append(row[1])

    x_raw = x_raw[1:]
    y_raw = y_raw[1:]
    coord = [x_raw, y_raw]

    return coord


class Chromosome:
    def __init__(self, chromosome_length, min, max):

        self.gene = []
        self.score = 0
        self.s = 0
        for i in range(chromosome_length):
            self.gene.append(random.uniform(min, max))
            self.s += self.gene[i] * self.gene[i]

        # Normalize the genes
        sum = math.sqrt(self.s)
        for i in range(chromosome_length):
            self.gene[i] /= sum

    def evaluate(self):
        arr = read_from_file()

        x = [float(arr[0][i]) for i in range(len(arr[0]))]
        y = [float(arr[1][i]) for i in range(len(arr[1]))]
        z = []
        for i in range(len(x)):
            z.append(self.gene[0] * x[i] + self.gene[1] * y[i])
        z_sigma = math.sqrt(variance(z))
        self.score = z_sigma


def generate_initial_population():
    list_of_chromosomes = []
    for i in range(Mu):
        list_of_chromosomes.append(Chromosome(2, -20, 20))
        list_of_chromosomes[i].evaluate()
    return list_of_chromosomes


def generate_new_seed(pop):
    seed = []
    temp = np.arange(Mu)
    np.random.shuffle(temp)
    for i in range(Lambda):
        t = random.randint(0, Mu - 1)
        seed.append(pop[temp[t]])
    return seed


def crossover(chromosome1, chromosome2):
    for i in range(len(chromosome1.gene)):
        if random.random() >= 0.5:
            chromosome1.gene[i] = chromosome2.gene[i]
    return chromosome1


def cross(gener):
    for ch in gener:
        if random.random() >= 1 - crossover_probability:
            temp = gener[random.randint(0, len(gener) - 1)]
            crossover(ch, temp)
    return gener


def mutation(chromosome):
    sum = 0
    for i in range(len(chromosome.gene)):
        chromosome.gene[i] += np.random.normal(0, 0.01, 1)[0]
        sum += chromosome.gene[i] * chromosome.gene[i]
    for i in range(len(chromosome.gene)):
        chromosome.gene[i] = chromosome.gene[i] / math.sqrt(sum)
    return chromosome


def evaluate_new_generation(generation):
    for i in range(len(generation)):
        generation[i].evaluate()
    return generation


def choose_new_generation(generation):
    population = []
    for i in range(Mu):
        q_tornomet = []
        for j in range(TOURNAMENT_SIZE):
            temp = random.randint(0, len(generation) - 1)
            q_tornomet.append(generation[temp])
        best = q_tornomet[0]
        for k in range(TOURNAMENT_SIZE):
            if (q_tornomet[k].score >= best.score):
                best = q_tornomet[k]
        population.append(best)

    best = population[0]
    worst = population[0]
    average = 0
    for i in range(Mu):
        if best.score <= population[i].score:
            best = population[i]
        if worst.score >= population[i].score:
            worst = population[i]
        average += population[i].score
    average /= len(population)
    print("Best Score:" + str(best.score) + "\nWorst score: " + str(worst.score) + "\nAverage:" + str(average))
    return population


def plot(array, chromosome):
    x = [float(array[0][i]) for i in range(len(array[0]))]
    y = [float(array[1][i]) for i in range(len(array[1]))]
    z = [chromosome[0] * x[i] + chromosome[1] * y[i] for i in range(len(x))]
    x_normal = [z[i] * chromosome[0] for i in range(len(z))]
    y_normal = [z[i] * chromosome[1] for i in range(len(z))]
    plt.plot(x, y, '.', color='blue')
    plt.plot(x_normal, y_normal, 'o', color='green',linewidth='3')
    plt.xlim(0, 100 * chromosome[0])
    plt.ylim(0, 100 * chromosome[1])
    tilt = "a=" + str(chromosome[0]) + "\nb=" + str(chromosome[1])
    plt.title(tilt)
    plt.axis([-10, 60, -10, 100])
    plt.show()


if __name__ == '__main__':

    initial_population = generate_initial_population()

    for chromosom in initial_population:
        if chromosom.score > score:
            gen = chromosom.gene
            score = chromosom.score

    for i in range(generations):
        print("Generation Number " + str(i + 1))
        generation = generate_new_seed(initial_population)
        for chromosom in generation:
            chromosom = mutation(chromosom)
        generation = cross(generation)
        generation = evaluate_new_generation(generation)
        initial_population = choose_new_generation(generation)

    for chromosom in initial_population:
        if chromosom.score > score:
            gen = chromosom.gene
            score = chromosom.score

    plot(read_from_file(), gen)
