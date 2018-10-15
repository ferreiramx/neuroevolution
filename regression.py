#!/usr/bin/python3
from __future__ import print_function

import sys
import time
import numpy as np
import pandas as pd
import MultiNEAT as NEAT

#################################################
global_best = -99999999
dataset = pd.read_csv(sys.argv[1], thousands=',')
rows = dataset.shape[0]
cols = dataset.shape[1]
print("Rows: ", rows, "Columns: ", cols)
vars = dataset[dataset.columns[0:(cols-2)]]
target = dataset[dataset.columns[(cols-1)]]
rrse_list = [-9999999] * rows
mae_list = [-9999999] * rows
generations = 0

#################################################

params = NEAT.Parameters()
params.PopulationSize = 50
params.DynamicCompatibility = True
params.NormalizeGenomeSize = True
params.WeightDiffCoeff = 0
params.CompatTreshold = 2.0
params.YoungAgeTreshold = 15
params.SpeciesMaxStagnation = 15
params.OldAgeTreshold = 35
params.MinSpecies = 1
params.MaxSpecies = 250
params.RouletteWheelSelection = False
params.RecurrentProb = 0.1
params.OverallMutationRate = 0.3

params.ArchiveEnforcement = False

params.MutateWeightsProb = 0.25

params.WeightMutationMaxPower = 1.0
params.WeightReplacementMaxPower = 1.0
params.MutateWeightsSevereProb = 0.35
params.WeightMutationRate = 0.25
params.WeightReplacementRate = 0.9

params.MaxWeight = 50000.0

params.MutateAddNeuronProb = 0.5
params.MutateAddLinkProb = 0.3
params.MutateRemLinkProb = 0.1
params.MutateRemSimpleNeuronProb = 0.1

# params.MinActivationA = 4.9
# params.MaxActivationA = 4.9

params.ActivationFunction_SignedSigmoid_Prob = 0
params.ActivationFunction_UnsignedSigmoid_Prob = 0
params.ActivationFunction_Tanh_Prob = 0
params.ActivationFunction_SignedStep_Prob = 0
params.ActivationFunction_Linear_Prob = 0.33
params.ActivationFunction_Relu_Prob = 0.33
params.ActivationFunction_Softmax_Prob = 0.33

params.CrossoverRate = 0.6
params.MultipointCrossoverRate = 0.6
params.SurvivalRate = 0.3

params.MutateNeuronTraitsProb = 0.3
params.MutateLinkTraitsProb = 0.3

params.AllowLoops = True
params.AllowClones = True


#################################################

def evaluate(genome, skip_idx):

    global generations
    global global_best
    global vars
    global target
    global rrse_list
    global mae_list
    global rows
    global cols

    net = NEAT.NeuralNetwork()
    genome.BuildPhenotype(net)
    error = 0
    denominator = 0
    mae = 0

    avgOutput = target.drop(target.index[skip_idx]).reset_index(drop=True).mean()

    for idx in range(rows):
        if idx == skip_idx:
            continue
        # print("Using row: ", idx)
        net.Flush()
        row = np.array(vars.loc[idx])
        inputs = np.concatenate((row, np.array([1.0])))
        # print("Inputs: ", inputs)
        net.Input(inputs)
        net.Activate()
        output = net.Output()
        # print("Network output: ", output[0])
        error += (target.iat[idx] - output[0]) ** 2
        mae += abs(target.iat[idx] - output[0])
        denominator += (target.iat[idx] - avgOutput) ** 2
        # print("Error: ", error)

    mae = -(mae / (rows - 1.0))
    rrse = -np.sqrt(error / denominator) * 100
    if rrse > global_best:
        # plt.imshow(Draw(net), interpolation='nearest')
        # plt.show()
        net.Flush()
        inputs = np.concatenate((np.array(vars.loc[skip_idx]), np.array([1.0])))
        net.Input(inputs)
        net.Activate()
        output = net.Output()
        # print("Network output: ", output[0])
        testrrse = np.sqrt(((target.iat[skip_idx] - output[0]) ** 2) / ((target.iat[skip_idx] - avgOutput) ** 2)) * 100
        testmae = abs(target.iat[skip_idx] - output[0])
        #if generations > 0:
            #print("[GEN ", generations, "] New Best Training Error (RRSE):", -rrse)
            #print("Training Error (MAE): ", -mae)
            #print("Test Error (RRSE): ", testrrse)
            #print("Test Error (MAE): ", testmae)
        rrse_list[skip_idx] = testrrse
        mae_list[skip_idx] = testmae

    return rrse


def evolve():

    global generations
    global global_best
    global rrse_list
    global mae_list
    global rows
    global cols

    # print("LOO Validation:")
    g = NEAT.Genome(0, cols, 0, 1, False, NEAT.ActivationFunction.LINEAR, NEAT.ActivationFunction.LINEAR, 0, params, 0)
    for test_idx in range(rows):
        pop = NEAT.Population(g, params, True, 1.0, 0)
        pop.RNG.Seed(int(time.clock() * 100))
        generations = 0
        global_best = -99999999
        no_improvement = 0
        while generations < 10:
            # if (generations > 20 and global_best < -100):
            #     pop = NEAT.Population(g, params, True, 1.0, 0)
            #     pop.RNG.Seed(int(time.clock() * 100))
            #     generations = 0
            #     global_best = -99999999
            #     no_improvement = 0
            genome_list = NEAT.GetGenomeList(pop)
            fitness_list = []
            for genome in genome_list:
                fitness_list.append(evaluate(genome, test_idx))
            NEAT.ZipFitness(genome_list, fitness_list)
            pop.Epoch()
            generations += 1
            best = max(fitness_list)
            print("[", test_idx, "] ", -global_best, " (", no_improvement, ")")
            if best > global_best:
                no_improvement = 0
                global_best = best
            else:
                no_improvement += 1

        # print("LOO test error (RRSE):")
        # print(rrse_list[test_idx])
        # print("LOO test error (MAE):")
        # print(mae_list[test_idx])

    print(rrse_list)
    print(mae_list)
    avg_rrse = np.mean(rrse_list)
    avg_mae = np.mean(mae_list)
    return [avg_rrse, avg_mae]


avg_score = 0
avg_mae = 0
nruns = 2
for run in range(nruns):
    score = evolve()
    avg_score += score[0]
    avg_mae += score[1]
    print("Run [", run, "] Test RRSE: ", score[0])
    print("Run [", run, "] Test MAE: ", score[1])

avg_score /= nruns
avg_mae /= nruns
print("Average RRSE from all runs: ", avg_score)
print("Average MAE from all runs: ", avg_mae)
