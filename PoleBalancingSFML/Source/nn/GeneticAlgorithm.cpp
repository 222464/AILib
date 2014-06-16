/*
AI Lib
Copyright (C) 2014 Eric Laukien

This software is provided 'as-is', without any express or implied
warranty.  In no event will the authors be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not
claim that you wrote the original software. If you use this software
in a product, an acknowledgment in the product documentation would be
appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be
misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

#include <nn/GeneticAlgorithm.h>

#include <numeric>
#include <algorithm>

using namespace nn;

// Default values
GeneticAlgorithm::NetworkDesc::NetworkDesc()
: _numInputs(2),
_numOutputs(1),
_numHiddenLayers(1),
_numNeuronsPerHiddenLayer(2),
_activationMultiplier(3.0f),
_outputTraceDecay(0.125f),
_weightTraceDecay(0.125f),
_minWeight(-1.0f),
_maxWeight(1.0f)
{}

// Default values
GeneticAlgorithm::GeneticAlgorithm()
: _weightMutationChance(0.125f),
_maxWeightPerturbation(0.0625f),
_averageWeightChance(0.5f),
_greedExponent(2.0f)
{}

void GeneticAlgorithm::create(size_t populationSize, const NetworkDesc &desc, unsigned long seed) {
	_generator.seed(seed);

	std::uniform_int_distribution<int> distribution;

	_population.resize(populationSize);
	_fitnesses.assign(populationSize, 0.0f);

	for (FeedForwardNeuralNetwork &network : _population) {
		network.createRandom(desc._numInputs, desc._numOutputs,
			desc._numHiddenLayers, desc._numNeuronsPerHiddenLayer,
			desc._minWeight, desc._maxWeight,
			distribution(_generator));

		network._activationMultiplier = desc._activationMultiplier;
		network._outputTraceDecay = desc._outputTraceDecay;
		network._weightTraceDecay = desc._weightTraceDecay;
	}
}

void GeneticAlgorithm::rescaleFitnesses() {
	float minFitness = *std::min_element(_fitnesses.begin(), _fitnesses.end());

	for (float &fitness : _fitnesses) {
		fitness -= minFitness;
		fitness = std::powf(fitness, _greedExponent);
	}
}

size_t GeneticAlgorithm::rouletteWheel(float totalFitness) {
	std::uniform_real_distribution<float> distribution(0.0f, totalFitness);

	float randomCusp = distribution(_generator);

	float sumSoFar = 0.0f;

	for (size_t i = 0; i < _fitnesses.size(); i++) {
		sumSoFar += _fitnesses[i];

		if (sumSoFar > randomCusp)
			return i;
	}

	return 0;
}

void GeneticAlgorithm::generation() {
	rescaleFitnesses();

	std::vector<FeedForwardNeuralNetwork> newPopulation;

	newPopulation.resize(getPopulationSize());

	float totalFitness = std::accumulate(_fitnesses.begin(), _fitnesses.end(), 0.0f);

	std::uniform_int_distribution<unsigned long> distributionInt;

	for (size_t i = 0; i < getPopulationSize(); i++) {
		size_t parent1Index = rouletteWheel(totalFitness);
		size_t parent2Index = rouletteWheel(totalFitness);

		newPopulation[i].createFromParents(_population[parent1Index], _population[parent2Index], _averageWeightChance, distributionInt(_generator));
		newPopulation[i].mutate(_weightMutationChance, _maxWeightPerturbation, distributionInt(_generator));
	}

	// Set new population as current population
	_population = newPopulation;
}