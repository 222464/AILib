#include "GeneticAlgorithm.h"

#include <numeric>
#include <algorithm>

using namespace ctrnn;

GeneticAlgorithm::GeneticAlgorithm()
{}

void GeneticAlgorithm::create(size_t populationSize,
	size_t numInputs, size_t numOutputs, size_t numHidden,
	float minWeight, float maxWeight, float minTau, float maxTau, float minNoise, float maxNoise,
	std::mt19937 &generator)
{
	std::uniform_int_distribution<int> distribution;

	_population.resize(populationSize);
	_fitnesses.assign(populationSize, 0.0f);

	for (CTRNN &network : _population) {
		network.createRandom(numInputs, numOutputs,
			numHidden,	minWeight, maxWeight, minTau, maxTau, minNoise, maxNoise,
			generator);
	}
}

void GeneticAlgorithm::rescaleFitnesses(float greedExponent) {
	float minFitness = *std::min_element(_fitnesses.begin(), _fitnesses.end());

	for (float &fitness : _fitnesses) {
		fitness -= minFitness;
		fitness = std::pow(fitness, greedExponent);
	}
}

size_t GeneticAlgorithm::rouletteWheel(float totalFitness, std::mt19937 &generator) {
	std::uniform_real_distribution<float> distribution(0.0f, totalFitness);

	float randomCusp = distribution(generator);

	float sumSoFar = 0.0f;

	for (size_t i = 0; i < _fitnesses.size(); i++) {
		sumSoFar += _fitnesses[i];

		if (sumSoFar > randomCusp)
			return i;
	}

	return 0;
}

void GeneticAlgorithm::generation(float weightPerturbationChance, float maxWeightPerturbation, float averageWeightsChance,
	float tauPerturbationChance, float maxTauPerturbation, float averageTausChance,
	float noiseStdDevPerturbationChance, float maxNoiseStdDevPerturbation, float averageNoiseStdDevChance,
	float greedExponent, size_t numElites,
	std::mt19937 &generator)
{
	rescaleFitnesses(greedExponent);

	std::vector<CTRNN> newPopulation(getPopulationSize());

	// Find elites by sorting fitnesses
	struct FitnessAndIndex {
		float _fitness;
		size_t _index;

		bool operator<(const FitnessAndIndex &other) {
			return _fitness > other._fitness;
		}
	};

	std::vector<FitnessAndIndex> fitnessesAndIndices(getPopulationSize());

	for (size_t i = 0; i < fitnessesAndIndices.size(); i++) {
		fitnessesAndIndices[i]._fitness = _fitnesses[i];
		fitnessesAndIndices[i]._index = i;
	}

	std::sort(fitnessesAndIndices.begin(), fitnessesAndIndices.end());

	for (size_t i = 0; i < numElites; i++)
		newPopulation[i] = _population[fitnessesAndIndices[i]._index];

	float totalFitness = std::accumulate(_fitnesses.begin(), _fitnesses.end(), 0.0f);

	for (size_t i = numElites; i < getPopulationSize(); i++) {
		size_t parent1Index = rouletteWheel(totalFitness, generator);
		size_t parent2Index = rouletteWheel(totalFitness, generator);

		newPopulation[i].createFromParents(_population[parent1Index], _population[parent2Index], averageWeightsChance, averageTausChance, averageNoiseStdDevChance, generator);
		newPopulation[i].mutate(weightPerturbationChance, maxWeightPerturbation, tauPerturbationChance, maxTauPerturbation, noiseStdDevPerturbationChance, maxNoiseStdDevPerturbation, generator);
	}

	// Set new population as current population
	_population = newPopulation;
}