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

#include <hypernet/EvolutionaryAlgorithm.h>

#include <algorithm>

using namespace hn;

EvolutionaryAlgorithm::EvolutionaryAlgorithm()
: _greedExponent(4.0f),
_numElites(4),
_weightAverageChance(0.2f),
_memoryAverageChance(0.2f),
_weightPerturbationChance(0.3f),
_weightPerturbationStdDev(0.2f),
_memoryPerturbationChance(0.3f),
_memoryPerturbationStdDev(0.2f)
{}

size_t EvolutionaryAlgorithm::rouletteWheel(float fitnessSum, std::mt19937 &generator) {
	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
	
	float randomCusp = dist01(generator) * fitnessSum;

	float sumSoFar = 0.0f;

	for (size_t i = 0; i < _fitnesses.size(); i++) {
		sumSoFar += _fitnesses[i];

		if (sumSoFar >= randomCusp)
			return i;
	}

	return 0;
}

void EvolutionaryAlgorithm::create(const Config &config, size_t populationSize, int preTrainIterations, float preTrainAlpha, float preTrainMin, float preTrainMax, std::mt19937 &generator, float activationMultiplier) {
	_population.resize(populationSize);
	_fitnesses.resize(populationSize);

	for (size_t i = 0; i < populationSize; i++) {
		_population[i].reset(new HyperNet());

		_population[i]->createRandom(config, preTrainIterations, preTrainAlpha, preTrainMin, preTrainMax, generator, activationMultiplier);

		_fitnesses[i] = 0.0f;
	}
}

void EvolutionaryAlgorithm::generation(const Config &config, std::mt19937 &generator) {
	// Normalize fitnesses
	float minimum = _fitnesses[0];
	float maximum = _fitnesses[0];

	for (size_t i = 1; i < _fitnesses.size(); i++) {
		minimum = std::min(minimum, _fitnesses[i]);
		maximum = std::max(maximum, _fitnesses[i]);
	}

	float fitnessSum = 0.0f;

	if (maximum == minimum) {
		for (size_t i = 0; i < _fitnesses.size(); i++)
			_fitnesses[i] = 0.0f;
	}
	else {
		float rangeInv = 1.0f / (maximum - minimum);

		// Rescale fitnesses
		for (size_t i = 0; i < _fitnesses.size(); i++)
			fitnessSum += (_fitnesses[i] = std::pow((_fitnesses[i] - minimum) * rangeInv, _greedExponent));
	}

	std::vector<std::shared_ptr<HyperNet>> newPopulation(_population.size());

	std::list<FitnessAndIndex> fitnessesAndIndices;

	for (size_t i = 0; i < _population.size(); i++)
		fitnessesAndIndices.push_back(FitnessAndIndex { _fitnesses[i], i });

	// Add elites to population
	for (size_t i = 0; i < _numElites; i++) {
		// Find highest fitness
		std::list<FitnessAndIndex>::iterator highestIt = fitnessesAndIndices.begin();

		for (std::list<FitnessAndIndex>::iterator it = fitnessesAndIndices.begin(); it != fitnessesAndIndices.end(); it++)
		if (it->_fitness > highestIt->_fitness)
			highestIt = it;

		newPopulation[i] = _population[highestIt->_index];

		fitnessesAndIndices.erase(highestIt);
	}

	// Reproduce into new population (elites already done)
	for (size_t i = _numElites; i < newPopulation.size(); i++) {
		size_t parentIndex1 = rouletteWheel(fitnessSum, generator);
		size_t parentIndex2 = rouletteWheel(fitnessSum, generator);

		newPopulation[i].reset(new HyperNet());

		newPopulation[i]->createFromParents(*_population[parentIndex1], *_population[parentIndex2], _weightAverageChance, _memoryAverageChance, generator);
		newPopulation[i]->mutate(_weightPerturbationChance, _weightPerturbationStdDev, _memoryPerturbationChance, _memoryPerturbationStdDev, generator);
	}

	_population = newPopulation;
}