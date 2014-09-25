#pragma once

#include "CTRNN.h"

namespace ctrnn {
	class GeneticAlgorithm {
	private:
		std::vector<CTRNN> _population;
		std::vector<float> _fitnesses;

		// Rescale to all positive fitnesses and greedify
		void rescaleFitnesses(float greedExponent);

		size_t rouletteWheel(float totalFitness, std::mt19937 &generator);

	public:
		GeneticAlgorithm();

		void create(size_t populationSize,
			size_t numInputs, size_t numOutputs, size_t numHidden,
			float minWeight, float maxWeight, float minTau, float maxTau, float minNoise, float maxNoise,
			std::mt19937 &generator);

		// Creates a new generation based on set fitnesses
		void generation(float weightPerturbationChance, float maxWeightPerturbation, float averageWeightsChance,
			float tauPerturbationChance, float maxTauPerturbation, float averageTausChance,
			float noiseStdDevPerturbationChance, float maxNoiseStdDevPerturbation, float averageNoiseStdDevChance,
			float greedExponent, size_t numElites,
			std::mt19937 &generator);

		size_t getPopulationSize() const {
			return _population.size();
		}

		float getFitness(size_t i) const {
			return _fitnesses[i];
		}

		void setFitness(size_t i, float value) {
			_fitnesses[i] = value;
		}

		const CTRNN &getPopulationMember(size_t i) const {
			return _population[i];
		}
	};
}