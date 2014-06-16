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

#pragma once

#include <hypernet/HyperNet.h>

namespace hn {
	class EvolutionaryAlgorithm {
	private:
		struct FitnessAndIndex {
			float _fitness;
			size_t _index;
		};

		std::vector<std::shared_ptr<HyperNet>> _population;
		std::vector<float> _fitnesses;

		size_t rouletteWheel(float fitnessSum, std::mt19937 &generator);

	public:
		float _greedExponent;
		float _weightAverageChance;
		float _memoryAverageChance;
		float _weightPerturbationChance;
		float _weightPerturbationStdDev;
		float _memoryPerturbationChance;
		float _memoryPerturbationStdDev;

		size_t _numElites;

		EvolutionaryAlgorithm();

		void create(const Config &config, size_t populationSize, int preTrainIterations, float preTrainAlpha, float preTrainMin, float preTrainMax, std::mt19937 &generator, float activationMultiplier);

		void generation(const Config &config, std::mt19937 &generator);

		size_t getPopulationSize() const {
			return _population.size();
		}

		float getFitness(size_t index) const {
			return _fitnesses[index];
		}

		void setFitness(size_t index, float value) {
			_fitnesses[index] = value;
		}

		std::shared_ptr<HyperNet> getHyperNet(size_t index) const {
			return _population[index];
		}
	};
}