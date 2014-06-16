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

#include <nn/FeedForwardNeuralNetwork.h>

#include <random>
#include <memory>

namespace nn {
	class GeneticAlgorithm {
	private:
		std::vector<FeedForwardNeuralNetwork> _population;
		std::vector<float> _fitnesses;

		std::mt19937 _generator;

		// Rescale to all positive fitnesses
		void rescaleFitnesses();

		size_t rouletteWheel(float totalFitness);

	public:
		// Parameters used to create neural networks
		struct NetworkDesc {
			size_t _numInputs;
			size_t _numOutputs;
			size_t _numHiddenLayers;
			size_t _numNeuronsPerHiddenLayer;

			float _activationMultiplier; // Sensitivity of neurons
			float _outputTraceDecay; // Decay rate of output traces
			float _weightTraceDecay; // Decay rate of weight traces
			float _minWeight;
			float _maxWeight;

			NetworkDesc();
		};

		float _weightMutationChance; // Chance that a weight will be perturbed
		float _maxWeightPerturbation; // Maximum perturbation a weight can incur
		float _averageWeightChance; // Chance that weights will be averaged during crossover

		float _greedExponent; // Higher values make search greedier

		GeneticAlgorithm();

		void create(size_t populationSize, const NetworkDesc &desc, unsigned long seed);

		// Creates a new generation based on set fitnesses
		void generation();

		size_t getPopulationSize() const {
			return _population.size();
		}

		float getFitness(size_t i) const {
			return _fitnesses[i];
		}

		void setFitness(size_t i, float value) {
			_fitnesses[i] = value;
		}

		const FeedForwardNeuralNetwork &getPopulationMember(size_t i) const {
			return _population[i];
		}
	};
}

