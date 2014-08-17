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

#include <vector>
#include <random>

namespace raahn {
	class HebbianLearner {
	private:
		struct Synapse {
			float _weight;
			float _trace;
		};

		struct Node {
			std::vector<Synapse> _weights;

			float _output;
			float _outputTrace;

			Synapse _bias;
		};

		std::vector<std::vector<Node>> _hiddenLayers;
		std::vector<Node> _outputLayer;

		float crossoverChooseWeight(float w1, float w2, float averageChance, std::mt19937 &generator);

	public:
		void createRandom(size_t numInputs, size_t numOutputs, size_t numHiddenLayers, size_t numNeuronsPerHiddenLayer, float minWeight, float maxWeight, std::mt19937 &generator);
		void createFromParents(const HebbianLearner &parent1, const HebbianLearner &parent2, float averageChance, std::mt19937 &generator);
		void mutate(float perturbationChance, float perturbationStdDev, std::mt19937 &generator);

		void process(const std::vector<float> &inputs, std::vector<float> &outputs, float modulation, float traceDecay, float outputDecay, float breakRate, std::mt19937 &generator);
		void process(const std::vector<float> &inputs, std::vector<float> &outputs, float activationMultiplier, float modulation, float traceDecay, float outputDecay, float breakRate, std::mt19937 &generator);

		static float sigmoid(float x) {
			return 1.0f / (1.0f + std::exp(-x));
		}

		size_t getNumInputs() const {
			if (_hiddenLayers.empty())
				return _outputLayer[0]._weights.size();

			return _hiddenLayers[0][0]._weights.size();
		}

		size_t getNumOutputs() const {
			return _outputLayer.size();
		}

		size_t getNumHiddenLayers() const {
			return _hiddenLayers.size();
		}

		size_t getNumNeuronsPerHiddenLayer() const {
			if (_hiddenLayers.empty())
				return 0;

			return _hiddenLayers[0].size();
		}
	};
}