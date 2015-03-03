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
#include <iostream>

namespace hn {
	class FunctionApproximator {
	private:
		struct Node {
			std::vector<float> _weights;

			float _bias;
		};

		std::vector<std::vector<Node>> _hiddenLayers;
		std::vector<Node> _outputLayer;

		float crossoverChooseWeight(float w1, float w2, float averageChance, std::mt19937 &generator);

	public:
		void createRandom(size_t numInputs, size_t numOutputs, size_t numHiddenLayers, size_t numNeuronsPerHiddenLayer, float minWeight, float maxWeight, std::mt19937 &generator);
		void createFromParents(const FunctionApproximator &parent1, const FunctionApproximator &parent2, float averageChance, std::mt19937 &generator);
		void mutate(float perturbationChance, float perturbationStdDev, std::mt19937 &generator);

		// Returns last index of weight vector
		size_t createFromWeightsVector(size_t numInputs, size_t numOutputs, size_t numHiddenLayers, size_t numNeuronsPerHiddenLayer, const std::vector<float> &weights, size_t startIndex = 0);
		void getWeightsVector(std::vector<float> &weights);

		void process(const std::vector<float> &inputs, std::vector<float> &outputs, float activationMultiplier);
		void process(const std::vector<float> &inputs, std::vector<std::vector<float>> &layerOutputs, float activationMultiplier);
		void backpropagate(const std::vector<float> &inputs, const std::vector<std::vector<float>> &layerOutputs, const std::vector<float> &targetOutputs, float alpha);
		void getInputError(const std::vector<float> &inputs, const std::vector<std::vector<float>> &layerOutputs, const std::vector<float> &targetOutputs, std::vector<float> &inputErrors);

		void writeToStream(std::ostream &os) const;
		void readFromStream(std::istream &is);

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