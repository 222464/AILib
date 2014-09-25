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

namespace deep {
	class FA {
	private:
		struct Connection {
			float _weight;
			float _prevDWeight;
			float _eligibility;

			Connection()
				: _eligibility(0.0f), _prevDWeight(0.0f)
			{}
		};

		struct Node {
			std::vector<Connection> _connections;

			Connection _bias;

			float _output;
			float _error;

			Node()
				: _output(0.0f), _error(0.0f)
			{}
		};

		std::vector<std::vector<Node>> _hiddenLayers;
		std::vector<Node> _outputLayer;

		float crossoverChooseWeight(float w1, float w2, float averageChance, std::mt19937 &generator);

	public:
		void createRandom(int numInputs, int numOutputs, int numHiddenLayers, int numNeuronsPerHiddenLayer, float weightStdDev, std::mt19937 &generator);
		void createFromParents(const FA &parent1, const FA &parent2, float averageChance, std::mt19937 &generator);
		void mutate(float perturbationChance, float perturbationStdDev, std::mt19937 &generator);

		// Returns last index of weight vector
		int createFromWeightsVector(int numInputs, int numOutputs, int numHiddenLayers, int numNeuronsPerHiddenLayer, const std::vector<float> &weights, int startIndex = 0);
		void getWeightsVector(std::vector<float> &weights);

		void process(const std::vector<float> &inputs, std::vector<float> &outputs);

		void backpropagate(const std::vector<float> &inputs, const std::vector<float> &targetOutputs, float alpha, float momentum);

		void adapt(const std::vector<float> &inputs, const std::vector<float> &targetOutputs, float alpha, float error, float eligibilityDecay, float momentum);

		void decayWeights(float decayMultiplier);

		void writeToStream(std::ostream &os) const;
		void readFromStream(std::istream &is);

		static float sigmoid(float x) {
			return 1.0f / (1.0f + std::exp(-x));
		}

		int getNumInputs() const {
			if (_hiddenLayers.empty())
				return _outputLayer[0]._connections.size();

			return _hiddenLayers[0][0]._connections.size();
		}

		int getNumOutputs() const {
			return _outputLayer.size();
		}

		int getNumHiddenLayers() const {
			return _hiddenLayers.size();
		}

		int getNumNeuronsPerHiddenLayer() const {
			if (_hiddenLayers.empty())
				return 0;

			return _hiddenLayers[0].size();
		}
	};
}