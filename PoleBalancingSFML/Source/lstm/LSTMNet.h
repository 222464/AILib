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

namespace lstm {
	class LSTMNet {
	public:
		struct Synapse {
			float _weight;
			float _trace;
		};

		struct MemorySynapse {
			float _weight;
			float _trace;
			float _derivative;
		};

		struct Node {
			std::vector<Synapse> _synapses;
			Synapse _bias;

			float _prevOutput;
			float _output;
		};

		struct MemoryNode {
			std::vector<MemorySynapse> _synapses;
			MemorySynapse _bias;

			float _prevOutput;
			float _output;
		};

		struct MemoryCell {
			MemoryNode _inputGate;
			Node _outputGate;
			MemoryNode _forgetGate;

			std::vector<MemorySynapse> _synapses;
			MemorySynapse _bias;

			float _net;
			float _state;
			float _prevOutput;
			float _output;
		};

		static float sigmoid(float x) {
			return 1.0f / (1.0f + std::expf(-x));
		}

	private:
		std::vector<Node> _outputNodes;
		std::vector<std::vector<Node>> _hiddenGroups;
		std::vector<MemoryCell> _memoryCells;

		std::vector<float> _currentInputs;

	public:
		void createRandom(size_t numInputs, size_t numOutputs, size_t hiddenSize, size_t numMemoryCells, float minWeight, float maxWeight, std::mt19937 &generator);

		void step();
		void updateQ(const std::vector<float> &targets, float error, float gammaLambda);
		void stepReinforce(float error, float gammaLambda, float offsetStdDev, std::mt19937 &generator);

		void setInput(size_t index, float value) {
			_currentInputs[index] = value;
		}

		float getInput(size_t index) const {
			return _currentInputs[index];
		}

		float getOutput(size_t index) const {
			return _outputNodes[index]._output;
		}

		size_t getNumInputs() const {
			return _currentInputs.size();
		}

		size_t getNumOutputs() const {
			return _outputNodes.size();
		}

		size_t getHiddenSize() {
			return _hiddenGroups[0].size();
		}
	};
}