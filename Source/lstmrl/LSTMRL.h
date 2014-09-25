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

namespace lstmrl {
	class LSTMRL {
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

		struct LSTMRLSettings {
			float _alpha;
			float _tauInv;
			float _gamma;
			float _lambda;
			float _epsilon;

			LSTMRLSettings()
				: _alpha(0.001f),
				_tauInv(10.0f),
				_gamma(0.95f),
				_lambda(0.9f),
				_epsilon(0.2f)
			{}
		};

		static float sigmoid(float x) {
			return 1.0f / (1.0f + std::exp(-x));
		}

	private:
		std::vector<Node> _outputNodes;
		std::vector<std::vector<Node>> _hiddenGroups;
		std::vector<MemoryCell> _memoryCells;

		std::vector<float> _currentInputs;
		std::vector<float> _prevInputs;

		float _prevValue;
		float _prevAdvantage;

		size_t _output;

		bool _prevActionExploratory;

	public:
		void createRandom(size_t numInputs, size_t numOutputs, size_t hiddenSize, size_t numMemoryCells, float minWeight, float maxWeight, std::mt19937 &generator);

		void step(float reward, const LSTMRLSettings &settings, std::mt19937 &generator);

		void setInput(size_t index, float value) {
			_currentInputs[index] = value;
		}

		size_t getOutput() const {
			return _output;
		}
	};
}