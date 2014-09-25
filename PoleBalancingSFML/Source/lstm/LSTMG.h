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

#include <unordered_map>
#include <lstm/TupleHash.h>
#include <random>

namespace lstm {
	class LSTMG {
	public:
		struct ExtendedTrace {
			float _trace;
			int _index;

			ExtendedTrace() {}
			ExtendedTrace(float trace, int index)
				: _trace(trace), _index(index)
			{}
		};

		struct Connection {
			int _inputIndex;

			float _weight;
			float _trace;
			int _gaterIndex;

			float _eligibility;
			float _prevWeight;

			float _prevGain;
			float _prevActivation;

			std::vector<ExtendedTrace> _extendedTraces;

			Connection()
				: _inputIndex(-1), _weight(0.0f), _trace(0.0f), _gaterIndex(-1),
				_eligibility(0.0f), _prevWeight(0.0f), _prevGain(0.0f),
				_prevActivation(0.0f)
			{}
		};

		struct ConnectionIndex {
			int _nodeIndex;
			int _connectionIndex;

			ConnectionIndex() {}
			ConnectionIndex(int nodeIndex, int connectionIndex)
				: _nodeIndex(nodeIndex), _connectionIndex(connectionIndex)
			{}
		};

		struct Unit {
			float _state;
			float _prevState;
			float _activation;

			float _bias;
			float _biasEligibility;
			float _prevBias;

			float _prevGain;
			float _prevActivation;

			int _recurrentConnectionIndex;

			std::vector<Connection> _ingoingConnections;
			std::vector<ConnectionIndex> _outgoingConnectionIndices;

			std::vector<int> _gatingConnections;

			Unit()
				: _state(0.0f), _prevState(0.0f), _activation(0.0f), _bias(0.0f),
				_biasEligibility(0.0f), _prevBias(0.0f), _prevGain(0.0f), _prevActivation(0.0f),
				_recurrentConnectionIndex(-1)
			{}
		};

		static float sigmoid(float x) {
			return 1.0f / (1.0f + std::exp(-x));
		}

		static float sigmoidDerivative(float x) {
			float s = sigmoid(x);

			return s * (1.0f - s);
		}

	private:
		std::vector<int> _inputIndices;
		std::vector<int> _outputIndices;

		std::vector<Unit> _units;

		std::vector<int> _orderedGaterIndices;

		float gain(int j, int ci);
		float theTerm(int j, int ci);

		bool connectionExists(int j, int i);

	public:
		void createRandomLayered(int numInputs, int numOutputs,
			int numMemoryLayers, int memoryLayerSize, int numHiddenLayers, int hiddenLayerSize,
			float minWeight, float maxWeight, std::mt19937 &randomGenerator);

		void createFromParents(const LSTMG &parent0, const LSTMG &parent1, float averageChance, std::mt19937 &generator);
		void mutate(float perturbationChance, float maxPerturbation, float changeFunctionChance, std::mt19937 &generator);

		void step(bool linearOutput);

		void getDeltas(const std::vector<float> &targets, float eligibilityDecay, bool linearOutput);

		void moveAlongDeltas(float error, float momentum = 0.0f);
		void moveAlongDeltasAndHebbian(float error, float hebbianAlpha, float momentum = 0.0f);

		void setInput(int index, float value) {
			_units[_inputIndices[index]]._activation = value;
		}

		float getInput(int index) const {
			return _units[_inputIndices[index]]._activation;
		}

		float getOutput(int index) const {
			return _units[_outputIndices[index]]._activation;
		}

		int getNumInputs() const {
			return _inputIndices.size();
		}

		int getNumOutputs() const {
			return _outputIndices.size();
		}

		void clear();
	};
}