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
		struct Connection {
			float _weight;
			float _trace;
			int _gaterIndex;

			float _eligibility;
		};

		struct Unit {
			float _state;
			float _prevState;
			float _activation;

			float _bias;
			float _biasEligibility;
	
			std::vector<int> _ingoingConnectionIndices;
			std::vector<int> _outgoingConnectionIndices;
		};

		static float sigmoid(float x) {
			return 1.0f / (1.0f + std::expf(-x));
		}

		static float sigmoidDerivative(float x) {
			float s = sigmoid(x);

			return s * (1.0f - s);
		}

	private:
		std::vector<int> _inputIndices;
		std::vector<int> _outputIndices;

		std::vector<Unit> _units;

		std::unordered_map<std::tuple<int, int>, Connection> _connections;
		std::unordered_map<std::tuple<int, int>, int> _gaterIndices;
		std::unordered_map<int, std::vector<std::tuple<int, int>>> _reverseGaterIndices;
		std::vector<int> _orderedGaterIndices;
		std::unordered_map<std::tuple<int, int, int>, float> _extendedTraces;

		std::unordered_map<std::tuple<int, int>, float> _prevGains;
		std::unordered_map<std::tuple<int, int>, float> _prevActivations;

		float gain(int j, int i);
		float theTerm(int j, int k);

	public:
		void createRandomLayered(int numInputs, int numOutputs,
			int numMemoryLayers, int memoryLayerSize, int numHiddenLayers, int hiddenLayerSize,
			float minWeight, float maxWeight, std::mt19937 &randomGenerator);

		void step(bool linearOutput);

		void getDeltas(const std::vector<float> &targets, float eligibilityDecay, bool linearOutput);

		void moveAlongDeltas(float error);

		void setInput(int index, float value) {
			_units[_inputIndices[index]]._activation = value;
		}

		float getInput(int index) const {
			return _units[_inputIndices[index]]._activation;
		}

		float getOutput(int index) const {
			return _units[_outputIndices[index]]._activation;
		}

		void clear();
	};
}