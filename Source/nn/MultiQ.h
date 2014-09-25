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

namespace nn {
	class MultiQ {
	private:
		nn::FeedForwardNeuralNetwork _network;

		size_t _output;

		float _prevValue;

		std::vector<float> _currentInputs;
		std::vector<float> _prevInputs;

	public:
		float _epsilon;
		float _gamma;
		float _alpha;

		std::mt19937 _generator;

		MultiQ();
		~MultiQ();

		void createRandom(size_t numInputs, size_t numOutputs, size_t numHiddenLayers, size_t numNeuronsPerHiddenLayer, float minWeight, float maxWeight, unsigned long seed);

		void setInput(size_t index, float value) {
			_currentInputs[index] = value;
		}

		void step(float reward);

		size_t getOutput() const {
			return _output;
		}
	};
}
