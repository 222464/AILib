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
#include <nn/MemoryCell.h>

namespace nn {
	class RLLSTMAgent {
	private:
		std::vector<float> _prevInputs;

		std::vector<MemoryCell> _memoryCells;

		size_t _numInputs;
		size_t _numActions;

		size_t _selectedAction;

		float _prevError;

	public:
		std::mt19937 _generator;

		float _alpha;
		float _gamma;
		float _k;

		float _explorationMultiplier;

		float _expAlpha;
		float _expMomentum;
		float _expGamma;

		size_t _numExpBackpropPasses;

		FeedForwardNeuralNetwork _rnn; // Main recurrent neural network
		FeedForwardNeuralNetwork _expnn; // Exploration neural network

		RLLSTMAgent();

		void createRandom(size_t numInputs, size_t numActions,
			size_t recNumHiddenLayers, size_t recNumNeuronsPerHiddenLayer,
			size_t expNumHiddenLayers, size_t expNumNeuronsPerHiddenLayer,
			size_t numMemoryCells, float minWeight, float maxWeight, unsigned long seed);

		void step(float fitness);

		size_t getSelectedAction() const {
			return _selectedAction;
		}

		//void writeToStream(std::ostream &stream);
		//void readFromStream(std::istream &stream);

		// Get a (previously set) input
		float getInput(size_t i) const {
			return _rnn.getInput(i);
		}

		// Set an input to the agent
		void setInput(size_t i, float value) {
			_rnn.setInput(i, value);
			_expnn.setInput(i, value);
		}

		size_t getNumInputs() const {
			return _numInputs;
		}

		size_t getNumActions() const {
			return _numActions;
		}
	};
}

