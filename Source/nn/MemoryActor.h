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
#include <list>

namespace nn {
	class MemoryActor {
	private:
		std::vector<MemoryCell> _memoryCells;

	public:
		FeedForwardNeuralNetwork _actor;

		// Create an agent with randomly initialized weights
		void createRandom(size_t numInputs, size_t numOutputs,
			size_t numHiddenLayers, size_t numNeuronsPerHiddenLayer,
			size_t numMemoryCells, float minWeight, float maxWeight, unsigned long seed);

		// Steps agent to a new state in the POMDP
		void step(float error);

		void writeToStream(std::ostream &stream);
		void readFromStream(std::istream &stream);

		// Get a (previously set) input
		float getInput(size_t i) const {
			return _actor.getInput(i);
		}

		// Set an input to the agent
		void setInput(size_t i, float value) {
			_actor.setInput(i, value);
		}

		// Get the output action of the agent
		float getOutput(size_t i) const {
			return _actor.getOutput(i);
		}

		size_t getNumInputs() const {
			return _actor.getNumInputs() - _memoryCells.size();
		}

		size_t getNumOutputs() const {
			return _actor.getNumOutputs() - _memoryCells.size() * 4;
		}
	};
}