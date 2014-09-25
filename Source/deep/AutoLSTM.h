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

#include <lstm/LSTMG.h>
#include <deep/DAutoEncoder.h>
#include <nn/FeedForwardNeuralNetwork.h>

namespace deep {
	class AutoLSTM {
	private:
		std::vector<DAutoEncoder> _autoEncoderStack;

		lstm::LSTMG _lstm;

		std::vector<float> _inputs;

	public:
		void createRandom(int numInputs, int numOutputs, int numFeatures, int numAutoEncoders, int numMemoryLayers, int memoryLayerSize, int numHiddenLayers, int hiddenLayerSize, float minWeight, float maxWeight, std::mt19937 &randomGenerator);

		void update(float autoEncoderAlpha);
		void backpropagate(const std::vector<float> &outputs, float lstmAlpha, float eligibiltyDecay);

		void setInput(int index, float value) {
			_inputs[index] = value;
		}

		float getInput(int index) const {
			return _inputs[index];
		}

		float getOutput(int index) const {
			return _lstm.getOutput(index);
		}

		size_t getNumInputs() const {
			return _inputs.size();
		}

		size_t getNumOutputs() const {
			return _lstm.getNumOutputs();
		}
	};
}