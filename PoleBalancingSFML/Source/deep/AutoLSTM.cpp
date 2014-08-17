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

#include <deep/AutoLSTM.h>

#include <assert.h>

using namespace deep;

void AutoLSTM::createRandom(int numInputs, int numOutputs, int numFeatures, int numAutoEncoders, int numMemoryLayers, int memoryLayerSize, int numHiddenLayers, int hiddenLayerSize, float minWeight, float maxWeight, std::mt19937 &randomGenerator) {
	_inputs.resize(numInputs);

	_lstm.createRandomLayered(numFeatures, numOutputs, numMemoryLayers, memoryLayerSize, numHiddenLayers, hiddenLayerSize, minWeight, maxWeight, randomGenerator);

	_autoEncoderStack.resize(numAutoEncoders);

	assert(numInputs >= numFeatures);

	int fDiff = numInputs - numFeatures;

	int prevInputs = numInputs;

	for (int i = 0; i < numAutoEncoders; i++) {
		int numOutputs = numFeatures + fDiff * (static_cast<float>(i) / static_cast<float>(numAutoEncoders));
		_autoEncoderStack[i].createRandom(prevInputs, numOutputs, minWeight, maxWeight, randomGenerator);
		prevInputs = numOutputs;
	}
}

void AutoLSTM::update(float autoEncoderAlpha) {
	std::vector<float> autoEncoderInputs = _inputs;

	for (size_t i = 0; i < _autoEncoderStack.size(); i++) {
		std::vector<float> autoEncoderOutputs(_autoEncoderStack[i].getNumOutputs());

		_autoEncoderStack[i].update(autoEncoderInputs, autoEncoderOutputs, autoEncoderAlpha);

		autoEncoderInputs = autoEncoderOutputs;
	}

	for (size_t i = 0; i < autoEncoderInputs.size(); i++)
		_lstm.setInput(i, autoEncoderInputs[i]);

	_lstm.step(true);
}

void AutoLSTM::backpropagate(const std::vector<float> &outputs, float lstmAlpha, float eligibiltyDecay) {
	_lstm.getDeltas(outputs, eligibiltyDecay, true);

	_lstm.moveAlongDeltas(lstmAlpha);
}