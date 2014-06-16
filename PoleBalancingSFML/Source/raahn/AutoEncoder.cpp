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

#include <raahn/AutoEncoder.h>

using namespace raahn;

void AutoEncoder::createRandom(size_t numInputs, size_t numOutputs, float minWeight, float maxWeight, std::mt19937 &generator) {
	_inputBiases.resize(numInputs);
	_hidden.resize(numOutputs);
	_inputErrorBuffer.resize(numInputs);

	std::uniform_real_distribution<float> distWeight(minWeight, maxWeight);

	for (size_t n = 0; n < _inputBiases.size(); n++)
		_inputBiases[n] = distWeight(generator);

	for (size_t n = 0; n < _hidden.size(); n++) {
		_hidden[n]._bias = distWeight(generator);

		_hidden[n]._weights.resize(numInputs);

		for (size_t w = 0; w < numInputs; w++)
			_hidden[n]._weights[w] = distWeight(generator);
	}
}

void AutoEncoder::update(const std::vector<float> &inputs, std::vector<float> &outputs, float alpha) {
	if (outputs.size() != _hidden.size())
		outputs.resize(_hidden.size());

	for (size_t n = 0; n < _hidden.size(); n++) {
		float sum = _hidden[n]._bias;

		for (size_t w = 0; w < _hidden[n]._weights.size(); w++)
			sum += inputs[w] * _hidden[n]._weights[w];

		outputs[n] = sigmoid(sum);
	}

	for (size_t n = 0; n < _inputBiases.size(); n++) {
		float sum = _inputBiases[n];

		for (size_t w = 0; w < _hidden.size(); w++)
			sum += outputs[w] * _hidden[w]._weights[n];

		_inputErrorBuffer[n] = inputs[n] - sigmoid(sum);
	}

	for (size_t n = 0; n < _inputBiases.size(); n++)
		_inputBiases[n] += alpha * _inputErrorBuffer[n];

	for (size_t n = 0; n < _hidden.size(); n++) {
		float error = 0.0f;

		for (size_t w = 0; w < _hidden[n]._weights.size(); w++)
			error += _inputErrorBuffer[w] * _hidden[n]._weights[w];

		_inputBiases[n] += alpha * -error;

		for (size_t w = 0; w < _hidden[n]._weights.size(); w++)
			_hidden[n]._weights[w] += alpha * _inputErrorBuffer[w] * outputs[n];
	}
}