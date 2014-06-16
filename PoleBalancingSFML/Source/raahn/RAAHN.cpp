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

#include <raahn/RAAHN.h>

#include <algorithm>
#include <iostream>

using namespace raahn;

void RAAHN::createRandom(size_t numInputs, size_t numFeatures, size_t numOutputs,
	size_t numRecurrentConnections, size_t numHebbianHidden, size_t numNeuronsPerHebbianHidden,
	float minWeight, float maxWeight, std::mt19937 &generator)
{
	_numOutputs = numOutputs;
	_inputs.assign(numInputs, 0.0f);
	_features.assign(numFeatures, 0.0f);
	_outputs.resize(numOutputs + numRecurrentConnections);
	_hebbianInputs.assign(numFeatures + numRecurrentConnections, 0.0f);

	std::uniform_real_distribution<float> distWeight(minWeight, maxWeight);

	for (size_t n = 0; n < _outputs.size(); n++) {
		_outputs[n]._bias._weight = distWeight(generator);
		_outputs[n]._bias._trace = 0.0f;

		_outputs[n]._output = 0.0f;

		_outputs[n]._weights.resize(_hebbianInputs.size());

		for (size_t w = 0; w < _hebbianInputs.size(); w++) {
			_outputs[n]._weights[w]._weight = distWeight(generator);
			_outputs[n]._weights[w]._trace = 0.0f;
		}
	}

	_autoEncoder.createRandom(numInputs, numFeatures, minWeight, maxWeight, generator);
}

void RAAHN::update(float autoEncoderAlpha, float modulation, float traceDecay, float breakRate, std::mt19937 &generator) {
	_autoEncoder.update(_inputs, _features, autoEncoderAlpha);

	size_t hebbianInputIndex = 0;

	// Add features to inputs
	for (size_t i = 0; i < _features.size(); i++)
		_hebbianInputs[hebbianInputIndex++] = _features[i];

	// Add recurrent outputs to inputs
	for (size_t i = _numOutputs; i < _outputs.size(); i++)
		_hebbianInputs[hebbianInputIndex++] = _outputs[i]._output;

	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

	for (size_t n = 0; n < _outputs.size(); n++) {
		_outputs[n]._bias._weight += modulation * _outputs[n]._bias._trace;

		float sum = _outputs[n]._bias._weight;

		for (size_t w = 0; w < _outputs[n]._weights.size(); w++) {
			_outputs[n]._weights[w]._weight += modulation * _outputs[n]._weights[w]._trace;
			std::cout << _outputs[n]._weights[w]._weight << std::endl;
			sum += _outputs[n]._weights[w]._weight * _hebbianInputs[w];
		}

		_outputs[n]._output = (dist01(generator) < breakRate) ? (dist01(generator) * 2.0f) - 1.0f : std::min(1.0f, std::max(-1.0f, sum));

		_outputs[n]._bias._trace += -traceDecay * _outputs[n]._bias._trace + _outputs[n]._output;

		for (size_t w = 0; w < _outputs[n]._weights.size(); w++)
			_outputs[n]._weights[w]._trace += -traceDecay * _outputs[n]._weights[w]._trace + _outputs[n]._output * _hebbianInputs[w];
	}
}