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
	_hebbianInputs.assign(numFeatures + numRecurrentConnections, 0.0f);
	_outputs.resize(numOutputs + numRecurrentConnections);

	_autoEncoder.createRandom(numInputs, numFeatures, minWeight, maxWeight, generator);
	_hebbianLearner.createRandom(numFeatures + numRecurrentConnections, numOutputs + numRecurrentConnections, numHebbianHidden, numNeuronsPerHebbianHidden, minWeight, maxWeight, generator);
}

void RAAHN::createFromParents(const RAAHN &parent1, const RAAHN &parent2, float averageChance, std::mt19937 &generator) {
	_numOutputs = parent1._numOutputs;
	_inputs.assign(parent1._inputs.size(), 0.0f);
	_features.assign(parent1._features.size(), 0.0f);
	_hebbianInputs.assign(parent1._hebbianInputs.size(), 0.0f);
	_outputs.resize(parent1._outputs.size());

	_autoEncoder.createFromParents(parent1._autoEncoder, parent2._autoEncoder, averageChance, generator);
	_hebbianLearner.createFromParents(parent1._hebbianLearner, parent2._hebbianLearner, averageChance, generator);
}

void RAAHN::mutate(float perturbationChance, float perturbationStdDev, std::mt19937 &generator) {
	_autoEncoder.mutate(perturbationChance, perturbationStdDev, generator);
	_hebbianLearner.mutate(perturbationChance, perturbationStdDev, generator);
}

void RAAHN::update(float autoEncoderAlpha, float modulation, float traceDecay, float outputDecay, float breakRate, std::mt19937 &generator) {
	_autoEncoder.update(_inputs, _features, autoEncoderAlpha);

	size_t hebbianInputIndex = 0;

	// Add features to inputs
	for (size_t i = 0; i < _features.size(); i++)
		_hebbianInputs[hebbianInputIndex++] = _features[i];

	// Add recurrent outputs to inputs
	for (size_t i = _numOutputs; i < _outputs.size(); i++)
		_hebbianInputs[hebbianInputIndex++] = _outputs[i];

	_hebbianLearner.process(_hebbianInputs, _outputs, modulation, traceDecay, outputDecay, breakRate, generator);
}