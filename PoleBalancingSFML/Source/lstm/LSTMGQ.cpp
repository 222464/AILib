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

#include <lstm/LSTMGQ.h>

#include <algorithm>

#include <iostream>

using namespace lstm;

void LSTMGQ::createRandom(size_t numInputs, size_t numOutputs,
	size_t numMemoryLayers, size_t numMemoryCellsPerLayer,
	size_t numHiddenLayers, size_t numNeuronsPerHiddenLayer,
	float minWeight, float maxWeight, std::mt19937 &randomGenerator)
{
	_inputs.clear();
	_inputs.assign(numInputs, 0.0f);

	_outputs.clear();
	_outputs.assign(numOutputs, 0.0f);

	_qNet.createRandomLayered(numInputs + numOutputs, 1, numMemoryLayers, numMemoryCellsPerLayer, numHiddenLayers, numNeuronsPerHiddenLayer, minWeight, maxWeight, randomGenerator);
}

void LSTMGQ::step(float reward, float gamma, float alpha, float deriveQAlpha, float eligibilityDecay, size_t policyDeriveIterations, float outputPerturbationStdDev, std::mt19937 &randomGenerator) {
	float prevQ = _qNet.getOutput(0);
	
	size_t numInputs = _inputs.size();
	size_t numOutputs = _outputs.size();

	// Derive policy
	std::vector<float> deriveOutputs(numOutputs, 0.0f);

	for (size_t s = 0; s < policyDeriveIterations; s++) {
		LSTMG derivNet = _qNet;

		for (size_t i = 0; i < numInputs; i++)
			derivNet.setInput(i, _inputs[i]);

		for (size_t i = 0; i < numOutputs; i++)
			derivNet.setInput(i + numInputs, deriveOutputs[i]);

		derivNet.step(true);

		std::vector<float> inputError;

		derivNet.getDeltas(std::vector<float>(1, 99.0f), inputError, 1.0f, true);

		for (size_t i = 0; i < numOutputs; i++) {
			deriveOutputs[i] += inputError[i] * deriveQAlpha;

			deriveOutputs[i] = std::min(1.0f, std::max(-1.0f, deriveOutputs[i]));
		}
	}

	// Perturb outputs
	std::normal_distribution<float> distPerturbation(0.0f, outputPerturbationStdDev);

	for (size_t i = 0; i < _outputs.size(); i++)
		_outputs[i] = std::min(1.0f, std::max(-1.0f, std::min(1.0f, std::max(-1.0f, deriveOutputs[i])) + distPerturbation(randomGenerator)));

	// Get Q for selected action
	LSTMG derivNet = _qNet;

	for (size_t i = 0; i < numInputs; i++)
		derivNet.setInput(i, _inputs[i]);

	for (size_t i = 0; i < numOutputs; i++)
		derivNet.setInput(i + numInputs, _outputs[i]);

	derivNet.step(true);

	float q = reward + gamma * derivNet.getOutput(0);

	float error = q - prevQ;

	_qNet.moveAlongDeltas(error * alpha);

	for (size_t i = 0; i < numInputs; i++)
		_qNet.setInput(i, _inputs[i]);

	for (size_t i = 0; i < numOutputs; i++)
		_qNet.setInput(i + numInputs, _outputs[i]);

	_qNet.step(true);

	_qNet.getDeltas(std::vector<float>(1, 99.0f), eligibilityDecay, true);
}