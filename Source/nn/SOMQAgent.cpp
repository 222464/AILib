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

#include <nn/SOMQAgent.h>

#include <iostream>

using namespace nn;

SOMQAgent::SOMQAgent()
: _prevQ(0.0f)
{}

void SOMQAgent::createRandom(size_t numInputs, size_t numOutputs, size_t dimensions, size_t dimensionSize, const nn::BrownianPerturbation &perturbation, float minWeight, float maxWeight, std::mt19937 &generator) {
	_numInputs = numInputs;
	_numOutputs = numOutputs;

	_stateActionSOM.createRandom(_numInputs + _numOutputs + 2, dimensions, dimensionSize, minWeight, maxWeight, generator);

	_input.assign(_numInputs, 0.0f);
	_output.assign(_numOutputs, 0.0f);

	_prevInput.assign(_numInputs, 0.0f);
	_prevOutput.assign(_numOutputs, 0.0f);
	_prevExploratoryOutput.assign(_numOutputs, 0.0f);

	_stateOnlyMask.resize(_stateActionSOM.getNumInputs());

	for (size_t i = 0; i < _numInputs; i++)
		_stateOnlyMask[i] = true;

	for (size_t i = _numInputs; i < _stateActionSOM.getNumInputs(); i++)
		_stateOnlyMask[i] = false;

	_stateActionMask.resize(_stateActionSOM.getNumInputs());

	for (size_t i = 0; i < _numInputs + _numOutputs; i++)
		_stateActionMask[i] = true;

	for (size_t i = _numInputs + _numOutputs; i < _stateActionSOM.getNumInputs(); i++)
		_stateActionMask[i] = false;

	_stateRewardMask = _stateOnlyMask;

	_stateRewardMask[_stateActionSOM.getNumInputs() - 2] = true;

	_stateActionRewardMask = _stateActionMask;

	_stateActionRewardMask[_stateActionSOM.getNumInputs() - 2] = true;

	for (size_t i = 0; i < _stateActionSOM.getNumNodes(); i++)
		_stateActionSOM.getNode(i)._weights[_stateActionSOM.getNumInputs() - 1] = 0.0f;
}

void SOMQAgent::step(float fitness, float alpha, float gamma, float traceDecay, float breakRate, float dt, std::mt19937 &generator) {
	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

	std::vector<float> initialVector(_stateActionSOM.getNumInputs());
	
	for (size_t i = 0; i < _numInputs; i++)
		initialVector[i] = _input[i];

	for (size_t i = _numInputs; i < _numInputs + _numOutputs; i++)
		initialVector[i] = _output[i];

	initialVector[_numInputs + _numOutputs] = 0.0f;
	initialVector[_numInputs + _numOutputs + 1] = 0.0f;

	// Identify unit in state map closest to input
	SOM::SOMCoords closestState = _stateActionSOM.getBestMatchingUnit(initialVector, _stateActionMask);

	float thisQ = _stateActionSOM.getNode(closestState)._weights[_numInputs + _numOutputs];

	float newQ = fitness + gamma * thisQ;

	std::cout << newQ << std::endl;

	float error = newQ - _prevQ;

	float originalPrevQ = _prevQ;

	_prevQ = thisQ;

	std::vector<float> updateVector(_numInputs + _numOutputs + 2);

	for (size_t i = 0; i < _numInputs; i++)
		updateVector[i] = _prevInput[i];

	if (error > 0.0f) {
		// Go towards exploratory output
		for (size_t i = 0; i < _numOutputs; i++)
			updateVector[i + _numInputs] = _prevExploratoryOutput[i];
	}
	else {
		// Stick with previous output without exploratory modification
		for (size_t i = 0; i < _numOutputs; i++)
			updateVector[i + _numInputs] = _prevOutput[i];
	}

	updateVector[_numInputs + _numOutputs] = newQ;
	updateVector[_numInputs + _numOutputs + 1] = 1.0f;

	// Update traces and associated Q values
	for (size_t i = 0; i < _stateActionSOM.getNumNodes(); i++) {
		_stateActionSOM.getNode(i)._weights[_numInputs + _numOutputs] += alpha * error * _stateActionSOM.getNode(i)._weights[_numInputs + _numOutputs + 1];

		_stateActionSOM.getNode(i)._weights[_numInputs + _numOutputs + 1] *= traceDecay;
	}

	SOM::SOMCoords closestForUpdate = _stateActionSOM.getBestMatchingUnit(updateVector, _stateActionRewardMask);

	_stateActionSOM.updateNeighborhood(closestForUpdate, updateVector);

	//_stateActionSOM.getNode(closestForUpdate)._weights[_stateActionSOM.getNumInputs() - 1] = 1.0f;

	// Select action
	for (size_t i = 0; i < _numOutputs; i++) {
		if (dist01(generator) < breakRate)
			_output[i] = dist01(generator) * 2.0f - 1.0f;
		else
			_output[i] = _stateActionSOM.getNode(closestState)._weights[_numInputs + i];

		_prevExploratoryOutput[i] = _output[i];
		_prevOutput[i] = _stateActionSOM.getNode(closestState)._weights[_numInputs + i];
	}

	_prevInput = _input;
}
