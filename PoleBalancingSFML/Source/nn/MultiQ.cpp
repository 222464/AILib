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

#include <nn/MultiQ.h>

#include <iostream>

using namespace nn;

MultiQ::MultiQ()
: _epsilon(0.1f), _output(0), _prevValue(0.0f), _gamma(0.95f), _alpha(0.01f)
{}

MultiQ::~MultiQ() {
}

void MultiQ::createRandom(size_t numInputs, size_t numOutputs, size_t numHiddenLayers, size_t numNeuronsPerHiddenLayer, float minWeight, float maxWeight, unsigned long seed) {
	_network.createRandom(numInputs, numOutputs, numHiddenLayers, numNeuronsPerHiddenLayer, minWeight, maxWeight, seed);

	_currentInputs.assign(numOutputs, 0.0f);
	_prevInputs.assign(numOutputs, 0.0f);
}

void MultiQ::step(float reward) {
	for (size_t i = 0; i < _network.getNumInputs(); i++)
		_network.setInput(i, _currentInputs[i]);

	_network.activateLinearOutputLayer();

	// Select action
	size_t prevOutput = _output;

	size_t maxAction = 0;

	for (size_t i = 1; i < _network.getNumOutputs(); i++) {
		if (_network.getOutput(i) > _network.getOutput(maxAction))
			maxAction = i;
	}

	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

	if (dist01(_generator) < _epsilon) {
		std::uniform_int_distribution<int> distAction(0, _network.getNumOutputs() - 1);

		_output = distAction(_generator);
	}
	else
		_output = maxAction;

	float newPrevQ = reward + _gamma * _network.getOutput(maxAction);

	_prevValue = _network.getOutput(_output);

	// Set previous inputs and update Q
	for (size_t i = 0; i < _network.getNumInputs(); i++)
		_network.setInput(i, _prevInputs[i]);

	_network.activateLinearOutputLayer();

	std::vector<float> targets(_network.getNumOutputs());

	for (size_t i = 0; i < _network.getNumOutputs(); i++)
		targets[i] = _network.getOutput(i);

	targets[prevOutput] = newPrevQ;

	FeedForwardNeuralNetwork::Gradient grad;

	_network.getGradientLinearOutputLayer(targets, grad);
	_network.moveAlongGradient(grad, _alpha);

	_prevInputs = _currentInputs;

	std::cout << "Q: " << _network.getOutput(0) << " " << _network.getOutput(1) << " " << _network.getOutput(2) << std::endl;
}