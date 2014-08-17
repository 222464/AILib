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

#include <elman/ElmanNetwork.h>

using namespace elman;

void ElmanNetwork::createRandom(size_t numInputs, size_t numOutputs, size_t numHidden, float minWeight, float maxWeight, std::mt19937 &generator) {
	_input.clear();
	_hidden.clear();
	_context.clear();
	_output.clear();

	std::uniform_real_distribution<float> weightDist(minWeight, maxWeight);

	_input.assign(numInputs, 0.0f);

	_hidden.resize(numHidden);

	size_t numHiddenInput = numInputs + numHidden;

	for (size_t i = 0; i < numHidden; i++) {
		_hidden[i]._bias._weight = weightDist(generator);

		_hidden[i]._connections.resize(numHiddenInput);

		for (size_t j = 0; j < numHiddenInput; j++)
			_hidden[i]._connections[j]._weight = weightDist(generator);
	}

	_context.resize(numHidden);

	_output.resize(numOutputs);

	for (size_t i = 0; i < numOutputs; i++) {
		_output[i]._bias._weight = weightDist(generator);

		_output[i]._connections.resize(numHidden);

		for (size_t j = 0; j < numHidden; j++)
			_output[i]._connections[j]._weight = weightDist(generator);
	}
}

void ElmanNetwork::activate() {
	// Update hidden
	for (size_t i = 0; i < _hidden.size(); i++) {
		float sum = _hidden[i]._bias._weight;

		// Inputs
		for (size_t j = 0; j < _input.size(); j++)
			sum += _hidden[i]._connections[j]._weight * _input[j];

		// Context
		for (size_t j = 0; j < _context.size(); j++)
			sum += _hidden[i]._connections[_input.size() + j]._weight * _context[j]._output;

		_hidden[i]._output = sigmoid(sum);
	}

	// Update output
	for (size_t i = 0; i < _output.size(); i++) {
		float sum = _output[i]._bias._weight;

		// Inputs
		for (size_t j = 0; j < _hidden.size(); j++)
			sum += _output[i]._connections[j]._weight * _hidden[j]._output;

		_output[i]._output = sum;
	}
}

void ElmanNetwork::updateContext() {
	// Update context
	for (size_t i = 0; i < _context.size(); i++)
		_context[i]._output = _hidden[i]._output;
}

void ElmanNetwork::calculateGradient(std::vector<float> &targets) {
	for (size_t i = 0; i < _output.size(); i++)
		_output[i]._error = targets[i] - _output[i]._output;

	for (size_t i = 0; i < _context.size(); i++) {
		float sum = 0.0f;

		for (size_t j = 0; j < _hidden.size(); j++)
			sum += _hidden[j]._error * _hidden[j]._connections[_input.size() + i]._weight;

		_context[i]._error = sum * _context[i]._output * (1.0f - _context[i]._output);
	}

	for (size_t i = 0; i < _hidden.size(); i++) {
		float sum = 0.0f;

		for (size_t j = 0; j < _output.size(); j++)
			sum += _output[j]._error * _output[j]._connections[i]._weight;

		sum += _context[i]._error;

		_hidden[i]._error = sum * _hidden[i]._output * (1.0f - _hidden[i]._output);
	}
}

void ElmanNetwork::moveAlongGradient(float alpha) {
	for (size_t i = 0; i < _output.size(); i++) {
		_output[i]._bias._weight += alpha * _output[i]._error;

		for (size_t j = 0; j < _hidden.size(); j++)
			_output[i]._connections[j]._weight += alpha * _output[i]._error * _hidden[j]._output;
	}

	for (size_t i = 0; i < _hidden.size(); i++) {
		_hidden[i]._bias._weight += alpha * _hidden[i]._error;

		for (size_t j = 0; j < _input.size(); j++)
			_hidden[i]._connections[j]._weight += alpha * _hidden[i]._error * _input[j];

		for (size_t j = 0; j < _context.size(); j++)
			_hidden[i]._connections[_input.size() + j]._weight += alpha * _hidden[i]._error * _context[j]._output;
	}
}