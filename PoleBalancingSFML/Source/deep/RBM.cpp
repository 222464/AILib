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

#include <deep/RBM.h>

#include <iostream>

using namespace deep;

void RBM::createRandom(size_t numVisible, size_t numHidden, double minWeight, double maxWeight, std::mt19937 &generator) {
	std::uniform_real_distribution<double> weightDist(minWeight, maxWeight);

	_visible.resize(numVisible + 1); // + 1 for bias

	_hidden.resize(numHidden + 1); // + 1 for bias

	for (size_t i = 0; i < _hidden.size(); i++) {
		_hidden[i]._connections.resize(_visible.size());

		for (size_t j = 0; j < _hidden[i]._connections.size(); j++)
			_hidden[i]._connections[j]._weight = weightDist(generator);
	}

	// Bias always outputs 1
	_visible.back()._probability = 1.0;
	_hidden.back()._probability = 1.0;
	_hidden.back()._output = 1.0;
}

void RBM::activate(std::mt19937 &generator) {
	std::uniform_real_distribution<double> dist01(0.0, 1.0);

	for (size_t i = 0; i < _hidden.size(); i++) {
		double sum = 0.0;

		for (size_t j = 0; j < _hidden[i]._connections.size(); j++)
			sum += _hidden[i]._connections[j]._weight * _visible[j]._probability;

		_hidden[i]._probability = sigmoid(sum);
		_hidden[i]._output = dist01(generator) < _hidden[i]._probability ? 1.0 : 0.0;

		for (size_t j = 0; j < _hidden[i]._connections.size(); j++)
			_hidden[i]._connections[j]._positive = _hidden[i]._probability * _visible[j]._probability;
	}
}

void RBM::activateLight() {
	for (size_t i = 0; i < _hidden.size(); i++) {
		double sum = 0.0;

		for (size_t j = 0; j < _hidden[i]._connections.size(); j++)
			sum += _hidden[i]._connections[j]._weight * _visible[j]._probability;

		_hidden[i]._probability = sigmoid(sum);
	}
}

void RBM::learn(double alpha, std::mt19937 &generator) {
	std::uniform_real_distribution<double> dist01(0.0, 1.0);

	for (size_t i = 0; i < _visible.size(); i++) {
		double sum = 0.0;

		for (size_t j = 0; j < _hidden.size(); j++)
			sum += _hidden[j]._connections[i]._weight * _hidden[j]._output;

		_visible[i]._probability = sigmoid(sum);
	}

	_visible.back()._probability = 1.0;

	for (size_t i = 0; i < _hidden.size(); i++) {
		double sum = 0.0;

		for (size_t j = 0; j < _hidden[i]._connections.size(); j++)
			sum += _hidden[i]._connections[j]._weight * _visible[j]._probability;

		_hidden[i]._probability = sigmoid(sum);

		for (size_t j = 0; j < _hidden[i]._connections.size(); j++)
			_hidden[i]._connections[j]._negative = _hidden[i]._probability * _visible[j]._probability;
	}

	for (size_t i = 0; i < _hidden.size(); i++)
	for (size_t j = 0; j < _hidden[i]._connections.size(); j++)
		_hidden[i]._connections[j]._weight += alpha * (_hidden[i]._connections[j]._positive - _hidden[i]._connections[j]._negative);
}