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

using namespace deep;

void RBM::createRandom(size_t numVisible, size_t numHidden, float minWeight, float maxWeight, std::mt19937 &generator) {
	std::uniform_real_distribution<float> weightDist(minWeight, maxWeight);

	_visible.resize(numVisible);

	for (size_t i = 0; i < _visible.size(); i++)
		_visible[i]._bias._weight = weightDist(generator);

	_hidden.resize(numHidden);

	for (size_t i = 0; i < _hidden.size(); i++) {
		_hidden[i]._bias._weight = weightDist(generator);

		_hidden[i]._connections.resize(numVisible);

		for (size_t j = 0; j < _hidden[i]._connections.size(); j++)
			_hidden[i]._connections[j]._weight = weightDist(generator);
	}
}

void RBM::activate(std::mt19937 &generator) {
	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

	for (size_t i = 0; i < _visible.size(); i++)
		_visible[i]._bias._positive = _visible[i]._output;

	for (size_t i = 0; i < _hidden.size(); i++) {
		float sum = _hidden[i]._bias._weight;

		for (size_t j = 0; j < _hidden[i]._connections.size(); j++)
			sum += _hidden[i]._connections[j]._weight * _visible[j]._output;

		_hidden[i]._output = dist01(generator) < sigmoid(sum) ? 1.0f : 0.0f;

		_hidden[i]._bias._positive = _hidden[i]._output;

		for (size_t j = 0; j < _hidden[i]._connections.size(); j++)
			_hidden[i]._connections[j]._positive = _hidden[i]._output * _visible[j]._output;
	}
}

void RBM::learn(float alpha, std::mt19937 &generator) {
	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

	for (size_t i = 0; i < _visible.size(); i++) {
		float sum = _visible[i]._bias._weight;

		for (size_t j = 0; j < _hidden.size(); j++)
			sum += _hidden[j]._connections[i]._weight * _hidden[j]._output;

		_visible[i]._output = dist01(generator) < sigmoid(sum) ? 1.0f : 0.0f;
	}

	for (size_t i = 0; i < _visible.size(); i++)
		_visible[i]._bias._negative = _visible[i]._output;

	for (size_t i = 0; i < _hidden.size(); i++) {
		float sum = _hidden[i]._bias._weight;

		for (size_t j = 0; j < _hidden[i]._connections.size(); j++)
			sum += _hidden[i]._connections[j]._weight * _visible[j]._output;

		_hidden[i]._output = dist01(generator) < sigmoid(sum) ? 1.0f : 0.0f;

		_hidden[i]._bias._negative = _hidden[i]._output;

		for (size_t j = 0; j < _hidden[i]._connections.size(); j++)
			_hidden[i]._connections[j]._negative = _hidden[i]._output * _visible[j]._output;
	}

	for (size_t i = 0; i < _visible.size(); i++)
		_visible[i]._bias._weight += alpha * (_visible[i]._bias._positive - _visible[i]._bias._negative);

	for (size_t i = 0; i < _hidden.size(); i++) {
		_hidden[i]._bias._weight += alpha * (_hidden[i]._bias._positive - _hidden[i]._bias._negative);

		for (size_t j = 0; j < _hidden[i]._connections.size(); j++)
			_hidden[i]._connections[j]._weight += alpha * (_hidden[i]._connections[j]._positive - _hidden[i]._connections[j]._negative);
	}
}