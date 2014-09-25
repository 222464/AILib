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

#include <deep/FERL.h>

#include <algorithm>

#include <iostream>

using namespace deep;

FERL::FERL()
: _zInv(1.0f), _prevMax(0.0f), _prevValue(0.0f)
{}

void FERL::createRandom(int numState, int numAction, int numHidden, float weightStdDev, std::mt19937 &generator) {
	_numState = numState;
	_numAction = numAction;

	_visible.resize(numState + numAction);

	_hidden.resize(numHidden);

	std::normal_distribution<float> weightDist(0.0f, weightStdDev);

	for (int vi = 0; vi < _visible.size(); vi++)
		_visible[vi]._bias._weight = weightDist(generator);
	
	for (int k = 0; k < _hidden.size(); k++) {
		_hidden[k]._bias._weight = weightDist(generator);

		_hidden[k]._connections.resize(_visible.size());

		for (int vi = 0; vi < _visible.size(); vi++)
			_hidden[k]._connections[vi]._weight = weightDist(generator);
	}

	_zInv = 1.0f / std::sqrt(static_cast<float>(numState));
}

void FERL::createFromParents(const FERL &parent1, const FERL &parent2, float averageChance, std::mt19937 &generator) {
	_numState = parent1._numState;
	_numAction = parent1._numAction;

	_visible.resize(_numState + _numAction);

	_hidden.resize(parent1._hidden.size());

	std::uniform_real_distribution<float> uniformDist(0.0f, 1.0f);

	for (int vi = 0; vi < _visible.size(); vi++)
		_visible[vi]._bias._weight = uniformDist(generator) < averageChance ? (parent1._visible[vi]._bias._weight + parent2._visible[vi]._bias._weight) * 0.5f : (uniformDist(generator) < 0.5f ? parent1._visible[vi]._bias._weight : parent2._visible[vi]._bias._weight);

	for (int k = 0; k < _hidden.size(); k++) {
		_hidden[k]._bias._weight = uniformDist(generator) < averageChance ? (parent1._hidden[k]._bias._weight + parent2._hidden[k]._bias._weight) * 0.5f : (uniformDist(generator) < 0.5f ? parent1._hidden[k]._bias._weight : parent2._hidden[k]._bias._weight);

		_hidden[k]._connections.resize(_visible.size());

		for (int vi = 0; vi < _visible.size(); vi++)
			_hidden[k]._connections[vi]._weight = uniformDist(generator) < averageChance ? (parent1._hidden[k]._connections[vi]._weight + parent2._hidden[k]._connections[vi]._weight) * 0.5f : (uniformDist(generator) < 0.5f ? parent1._hidden[k]._connections[vi]._weight : parent2._hidden[k]._connections[vi]._weight);
	}

	_zInv = 1.0f / std::sqrt(static_cast<float>(_numState));
}

void FERL::mutate(float perturbationStdDev, std::mt19937 &generator) {
	std::normal_distribution<float> perturbationDist(0.0f, perturbationStdDev);

	for (int vi = 0; vi < _visible.size(); vi++)
		_visible[vi]._bias._weight += perturbationDist(generator);

	for (int k = 0; k < _hidden.size(); k++) {
		_hidden[k]._bias._weight += perturbationDist(generator);

		_hidden[k]._connections.resize(_visible.size());

		for (int vi = 0; vi < _visible.size(); vi++)
			_hidden[k]._connections[vi]._weight += perturbationDist(generator);
	}
}

float FERL::freeEnergy() const {
	float sum = 0.0f;

	for (int k = 0; k < _hidden.size(); k++) {
		sum -= _hidden[k]._bias._weight * _hidden[k]._state;

		for (int vi = 0; vi < _visible.size(); vi++)
			sum -= _hidden[k]._connections[vi]._weight * _visible[vi]._state * _hidden[k]._state;

		sum += _hidden[k]._state * std::log(_hidden[k]._state) + (1.0f - _hidden[k]._state) * std::log(1.0f - _hidden[k]._state);
	}

	for (int vi = 0; vi < _visible.size(); vi++)
		sum -= _visible[vi]._bias._weight * _visible[vi]._state;

	return sum;
}

void FERL::step(const std::vector<float> &state, std::vector<float> &action, float reward, float qAlpha, float gamma, float lambdaGamma, float tauInv, int actionSearchIterations, int actionSearchSamples, float actionSearchAlpha, float breakChance, float perturbationStdDev, std::mt19937 &generator) {
	for (int i = 0; i < _numState; i++)
		_visible[i]._state = state[i];

	std::vector<float> maxAction(_numAction);

	float nextQ = -999999.0f;

	std::uniform_real_distribution<float> uniformDist(0.0f, 1.0f);

	for (int s = 0; s < actionSearchSamples; s++) {
		// Start with random inputs
		for (int j = 0; j < _numAction; j++)
			_visible[j + _numState]._state = uniformDist(generator) * 2.0f - 1.0f;

		activate();

		// Find best action and associated Q value
		for (int p = 0; p < actionSearchIterations; p++) {
			for (int j = 0; j < _numAction; j++) {
				int index = j + _numState;

				float sum = _visible[index]._bias._weight;

				for (int k = 0; k < _hidden.size(); k++)
					sum += _hidden[k]._connections[index]._weight * _hidden[k]._state;

				_visible[index]._state = std::min(1.0f, std::max(-1.0f, _visible[index]._state + actionSearchAlpha * sum));
			}

			// Q value of best action
			activate();
		}

		float q = value();

		if (q > nextQ) {
			nextQ = q;

			for (int j = 0; j < _numAction; j++)
				maxAction[j] = _visible[j + _numState]._state;
		}
	}

	// Actual action (perturbed from maximum)
	if (action.size() != _numAction)
		action.resize(_numAction);

	std::normal_distribution<float> perturbationDist(0.0f, perturbationStdDev);

	for (int j = 0; j < _numAction; j++)
	if (uniformDist(generator) < breakChance)
		action[j] = uniformDist(generator) * 2.0f - 1.0f;
	else
		action[j] = std::min(1.0f, std::max(-1.0f, maxAction[j] + perturbationDist(generator)));

	// Activate current (selected) action so eligibilities can be updated properly
	for (int j = 0; j < _numAction; j++)
		_visible[_numState + j]._state = action[j];

	activate();

	float predictedQ = value();

	// Update Q
	float newAdv = _prevMax + (reward + gamma * nextQ - _prevMax) * tauInv;

	float error = qAlpha * (newAdv - _prevValue);

	std::cout << predictedQ << std::endl;

	_prevMax = nextQ;
	_prevValue = predictedQ;

	updateOnError(error, lambdaGamma);
}

void FERL::activate() {
	for (int k = 0; k < _hidden.size(); k++) {
		float sum = _hidden[k]._bias._weight;

		for (int vi = 0; vi < _visible.size(); vi++)
			sum += _hidden[k]._connections[vi]._weight * _visible[vi]._state;

		_hidden[k]._state = sigmoid(sum);
	}
}

void FERL::updateOnError(float error, float lambdaGamma) {
	// Update weights
	for (int k = 0; k < _hidden.size(); k++) {
		_hidden[k]._bias._weight += error * _hidden[k]._bias._eligibility;

		for (int vi = 0; vi < _visible.size(); vi++)
			_hidden[k]._connections[vi]._weight += error * _hidden[k]._connections[vi]._eligibility;
	}

	for (int vi = 0; vi < _visible.size(); vi++)
		_visible[vi]._bias._weight += error * _visible[vi]._bias._eligibility;

	// Update eligibilities
	for (int k = 0; k < _hidden.size(); k++) {
		_hidden[k]._bias._eligibility = lambdaGamma * _hidden[k]._bias._eligibility + _zInv * _hidden[k]._state;

		for (int vi = 0; vi < _visible.size(); vi++)
			_hidden[k]._connections[vi]._eligibility = lambdaGamma * _hidden[k]._connections[vi]._eligibility + _zInv * _hidden[k]._state * _visible[vi]._state;
	}

	for (int vi = 0; vi < _visible.size(); vi++)
		_visible[vi]._bias._eligibility = lambdaGamma * _visible[vi]._bias._eligibility + _zInv * _visible[vi]._state;
}