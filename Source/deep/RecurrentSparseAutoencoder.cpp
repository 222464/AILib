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

#include <deep/RecurrentSparseAutoencoder.h>

using namespace deep;

void RecurrentSparseAutoencoder::createRandom(int numVisibleNodes, int numHiddenNodes, float sparsity, float minInitWeight, float maxInitWeight, std::mt19937 &generator) {
	_hiddenNodes.resize(numHiddenNodes);
	_visibleNodes.resize(numVisibleNodes);

	std::uniform_real_distribution<float> distInitWeight(minInitWeight, maxInitWeight);

	for (int h = 0; h < _hiddenNodes.size(); h++) {
		_hiddenNodes[h]._dutyCycle = sparsity;

		_hiddenNodes[h]._bias._weight = distInitWeight(generator);

		_hiddenNodes[h]._visibleHiddenConnections.resize(_visibleNodes.size());

		for (int v = 0; v < _visibleNodes.size(); v++)
			_hiddenNodes[h]._visibleHiddenConnections[v]._weight = distInitWeight(generator);

		_hiddenNodes[h]._hiddenHiddenConnections.resize(_hiddenNodes.size());

		for (int ho = 0; ho < _hiddenNodes.size(); ho++)
			_hiddenNodes[h]._hiddenHiddenConnections[ho]._weight = distInitWeight(generator);
	}

	for (int v = 0; v < _visibleNodes.size(); v++) {
		_visibleNodes[v]._bias._weight = distInitWeight(generator);

		_visibleNodes[v]._hiddenVisibleConnections.resize(_hiddenNodes.size());

		for (int ho = 0; ho < _hiddenNodes.size(); ho++)
			_visibleNodes[v]._hiddenVisibleConnections[ho]._weight = distInitWeight(generator);
	}
}

void RecurrentSparseAutoencoder::activate(float sparsity, float dutyCycleDecay) {
	int localActivity = std::round(sparsity * _hiddenNodes.size());

	for (int h = 0; h < _hiddenNodes.size(); h++) {
		float sum = _hiddenNodes[h]._bias._weight;

		for (int v = 0; v < _visibleNodes.size(); v++)
			sum += _hiddenNodes[h]._visibleHiddenConnections[v]._weight * _visibleNodes[v]._state;
		
		for (int ho = 0; ho < _hiddenNodes.size(); ho++)
			sum += _hiddenNodes[h]._hiddenHiddenConnections[ho]._weight * _hiddenNodes[ho]._statePrev;

		_hiddenNodes[h]._activation = sigmoid(sum);
	}

	// Sparsify
	for (int h = 0; h < _hiddenNodes.size(); h++) {
		float numHigher = 0.0f;

		for (int ho = 0; ho < _hiddenNodes.size(); ho++)
			if (_hiddenNodes[ho]._activation > _hiddenNodes[h]._activation)
				numHigher++;

		_hiddenNodes[h]._state = numHigher < localActivity ? 1.0f : 0.0f;

		_hiddenNodes[h]._dutyCycle = (1.0f - dutyCycleDecay) * _hiddenNodes[h]._dutyCycle + dutyCycleDecay * _hiddenNodes[h]._state;
	}

	// Reconstruct
	for (int v = 0; v < _visibleNodes.size(); v++) {
		float sum = _visibleNodes[v]._bias._weight;

		for (int h = 0; h < _hiddenNodes.size(); h++)
			sum += _visibleNodes[v]._hiddenVisibleConnections[h]._weight * _hiddenNodes[h]._state;

		_visibleNodes[v]._reconstruction = sum;
	}
}

void RecurrentSparseAutoencoder::learn(float sparsity, float alpha, float beta, float gamma, float momentum) {
	std::vector<float> reconstructionError(_visibleNodes.size());

	std::vector<float> hiddenErrors(_hiddenNodes.size());

	for (int v = 0; v < _visibleNodes.size(); v++)
		reconstructionError[v] = _visibleNodes[v]._state - _visibleNodes[v]._reconstruction;

	for (int h = 0; h < _hiddenNodes.size(); h++) {
		float error = 0.0f;

		for (int v = 0; v < _visibleNodes.size(); v++)
			error += _visibleNodes[v]._hiddenVisibleConnections[h]._weight * reconstructionError[v];

		error *= _hiddenNodes[h]._statePrev * _hiddenNodes[h]._activationPrev * (1.0f - _hiddenNodes[h]._activationPrev);

		hiddenErrors[h] = error;
	}

	for (int v = 0; v < _visibleNodes.size(); v++) {
		for (int h = 0; h < _hiddenNodes.size(); h++) {
			float delta = _visibleNodes[v]._hiddenVisibleConnections[h]._prevWeightDelta * momentum + alpha * reconstructionError[v] * _hiddenNodes[h]._statePrev;
			_visibleNodes[v]._hiddenVisibleConnections[h]._weight += delta;
			_visibleNodes[v]._hiddenVisibleConnections[h]._prevWeightDelta = delta;
		}

		float delta = _visibleNodes[v]._bias._prevWeightDelta * momentum + alpha * reconstructionError[v];
		_visibleNodes[v]._bias._weight += delta;
		_visibleNodes[v]._bias._prevWeightDelta = delta;
	}

	for (int h = 0; h < _hiddenNodes.size(); h++) {
		for (int v = 0; v < _visibleNodes.size(); v++) {
			float delta = _hiddenNodes[h]._visibleHiddenConnections[v]._prevWeightDelta * momentum + alpha * hiddenErrors[h] * _visibleNodes[v]._statePrev;
			_hiddenNodes[h]._visibleHiddenConnections[v]._weight += delta;
			_hiddenNodes[h]._visibleHiddenConnections[v]._prevWeightDelta = delta;
		}

		for (int ho = 0; ho < _hiddenNodes.size(); ho++) {
			float delta = _hiddenNodes[h]._hiddenHiddenConnections[ho]._prevWeightDelta * momentum + beta * hiddenErrors[h] * _hiddenNodes[ho]._statePrevPrev;
			_hiddenNodes[h]._hiddenHiddenConnections[ho]._weight += delta;
			_hiddenNodes[h]._hiddenHiddenConnections[ho]._prevWeightDelta = delta;
		}

		float delta = _hiddenNodes[h]._bias._prevWeightDelta * momentum + alpha * hiddenErrors[h] + gamma * (sparsity - _hiddenNodes[h]._dutyCycle);
		_hiddenNodes[h]._bias._weight += delta;
		_hiddenNodes[h]._bias._prevWeightDelta = delta;
	}
}

void RecurrentSparseAutoencoder::stepBegin() {
	for (int h = 0; h < _hiddenNodes.size(); h++) {
		_hiddenNodes[h]._activationPrev = _hiddenNodes[h]._activation;

		_hiddenNodes[h]._statePrevPrev = _hiddenNodes[h]._statePrev;
		_hiddenNodes[h]._statePrev = _hiddenNodes[h]._state;
	}

	for (int v = 0; v < _visibleNodes.size(); v++)
		_visibleNodes[v]._statePrev = _visibleNodes[v]._state;
}

void RecurrentSparseAutoencoder::clearMemory() {
	for (int h = 0; h < _hiddenNodes.size(); h++) {
		_hiddenNodes[h]._statePrevPrev = 0.0f;
		_hiddenNodes[h]._statePrev = 0.0f;
	}

	for (int v = 0; v < _visibleNodes.size(); v++)
		_visibleNodes[v]._statePrev = 0.0f;
}
