#include "SparseCoder.h"

#include <algorithm>

using namespace deep;

void SparseCoder::createRandom(int numVisible, int numHidden, std::mt19937 &generator) {
	std::uniform_real_distribution<float> weightDistVisibleHidden(0.0f, 1.0f);

	_visible.resize(numVisible);

	_hidden.resize(numHidden);

	for (int hi = 0; hi < numHidden; hi++) {
		_hidden[hi]._bias._weight = 0.0f;

		_hidden[hi]._visibleHiddenConnections.resize(numVisible);

		float dist2 = 0.0f;

		for (int vi = 0; vi < numVisible; vi++) {
			_hidden[hi]._visibleHiddenConnections[vi]._weight = weightDistVisibleHidden(generator);

			dist2 += _hidden[hi]._visibleHiddenConnections[vi]._weight * _hidden[hi]._visibleHiddenConnections[vi]._weight;
		}

		float normFactor = 1.0f / dist2;

		for (int vi = 0; vi < numVisible; vi++)
			_hidden[hi]._visibleHiddenConnections[vi]._weight *= normFactor;

		_hidden[hi]._hiddenHiddenConnections.resize(numHidden);

		for (int hio = 0; hio < numHidden; hio++)
			_hidden[hi]._hiddenHiddenConnections[hio]._weight = 0.0f;
	}
}

void SparseCoder::activate(float sparsity, float lambda, float dt) {
	// Activate
	for (int hi = 0; hi < _hidden.size(); hi++) {
		float sum = -_hidden[hi]._bias._weight;

		for (int vi = 0; vi < _visible.size(); vi++)
			sum += _hidden[hi]._visibleHiddenConnections[vi]._weight * _visible[vi]._input;

		for (int hio = 0; hio < _hidden.size(); hio++)
			sum += _hidden[hi]._hiddenHiddenConnections[hio]._weight * _hidden[hio]._activationPrev;

		float delta = sigmoid(sum * lambda) - _hidden[hi]._activationPrev;

		_hidden[hi]._activation = _hidden[hi]._activationPrev + delta * dt;

		_hidden[hi]._state = _hidden[hi]._activation > 0.5f ? 1.0f : 0.0f;
	}
}

void SparseCoder::reconstruct() {
	for (int vi = 0; vi < _visible.size(); vi++) {
		float sum = 0.0f;

		for (int hi = 0; hi < _hidden.size(); hi++)
			sum += _hidden[hi]._visibleHiddenConnections[vi]._weight * _hidden[hi]._state;

		_visible[vi]._reconstruction = sum;
	}
}

void SparseCoder::learn(float alpha, float beta, float gamma, float sparsity) {
	std::vector<float> visibleErrors(_visible.size());

	for (int vi = 0; vi < _visible.size(); vi++)
		visibleErrors[vi] = _visible[vi]._input - _visible[vi]._reconstruction;

	float sparsitySquared = sparsity * sparsity;

	for (int hi = 0; hi < _hidden.size(); hi++) {
		if (_hidden[hi]._state != 0.0f) {
			for (int vi = 0; vi < _visible.size(); vi++)
				_hidden[hi]._visibleHiddenConnections[vi]._weight += beta * _hidden[hi]._state * (_visible[vi]._input - _hidden[hi]._visibleHiddenConnections[vi]._weight);
		}

		for (int hio = 0; hio < _hidden.size(); hio++)
			_hidden[hi]._hiddenHiddenConnections[hio]._weight = std::min(0.0f, _hidden[hi]._hiddenHiddenConnections[hio]._weight - alpha * (_hidden[hi]._state * _hidden[hio]._state - sparsity * sparsity));

		_hidden[hi]._hiddenHiddenConnections[hi]._weight = 0.0f;

		_hidden[hi]._bias._weight += gamma * (_hidden[hi]._state - sparsity);
	}

	// Average out inhibitory weights
	for (int hi = 0; hi < _hidden.size() / 2; hi++) {
		for (int hio = 0; hio < _hidden.size(); hio++) {
			float average = 0.5f * (_hidden[hi]._hiddenHiddenConnections[hio]._weight + _hidden[hio]._hiddenHiddenConnections[hi]._weight);

			_hidden[hi]._hiddenHiddenConnections[hio]._weight = average;
			_hidden[hio]._hiddenHiddenConnections[hi]._weight = average;
		}
	}
}

void SparseCoder::stepEnd() {
	for (int hi = 0; hi < _hidden.size(); hi++) {
		_hidden[hi]._activationPrev = _hidden[hi]._activation;
		_hidden[hi]._statePrev = _hidden[hi]._state;
	}
}