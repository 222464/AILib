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

#include <algorithm>

using namespace deep;

void RecurrentSparseAutoencoder::createRandom(int numVisibleNodes, int numHiddenNodes, float sparsity, float minInitWeight, float maxInitWeight, float recurrentScalar, std::mt19937 &generator) {
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
			_hiddenNodes[h]._hiddenHiddenConnections[ho]._weight = distInitWeight(generator) * recurrentScalar;
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

		_hiddenNodes[h]._state = numHigher < localActivity ? _hiddenNodes[h]._activation : 0.0f;

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

void RecurrentSparseAutoencoder::learn(float sparsity, float stateLeak, float alpha, float beta, float gamma, float epsilon, float momentum, float traceDecay, float temperature) {
	std::vector<float> reconstructionError(_visibleNodes.size());

	std::vector<float> hiddenErrors(_hiddenNodes.size());

	for (int v = 0; v < _visibleNodes.size(); v++)
		reconstructionError[v] = _visibleNodes[v]._state - _visibleNodes[v]._reconstructionPrev;

	for (int h = 0; h < _hiddenNodes.size(); h++) {
		float error = 0.0f;

		for (int v = 0; v < _visibleNodes.size(); v++)
			error += _visibleNodes[v]._hiddenVisibleConnections[h]._weight * reconstructionError[v];

		//float s = _hiddenNodes[h]._activationPrev;

		//error /= _visibleNodes.size();
		//error *= std::max(stateLeak, _hiddenNodes[h]._statePrev) * s * (1.0f - s);

		hiddenErrors[h] = _hiddenNodes[h]._statePrev * (1.0f - _hiddenNodes[h]._statePrev) * error;// / _visibleNodes.size(); //hiddenErrors[h] = _hiddenNodes[h]._statePrev * (error > 0.0f ? (1.0f - _hiddenNodes[h]._activationPrev) : (0.0f - _hiddenNodes[h]._activationPrev));
	}

	for (int v = 0; v < _visibleNodes.size(); v++) {
		for (int h = 0; h < _hiddenNodes.size(); h++) {
			float eligibility = reconstructionError[v] * _hiddenNodes[h]._statePrev;

			float newTrace = (1.0f - traceDecay) * _visibleNodes[v]._hiddenVisibleConnections[h]._trace + epsilon * std::exp(-std::abs(_visibleNodes[v]._hiddenVisibleConnections[h]._trace * temperature)) * eligibility;

			float delta = _visibleNodes[v]._hiddenVisibleConnections[h]._prevWeightDelta * momentum + alpha * newTrace;
			
			_visibleNodes[v]._hiddenVisibleConnections[h]._weight += delta;
			_visibleNodes[v]._hiddenVisibleConnections[h]._trace = newTrace;
			_visibleNodes[v]._hiddenVisibleConnections[h]._prevWeightDelta = delta;
		}

		float eligibility = reconstructionError[v];

		float newTrace = (1.0f - traceDecay) * _visibleNodes[v]._bias._trace + epsilon * std::exp(-std::abs(_visibleNodes[v]._bias._trace * temperature)) * eligibility;

		float delta = _visibleNodes[v]._bias._prevWeightDelta * momentum + alpha * newTrace;

		_visibleNodes[v]._bias._weight += delta;
		_visibleNodes[v]._bias._trace = newTrace;
		_visibleNodes[v]._bias._prevWeightDelta = delta;
	}

	for (int h = 0; h < _hiddenNodes.size(); h++) {
		for (int v = 0; v < _visibleNodes.size(); v++) {
			float eligibility = hiddenErrors[h] * _visibleNodes[v]._statePrev;

			float newTrace = (1.0f - traceDecay) * _hiddenNodes[h]._visibleHiddenConnections[v]._trace + epsilon * std::exp(-std::abs(_hiddenNodes[h]._visibleHiddenConnections[v]._trace * temperature)) * eligibility;

			float delta = _hiddenNodes[h]._visibleHiddenConnections[v]._prevWeightDelta * momentum + alpha * newTrace + gamma * (sparsity - _hiddenNodes[h]._dutyCycle) * _visibleNodes[v]._statePrev;
			
			_hiddenNodes[h]._visibleHiddenConnections[v]._weight += delta;
			_hiddenNodes[h]._visibleHiddenConnections[v]._trace = newTrace;
			_hiddenNodes[h]._visibleHiddenConnections[v]._prevWeightDelta = delta;
		}

		for (int ho = 0; ho < _hiddenNodes.size(); ho++) {
			float eligibility = hiddenErrors[h] * _hiddenNodes[ho]._statePrevPrev;

			float newTrace = (1.0f - traceDecay) * _hiddenNodes[h]._hiddenHiddenConnections[ho]._trace + epsilon * std::exp(-std::abs(_hiddenNodes[h]._hiddenHiddenConnections[ho]._trace * temperature)) * eligibility;

			float delta = _hiddenNodes[h]._hiddenHiddenConnections[ho]._prevWeightDelta * momentum + beta * newTrace + gamma * (sparsity - _hiddenNodes[h]._dutyCycle) * _hiddenNodes[ho]._statePrevPrev;
			
			_hiddenNodes[h]._hiddenHiddenConnections[ho]._weight += delta;
			_hiddenNodes[h]._hiddenHiddenConnections[ho]._trace = newTrace;
			_hiddenNodes[h]._hiddenHiddenConnections[ho]._prevWeightDelta = delta;
		}

		float eligibility = hiddenErrors[h];

		float newTrace = (1.0f - traceDecay) * _hiddenNodes[h]._bias._trace + epsilon * std::exp(-std::abs(_hiddenNodes[h]._bias._trace * temperature)) * eligibility;

		float delta = _hiddenNodes[h]._bias._prevWeightDelta * momentum + alpha * newTrace + gamma * (sparsity - _hiddenNodes[h]._dutyCycle);

		_hiddenNodes[h]._bias._weight += delta;
		_hiddenNodes[h]._bias._trace = newTrace;
		_hiddenNodes[h]._bias._prevWeightDelta = delta;
	}
}

void RecurrentSparseAutoencoder::learnExperience(const Experience &experience, float sparsity, float stateLeak, float alpha, float beta, float gamma, float epsilon, float momentum, float traceDecay, float temperature) {
	// ---------------------------- Activate ------------------------------

	int localActivity = std::round(sparsity * _hiddenNodes.size());

	for (int h = 0; h < _hiddenNodes.size(); h++) {
		float sum = _hiddenNodes[h]._bias._weight;

		for (int v = 0; v < _visibleNodes.size(); v++)
			sum += _hiddenNodes[h]._visibleHiddenConnections[v]._weight * experience._visibleStatesPrev[v];

		for (int ho = 0; ho < _hiddenNodes.size(); ho++)
			sum += _hiddenNodes[h]._hiddenHiddenConnections[ho]._weight * experience._hiddenStatesPrevPrev[ho];

		_hiddenNodes[h]._activation = sigmoid(sum);
	}

	// Sparsify
	for (int h = 0; h < _hiddenNodes.size(); h++) {
		float numHigher = 0.0f;

		for (int ho = 0; ho < _hiddenNodes.size(); ho++)
			if (_hiddenNodes[ho]._activation > _hiddenNodes[h]._activation)
				numHigher++;

		_hiddenNodes[h]._state = numHigher < localActivity ? _hiddenNodes[h]._activation : 0.0f;
	}

	// Reconstruct
	for (int v = 0; v < _visibleNodes.size(); v++) {
		float sum = _visibleNodes[v]._bias._weight;

		for (int h = 0; h < _hiddenNodes.size(); h++)
			sum += _visibleNodes[v]._hiddenVisibleConnections[h]._weight * _hiddenNodes[h]._state;

		_visibleNodes[v]._reconstruction = sum;
	}

	// ------------------------------ Learn ------------------------------
	
	std::vector<float> reconstructionError(_visibleNodes.size());

	std::vector<float> hiddenErrors(_hiddenNodes.size());

	for (int v = 0; v < _visibleNodes.size(); v++)
		reconstructionError[v] = experience._visibleStates[v] - _visibleNodes[v]._reconstruction;

	for (int h = 0; h < _hiddenNodes.size(); h++) {
		float error = 0.0f;

		for (int v = 0; v < _visibleNodes.size(); v++)
			error += _visibleNodes[v]._hiddenVisibleConnections[h]._weight * reconstructionError[v];

		//float s = _hiddenNodes[h]._activationPrev;

		//error /= _visibleNodes.size();
		//error *= std::max(stateLeak, _hiddenNodes[h]._statePrev) * s * (1.0f - s);

		hiddenErrors[h] = _hiddenNodes[h]._statePrev * (1.0f - _hiddenNodes[h]._statePrev) * error;// / _visibleNodes.size();
	}

	for (int v = 0; v < _visibleNodes.size(); v++) {
		for (int h = 0; h < _hiddenNodes.size(); h++) {
			float eligibility = reconstructionError[v] * experience._hiddenStatesPrev[h];

			float newTrace = (1.0f - traceDecay) * _visibleNodes[v]._hiddenVisibleConnections[h]._trace + epsilon * std::exp(-std::abs(_visibleNodes[v]._hiddenVisibleConnections[h]._trace * temperature)) * eligibility;

			float delta = _visibleNodes[v]._hiddenVisibleConnections[h]._prevWeightDelta * momentum + alpha * newTrace;

			_visibleNodes[v]._hiddenVisibleConnections[h]._weight += delta;
			_visibleNodes[v]._hiddenVisibleConnections[h]._trace = newTrace;
			_visibleNodes[v]._hiddenVisibleConnections[h]._prevWeightDelta = delta;
		}

		float eligibility = reconstructionError[v];

		float newTrace = (1.0f - traceDecay) * _visibleNodes[v]._bias._trace + epsilon * std::exp(-std::abs(_visibleNodes[v]._bias._trace * temperature)) * eligibility;

		float delta = _visibleNodes[v]._bias._prevWeightDelta * momentum + alpha * newTrace;

		_visibleNodes[v]._bias._weight += delta;
		_visibleNodes[v]._bias._trace = newTrace;
		_visibleNodes[v]._bias._prevWeightDelta = delta;
	}

	for (int h = 0; h < _hiddenNodes.size(); h++) {
		for (int v = 0; v < _visibleNodes.size(); v++) {
			float eligibility = hiddenErrors[h] * experience._visibleStatesPrev[v];

			float newTrace = (1.0f - traceDecay) * _hiddenNodes[h]._visibleHiddenConnections[v]._trace + epsilon * std::exp(-std::abs(_hiddenNodes[h]._visibleHiddenConnections[v]._trace * temperature)) * eligibility;

			float delta = _hiddenNodes[h]._visibleHiddenConnections[v]._prevWeightDelta * momentum + alpha * newTrace + gamma * (sparsity - _hiddenNodes[h]._dutyCycle) * _visibleNodes[v]._statePrev;

			_hiddenNodes[h]._visibleHiddenConnections[v]._weight += delta;
			_hiddenNodes[h]._visibleHiddenConnections[v]._trace = newTrace;
			_hiddenNodes[h]._visibleHiddenConnections[v]._prevWeightDelta = delta;
		}

		for (int ho = 0; ho < _hiddenNodes.size(); ho++) {
			float eligibility = hiddenErrors[h] * experience._hiddenStatesPrevPrev[ho];

			float newTrace = (1.0f - traceDecay) * _hiddenNodes[h]._hiddenHiddenConnections[ho]._trace + epsilon * std::exp(-std::abs(_hiddenNodes[h]._hiddenHiddenConnections[ho]._trace * temperature)) * eligibility;

			float delta = _hiddenNodes[h]._hiddenHiddenConnections[ho]._prevWeightDelta * momentum + beta * newTrace + gamma * (sparsity - _hiddenNodes[h]._dutyCycle) * _hiddenNodes[ho]._statePrevPrev;

			_hiddenNodes[h]._hiddenHiddenConnections[ho]._weight += delta;
			_hiddenNodes[h]._hiddenHiddenConnections[ho]._trace = newTrace;
			_hiddenNodes[h]._hiddenHiddenConnections[ho]._prevWeightDelta = delta;
		}

		float eligibility = hiddenErrors[h];

		float newTrace = (1.0f - traceDecay) * _hiddenNodes[h]._bias._trace + epsilon * std::exp(-std::abs(_hiddenNodes[h]._bias._trace * temperature)) * eligibility;

		float delta = _hiddenNodes[h]._bias._prevWeightDelta * momentum + alpha * newTrace + gamma * (sparsity - _hiddenNodes[h]._dutyCycle);

		_hiddenNodes[h]._bias._weight += delta;
		_hiddenNodes[h]._bias._trace = newTrace;
		_hiddenNodes[h]._bias._prevWeightDelta = delta;
	}
}

void RecurrentSparseAutoencoder::stepBegin() {
	for (int h = 0; h < _hiddenNodes.size(); h++) {
		_hiddenNodes[h]._activationPrev = _hiddenNodes[h]._activation;

		_hiddenNodes[h]._statePrevPrev = _hiddenNodes[h]._statePrev;
		_hiddenNodes[h]._statePrev = _hiddenNodes[h]._state;
	}

	for (int v = 0; v < _visibleNodes.size(); v++) {
		_visibleNodes[v]._statePrev = _visibleNodes[v]._state;

		_visibleNodes[v]._reconstructionPrev = _visibleNodes[v]._reconstruction;
	}
}

void RecurrentSparseAutoencoder::clearMemory() {
	for (int h = 0; h < _hiddenNodes.size(); h++) {
		_hiddenNodes[h]._statePrevPrev = 0.0f;
		_hiddenNodes[h]._statePrev = 0.0f;
	}

	for (int v = 0; v < _visibleNodes.size(); v++)
		_visibleNodes[v]._statePrev = 0.0f;
}

void RecurrentSparseAutoencoder::getCurrentExperience(Experience &experience) const {
	experience._visibleStates.resize(_visibleNodes.size());
	experience._visibleStatesPrev.resize(_visibleNodes.size());
	experience._hiddenStatesPrev.resize(_hiddenNodes.size());
	experience._hiddenStatesPrevPrev.resize(_hiddenNodes.size());

	for (int v = 0; v < _visibleNodes.size(); v++) {
		experience._visibleStates[v] = _visibleNodes[v]._state;
		experience._visibleStatesPrev[v] = _visibleNodes[v]._statePrev;
	}

	for (int h = 0; h < _hiddenNodes.size(); h++) {
		experience._hiddenStatesPrev[h] = _hiddenNodes[h]._statePrev;
		experience._hiddenStatesPrevPrev[h] = _hiddenNodes[h]._statePrevPrev;
	}
}