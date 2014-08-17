#include "CTRNN.h"

#include <algorithm>

using namespace ctrnn;

void CTRNN::createRandom(size_t numInputs, size_t numOutputs, size_t numHidden, float minWeight, float maxWeight, float minTau, float maxTau, float minNoiseStdDev, float maxNoiseStdDev, std::mt19937 &generator) {
	_numOutputs = numOutputs;
	_numHidden = numHidden;

	_inputs.resize(numInputs);

	for (size_t i = 0; i < _inputs.size(); i++)
		_inputs[i] = 0.0f;

	_numNodesHiddenOutput = _numHidden + _numOutputs;
	_numNodesTotal = _numNodesHiddenOutput + _inputs.size();

	_weightMatrix.resize(_numNodesTotal * _numNodesHiddenOutput);
	_nodes.resize(_numNodesHiddenOutput);

	std::uniform_real_distribution<float> weightDist(minWeight, maxWeight);

	for (size_t i = 0; i < _weightMatrix.size(); i++) {
		_weightMatrix[i]._weight = weightDist(generator);
		_weightMatrix[i]._trace = 0.0f;
	}

	std::uniform_real_distribution<float> tauDist(minTau, maxTau);
	std::uniform_real_distribution<float> noiseStdDevDist(minNoiseStdDev, maxNoiseStdDev);

	for (size_t i = 0; i < _nodes.size(); i++) {
		_nodes[i]._bias = weightDist(generator);
		_nodes[i]._state = 0.0f;
		_nodes[i]._tauInv = 1.0f / tauDist(generator);
		_nodes[i]._noiseStdDev = noiseStdDevDist(generator);
		_nodes[i]._prevOutput = 0.0f;
		_nodes[i]._output = 0.0f;
	}
}

void CTRNN::createFromParents(const CTRNN &parent1, const CTRNN &parent2, float averageWeightsChance, float averageTausChance, float averageNoiseStdDevChance, std::mt19937 &generator) {
	_numOutputs = parent1._numOutputs;
	_numHidden = parent1._numHidden;

	_inputs.resize(parent1._inputs.size());

	for (size_t i = 0; i < _inputs.size(); i++)
		_inputs[i] = 0.0f;

	_numNodesHiddenOutput = _numHidden + _numOutputs;
	_numNodesTotal = _numNodesHiddenOutput + _inputs.size();

	_weightMatrix.resize(_numNodesTotal * _numNodesHiddenOutput);
	_nodes.resize(_numNodesHiddenOutput);

	std::uniform_real_distribution<float> uniformDist(0.0f, 1.0f);

	for (size_t i = 0; i < _weightMatrix.size(); i++) {
		if (uniformDist(generator) < averageWeightsChance)
			_weightMatrix[i]._weight = (parent1._weightMatrix[i]._weight + parent2._weightMatrix[i]._weight) * 0.5f;
		else
			_weightMatrix[i]._weight = uniformDist(generator) < 0.5f ? parent1._weightMatrix[i]._weight : parent2._weightMatrix[i]._weight;
	}

	for (size_t i = 0; i < _nodes.size(); i++) {
		if (uniformDist(generator) < averageWeightsChance)
			_nodes[i]._bias = (parent1._nodes[i]._bias + parent2._nodes[i]._bias) * 0.5f;
		else
			_nodes[i]._bias = uniformDist(generator) < 0.5f ? parent1._nodes[i]._bias : parent2._nodes[i]._bias;

		_nodes[i]._state = 0.0f;
		
		if (uniformDist(generator) < averageTausChance)
			_nodes[i]._tauInv = (parent1._nodes[i]._tauInv + parent2._nodes[i]._tauInv) * 0.5f;
		else
			_nodes[i]._tauInv = uniformDist(generator) < 0.5f ? parent1._nodes[i]._tauInv : parent2._nodes[i]._tauInv;

		if (uniformDist(generator) < averageNoiseStdDevChance)
			_nodes[i]._noiseStdDev = (parent1._nodes[i]._noiseStdDev + parent2._nodes[i]._noiseStdDev) * 0.5f;
		else
			_nodes[i]._noiseStdDev = uniformDist(generator) < 0.5f ? parent1._nodes[i]._noiseStdDev : parent2._nodes[i]._noiseStdDev;

		_nodes[i]._prevOutput = 0.0f;
		_nodes[i]._output = 0.0f;
	}
}

void CTRNN::mutate(float weightPerturbationChance, float maxWeightPerturbation, float tauPerturbationChance, float maxTauPerturbation, float noiseStdDevPerturbationChance, float maxNoiseStdDevPerturbation, std::mt19937 &generator) {
	std::uniform_real_distribution<float> uniformDist(0.0f, 1.0f);
	
	std::uniform_real_distribution<float> weightPerturbationDist(-maxWeightPerturbation, maxWeightPerturbation);

	for (size_t i = 0; i < _weightMatrix.size(); i++)
	if (uniformDist(generator) < weightPerturbationChance)
		_weightMatrix[i]._weight += weightPerturbationDist(generator);
	
	std::uniform_real_distribution<float> tauPerturbationDist(-maxTauPerturbation, maxTauPerturbation);

	std::uniform_real_distribution<float> noiseStdDevPerturbationDist(-maxNoiseStdDevPerturbation, maxNoiseStdDevPerturbation);

	for (size_t i = 0; i < _nodes.size(); i++) {
		if (uniformDist(generator) < tauPerturbationChance) {
			float tau = 1.0f / _nodes[i]._tauInv;

			tau = std::max(0.01f, tau + tau * tauPerturbationDist(generator));

			_nodes[i]._tauInv = 1.0f / tau;
		}

		if (uniformDist(generator) < noiseStdDevPerturbationChance)
			_nodes[i]._noiseStdDev = std::max(0.01f, _nodes[i]._noiseStdDev + noiseStdDevPerturbationDist(generator));
	}
}

void CTRNN::step(float dt, float reward, float traceDecay, std::mt19937 &generator) {
	for (size_t i = 0; i < _nodes.size(); i++) {
		std::normal_distribution<float> noiseDist(0.0f, _nodes[i]._noiseStdDev);

		float sum = noiseDist(generator);

		for (size_t j = 0; j < _nodes.size(); j++) {
			Weight &w = getWeight(i, j);

			w._weight += reward * w._trace;

			sum += w._weight * _nodes[j]._prevOutput;
		}

		for (size_t j = 0; j < _inputs.size(); j++) {
			Weight &w = getWeight(i, j + _numNodesHiddenOutput);

			w._weight += reward * w._trace;

			sum += w._weight * _inputs[j];
		}

		_nodes[i]._state += dt * _nodes[i]._tauInv * (-_nodes[i]._state + sum);
		_nodes[i]._output = sigmoid(_nodes[i]._state + _nodes[i]._bias);
	}

	for (size_t i = 0; i < _nodes.size(); i++)
		_nodes[i]._prevOutput = _nodes[i]._output;

	// Hebbian
	for (size_t i = 0; i < _nodes.size(); i++) {
		for (size_t j = 0; j < _nodes.size(); j++) {
			Weight &w = getWeight(i, j);

			w._trace += -traceDecay * w._trace + w._weight * _nodes[j]._output * (-1.0f + _nodes[i]._output) + (1.0f - w._weight) * _nodes[j]._output * _nodes[i]._output;
		}

		for (size_t j = 0; j < _inputs.size(); j++) {
			Weight &w = getWeight(i, j + _numNodesHiddenOutput);

			w._trace += -traceDecay * w._trace + w._weight * _nodes[j]._output * (-1.0f + _nodes[i]._output) + (1.0f - w._weight) * _nodes[j]._output * _nodes[i]._output;
		}
	}
}

void CTRNN::clear() {
	for (size_t i = 0; i < _nodes.size(); i++) {
		_nodes[i]._state = 0.0f;
		_nodes[i]._prevOutput = 0.0f;
		_nodes[i]._output = 0.0f;
	}

	for (size_t i = 0; i < _weightMatrix.size(); i++)
		_weightMatrix[i]._trace = 0.0f;
}