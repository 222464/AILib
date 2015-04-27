#include "RSARL.h"

#include <algorithm>

#include <iostream>

using namespace deep;

void RSARL::createRandom(int numInputs, int numOutputs, int numHidden, float sparsity, float minInitWeight, float maxInitWeight, float recurrentScalar, std::mt19937 &generator) {
	_numInputs = numInputs;

	_sparsity = sparsity;

	_rsa.createRandom(numInputs + numOutputs + 1, numHidden, sparsity, minInitWeight, maxInitWeight, recurrentScalar, generator);

	//_qNodes.resize(numHidden);

	_prevValue = 0.0f;

	_outputs.resize(numOutputs, 0.0f);
	_maxOutputs.resize(numOutputs, 0.0f);
}

void RSARL::dqOverDo(std::vector<float> &deltaO) {
	std::vector<float> hiddenErrors(_rsa.getNumHiddenNodes());

	for (int h = 0; h < _rsa.getNumHiddenNodes(); h++)
		hiddenErrors[h] = _rsa._hiddenNodes[h]._statePrev * (1.0f - _rsa._hiddenNodes[h]._activationPrev);

	deltaO.resize(getNumOutputs());

	for (int v = _numInputs; v < _rsa.getNumVisibleNodes(); v++) {
		float error = 0.0f;

		for (int h = 0; h < _rsa.getNumHiddenNodes(); h++)
			error += hiddenErrors[h] * _rsa._hiddenNodes[h]._visibleHiddenConnections[v]._weight;

		deltaO[v - _numInputs] = error;
	}
}

void RSARL::step(float reward, int actionSamples, int experienceSamples, float rsaStateLeak, float rsaAlpha, float rsaBeta, float rsaGamma, float rsaEpsilon, float rsaDutyCycleDecay, float rsaMomentum, float rsaTraceDecay, float rsaTemperature, float qTraceDecay, float qAlpha, float qUpdateAlpha, float qGamma, float breakChance, std::mt19937 &generator) {
	std::uniform_real_distribution<float> uniformDist(0.0f, 1.0f);

	_rsa.stepBegin();

	std::vector<float> maxInput(_rsa.getNumVisibleNodes());
	std::vector<float> input(_rsa.getNumVisibleNodes());

	for (int i = 0; i < _numInputs; i++)
		maxInput[i] = input[i] = _rsa.getVisibleNodeState(i);

	for (int i = 0; i < getNumOutputs(); i++) {
		input[_numInputs + i] = _outputs[i];
		maxInput[_numInputs + i] = _maxOutputs[i];
	}

	//input[_outputs.size()] = _prevValue;
	maxInput[_outputs.size()] = _prevValue;

	for (int i = 0; i < _outputs.size(); i++)
		_rsa.setVisibleNodeState(_numInputs + i, _outputs[i]);

	_rsa.activate(_sparsity, rsaDutyCycleDecay);

	std::vector<float> hiddenStates(_rsa.getNumHiddenNodes());

	for (int h = 0; h < hiddenStates.size(); h++)
		hiddenStates[h] = _rsa.getHiddenNodeState(h);

	std::vector<float> visibleStates(_rsa.getNumVisibleNodes());

	for (int v = 0; v < visibleStates.size(); v++)
		visibleStates[v] = _rsa.getVisibleNodeState(v);

	std::vector<float> reconstructions(_rsa.getNumVisibleNodes());

	for (int v = 0; v < reconstructions.size(); v++)
		reconstructions[v] = _rsa.getVisibleNodeReconstruction(v);

	/*for (int i = 0; i < actionSamples; i++) {
		_rsa.activate(_sparsity, rsaDutyCycleDecay);

		std::vector<float> deltaO;

		dqOverDo(deltaO);

		for (int j = 0; j < getNumOutputs(); j++) {
			int v = _numInputs + j;
			_rsa.setVisibleNodeState(v, std::min(1.0f, std::max(-1.0f, _rsa.getVisibleNodeState(v) + deltaO[j] * rsaAlpha)));
		}

		float nextQ = 0.0f;

		for (int h = 0; h < _rsa.getNumHiddenNodes(); h++)
			nextQ += _rsa.getHiddenNodeState(h) * _qNodes[h]._q;

		std::cout << nextQ << std::endl;
	}*/

	for (int i = 0; i < _outputs.size(); i++) {
		_maxOutputs[i] = std::min(1.0f, std::max(-1.0f, _rsa.getVisibleNodeReconstruction(_numInputs + i)));

		if (uniformDist(generator) < breakChance)
			_outputs[i] = uniformDist(generator) * 2.0f - 1.0f;
		else
			_outputs[i] = _maxOutputs[i];
	}

	//float nextQ = 0.0f;

	//for (int h = 0; h < _rsa.getNumHiddenNodes(); h++)
	//	nextQ += _rsa.getHiddenNodeState(h) * _qNodes[h]._q;

	float nextQ = _rsa.getVisibleNodeReconstruction(_numInputs + _outputs.size());

	float tdError = reward + qGamma * nextQ - _prevValue;

	float newQ = _prevValue + qAlpha * tdError;

	input[_numInputs + _outputs.size()] = newQ;
	maxInput[_numInputs + _outputs.size()] = _prevValue;

	// Propagate Q down chain
	float g = qGamma;

	for (std::list<Experience>::iterator it = _experiences.begin(); it != _experiences.end(); it++) {
		it->_visibleStates[_numInputs + _outputs.size()] += qAlpha * tdError * g;

		g *= qGamma;
	}

	// Push new experience
	Experience experience;
	experience._hiddenStates.resize(_rsa.getNumHiddenNodes());

	for (int h = 0; h < _rsa.getNumHiddenNodes(); h++)
		experience._hiddenStates[h] = _rsa.getHiddenNodeState(h);

	experience._maxVisibleStates = maxInput;
	experience._visibleStates = input;

	_experiences.push_front(experience);

	while (_experiences.size() > _experienceBufferLength)
		_experiences.pop_back();

	_prevValue = nextQ;

	// Transform chain to get random access
	std::vector<std::list<Experience>::iterator> expIters(_experiences.size());

	int index = 0;

	for (std::list<Experience>::iterator it = _experiences.begin(); it != _experiences.end(); it++)
		expIters[index++] = it;

	if (expIters.size() > 2) {
		std::uniform_int_distribution<int> sampleDist(0, std::max(0, static_cast<int>(expIters.size()) - 3));

		for (int i = 0; i < experienceSamples; i++) {
			int j = sampleDist(generator);

			RecurrentSparseAutoencoder::Experience rsaExp;

			if (j >= static_cast<int>(expIters.size()) - 3)
				rsaExp._hiddenStatesPrevPrev.assign(_rsa.getNumHiddenNodes(), 0.0f);
			else {
				rsaExp._hiddenStatesPrevPrev.resize(_rsa.getNumHiddenNodes());

				for (int k = 0; k < _rsa.getNumHiddenNodes(); k++)
					rsaExp._hiddenStatesPrevPrev[k] = expIters[j + 3]->_hiddenStates[k];
			}

			rsaExp._visibleStatesPrev.resize(_rsa.getNumVisibleNodes());

			for (int k = 0; k < _rsa.getNumVisibleNodes(); k++)
				rsaExp._visibleStatesPrev[k] = expIters[j + 2]->_maxVisibleStates[k];

			rsaExp._hiddenStatesPrev.resize(_rsa.getNumHiddenNodes());

			for (int k = 0; k < _rsa.getNumHiddenNodes(); k++)
				rsaExp._hiddenStatesPrev[k] = expIters[j + 2]->_hiddenStates[k];

			rsaExp._visibleStates.resize(_rsa.getNumVisibleNodes());

			if (expIters[j + 1]->_visibleStates[_numInputs + _outputs.size()] > expIters[j + 1]->_maxVisibleStates[_numInputs + _outputs.size()]) {
				for (int k = 0; k < _rsa.getNumVisibleNodes(); k++)
					rsaExp._visibleStates[k] = expIters[j]->_visibleStates[k];
			}
			else {	
				for (int k = 0; k < _rsa.getNumVisibleNodes(); k++)
					rsaExp._visibleStates[k] = expIters[j]->_maxVisibleStates[k];
			}

			_rsa.learnExperience(rsaExp, _sparsity, rsaStateLeak, rsaAlpha, rsaBeta, rsaGamma, rsaEpsilon, rsaMomentum, 1.0f, 1.0f);
		}
	}

	// Reactivate into correct state
	for (int h = 0; h < hiddenStates.size(); h++)
		_rsa.setHiddenNodeState(h, hiddenStates[h]);

	for (int v = 0; v < visibleStates.size(); v++)
		_rsa.setVisibleNodeState(v, visibleStates[v]);

	for (int v = 0; v < reconstructions.size(); v++)
		_rsa._visibleNodes[v]._reconstruction = reconstructions[v];

	std::cout << nextQ << " " << tdError << std::endl;

	//if (tdError < 0.0f) {
	//	for (int i = 0; i < _outputs.size(); i++)
	//		_rsa.setVisibleNodeState(_numInputs + i, _maxOutputs[i]);
	//}

	//_rsa.learn(_sparsity, rsaStateLeak, rsaAlpha * tdError, rsaBeta * tdError, rsaGamma, rsaEpsilon, rsaMomentum, rsaTraceDecay, rsaTemperature);
}