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

#include <htmrl/HTMRL.h>

#include <iostream>

using namespace htmrl;

float htmrl::defaultBoostFunction(float active, float minimum) {
	return (1.0f - minimum) + std::max(0.0f, -(minimum - active));
}

HTMRL::HTMRL()
: _encodeBlobRadius(1), _replaySampleFrames(3), _maxReplayChainSize(300),
_backpropPassesActor(100),
_backpropPassesCritic(100),
_prevMaxQ(0.0f), _prevValue(0.0f), _actionInputVocalness(10.0f),
_variance(0.0f)
{}

void HTMRL::createRandom(int inputWidth, int inputHeight, int inputDotsWidth, int inputDotsHeight, int numOutputs, int actorNumHiddenLayers, int actorNumNodesPerHiddenLayer, int criticNumHiddenLayers, int criticNumNodesPerHiddenLayer, float actorCriticInitWeightStdDev, const std::vector<RegionDesc> &regionDescs, std::mt19937 &generator) {
	_inputWidth = inputWidth;
	_inputHeight = inputHeight;

	_inputDotsWidth = inputDotsWidth;
	_inputDotsHeight = inputDotsHeight;

	_inputMax = _inputDotsWidth * _inputDotsHeight;

	_regionDescs = regionDescs;

	_inputf.resize(_inputWidth * _inputHeight * 2);
	_inputb.resize(_inputWidth * _inputHeight * _inputMax);

	for (int i = 0; i < _inputf.size(); i++)
		_inputf[i] = 0.0f;

	int dotsWidth = _inputWidth * _inputDotsWidth;
	int dotsHeight = _inputHeight * _inputDotsHeight;

	_regions.resize(_regionDescs.size());

	for (int i = 0; i < _regions.size(); i++) {
		_regions[i].createRandom(dotsWidth, dotsHeight, _regionDescs[i]._connectionRadius, _regionDescs[i]._initInhibitionRadius,
			_regionDescs[i]._initNumSegments, _regionDescs[i]._regionWidth, _regionDescs[i]._regionHeight, _regionDescs[i]._columnSize,
			_regionDescs[i]._permanenceDistanceBias, _regionDescs[i]._permanenceDistanceFalloff, _regionDescs[i]._permanenceBiasFloor,
			_regionDescs[i]._connectionPermanenceTarget, _regionDescs[i]._connectionPermanenceStdDev, generator);

		dotsWidth = _regionDescs[i]._regionWidth;
		dotsHeight = _regionDescs[i]._regionHeight;
	}

	int stateSize = dotsWidth * dotsHeight;

	_actor.createRandom(stateSize, numOutputs, actorNumHiddenLayers, actorNumNodesPerHiddenLayer, actorCriticInitWeightStdDev, generator);
	_critic.createRandom(stateSize + numOutputs, 1, criticNumHiddenLayers, criticNumNodesPerHiddenLayer, actorCriticInitWeightStdDev, generator);

	_outputs.clear();
	_outputs.assign(numOutputs, 0.0f);

	_exploratoryOutputs.clear();
	_exploratoryOutputs.assign(numOutputs, 0.0f);

	_prevOutputs.clear();
	_prevOutputs.assign(numOutputs, 0.0f);

	_prevExploratoryOutputs.clear();
	_prevExploratoryOutputs.assign(numOutputs, 0.0f);

	_prevLayerInputb.clear();
	_prevLayerInputb.assign(_regions.back().getRegionWidth() * _regions.back().getRegionHeight(), false);
}

void HTMRL::decodeInput() {
	for (int i = 0; i < _inputb.size(); i++)
		_inputb[i] = false;

	int inputBWidth = _inputWidth * _inputDotsWidth;

	for (int x = 0; x < _inputWidth; x++)
	for (int y = 0; y < _inputHeight; y++) {
		int bX = x * _inputDotsWidth;
		int bY = y * _inputDotsHeight;

		int eX = bX + _inputDotsWidth;
		int eY = bY + _inputDotsHeight;

		int numDotsX = static_cast<int>((_inputf[x + y * _inputWidth + 0 * _inputWidth * 2] * 0.5f + 0.5f) * _inputDotsWidth);
		int numDotsY = static_cast<int>((_inputf[x + y * _inputWidth + 1 * _inputWidth * 2] * 0.5f + 0.5f) * _inputDotsHeight);

		int dotX = bX + numDotsX;
		int dotY = bY + numDotsY;

		for (int dx = -_encodeBlobRadius; dx <= _encodeBlobRadius; dx++)
		for (int dy = -_encodeBlobRadius; dy <= _encodeBlobRadius; dy++) {
			int pX = dotX + dx;
			int pY = dotY + dy;

			if (pX >= bX && pX < eX && pY >= bY && pY < eY)
				_inputb[pX + pY * inputBWidth] = true;
		}
	}
}

void HTMRL::step(float reward, float backpropAlphaActor, float backpropAlphaCritic, float alphaActor, float alphaCritic, float momentumActor, float momentumCritic, float gamma, float lambda, float tauInv, float perturbationStdDev, float breakRate, float policySearchStdDev, float actionMomentum, float varianceDecay, std::mt19937 &generator) {
	decodeInput();

	std::vector<bool> layerInput = _inputb;

	for (int i = 0; i < _regions.size(); i++) {
		_regions[i].stepBegin();

		_regions[i].spatialPooling(layerInput, _regionDescs[i]._minPermanence, _regionDescs[i]._minOverlap, _regionDescs[i]._desiredLocalActivity,
			_regionDescs[i]._spatialPermanenceIncrease, _regionDescs[i]._spatialPermanenceDecrease, _regionDescs[i]._minDutyCycleRatio, _regionDescs[i]._activeDutyCycleDecay,
			_regionDescs[i]._overlapDutyCycleDecay, _regionDescs[i]._subOverlapPermanenceIncrease, _regionDescs[i]._boostFunction);

		_regions[i].temporalPoolingLearn(_regionDescs[i]._minPermanence, _regionDescs[i]._learningRadius, _regionDescs[i]._minLearningThreshold,
			_regionDescs[i]._activationThreshold, _regionDescs[i]._newNumConnections, _regionDescs[i]._temporalPermanenceIncrease,
			_regionDescs[i]._temporalPermanenceDecrease, _regionDescs[i]._newConnectionPermanence, _regionDescs[i]._maxSteps, generator);

		layerInput.resize(_regions[i].getRegionWidth() * _regions[i].getRegionHeight());

		for (int x = 0; x < _regions[i].getRegionWidth(); x++)
		for (int y = 0; y < _regions[i].getRegionHeight(); y++)
			layerInput[x + y * _regions[i].getRegionWidth()] = _regions[i].getOutput(x, y);
	}

	std::vector<float> layerInputf(layerInput.size());

	for (int i = 0; i < layerInputf.size(); i++)
		layerInputf[i] = layerInput[i] ? 1.0f : 0.0f;

	// Find maxmimum action
	_actor.process(layerInputf, _outputs);

	// Get maximum Q prediction from critic
	std::vector<float> criticInput(_critic.getNumInputs());
	std::vector<float> criticOutput(1);

	for (int i = 0; i < layerInputf.size(); i++)
		criticInput[i] = layerInputf[i];

	for (int i = 0; i < _actor.getNumOutputs(); i++)
		criticInput[i + layerInputf.size()] = std::min(1.0f, std::max(-1.0f, _outputs[i])) * _actionInputVocalness;

	_critic.process(criticInput, criticOutput);

	float nextMaxQ = criticOutput[0];

	// Generate exploratory action
	std::uniform_real_distribution<float> uniformDist(0.0f, 1.0f);
	std::normal_distribution<float> perturbationDist(0.0f, perturbationStdDev);

	for (int i = 0; i < _exploratoryOutputs.size(); i++)
	if (uniformDist(generator) < breakRate)
		_exploratoryOutputs[i] = uniformDist(generator) * 2.0f - 1.0f;
	else
		_exploratoryOutputs[i] = std::min(1.0f, std::max(-1.0f, std::min(1.0f, std::max(-1.0f, _outputs[i])) + perturbationDist(generator)));

	// Q at exploratory action
	for (int i = 0; i < _actor.getNumOutputs(); i++)
		criticInput[i + layerInputf.size()] = _exploratoryOutputs[i] * _actionInputVocalness;

	_critic.process(criticInput, criticOutput);

	float nextValue = criticOutput[0];

	float newAdv = _prevMaxQ + (reward + gamma * nextMaxQ - _prevMaxQ) * tauInv;

	float errorCritic = lambda * (newAdv - _prevValue);

	// Update previous samples
	float prevV = std::max(_prevValue + errorCritic, _prevMaxQ);

	for (std::list<ReplaySample>::iterator it = _replayChain.begin(); it != _replayChain.end(); it++) {
		it->_criticOutput = (1.0f - lambda) * it->_criticOutput + lambda * (it->_criticOutput + (it->_reward + gamma * prevV - it->_criticOutput) * tauInv);

		prevV = it->_optimalQ;
	}

	// Add sample to chain
	ReplaySample sample;
	sample._actorInputsb = _prevLayerInputb;

	sample._actorOutputsExploratory.resize(_actor.getNumOutputs());

	for (int i = 0; i < _actor.getNumOutputs(); i++)
		sample._actorOutputsExploratory[i] = std::min(1.0f, std::max(-1.0f, _prevExploratoryOutputs[i]));

	sample._actorOutputsOptimal.assign(_actor.getNumOutputs(), 0.0f);

	sample._criticOutput = _prevValue + errorCritic;

	sample._reward = reward;

	sample._prevDAction.assign(_actor.getNumOutputs(), 0.0f);

	sample._optimalQ = std::max(_prevValue + errorCritic, _prevMaxQ);
	sample._exploratoryQ = sample._criticOutput;

	_replayChain.push_front(sample);

	while (_replayChain.size() > _maxReplayChainSize)
		_replayChain.pop_back();

	// Get random access to samples
	std::vector<ReplaySample*> pReplaySamples(_replayChain.size());

	int index = 0;

	for (std::list<ReplaySample>::iterator it = _replayChain.begin(); it != _replayChain.end(); it++, index++)
		pReplaySamples[index] = &(*it);

	// Rehearse
	std::uniform_int_distribution<int> sampleDist(0, pReplaySamples.size() - 1);

	std::vector<float> actorOutputs(_actor.getNumOutputs());
	std::vector<float> inputf(_actor.getNumInputs());
	std::vector<float> inputWithActionf(_actor.getNumInputs() + _actor.getNumOutputs());
	//std::vector<float> actorOutputOptimal(_actor.getNumOutputs());
	std::vector<float> actorOutputsExploratory(_actor.getNumOutputs());

	int numActionsKept = 0;

	for (int s = 0; s < _backpropPassesCritic; s++) {
		int replayIndex = sampleDist(generator);

		float sampleImportance = 1.0f;// std::pow(gamma, replayIndex);

		ReplaySample* pSample = pReplaySamples[replayIndex];

		for (int i = 0; i < _actor.getNumInputs(); i++)
			inputWithActionf[i] = inputf[i] = pSample->_actorInputsb[i] ? 1.0f : 0.0f;

		for (int i = 0; i < _actor.getNumOutputs(); i++)
			inputWithActionf[i + _actor.getNumInputs()] = pSample->_actorOutputsExploratory[i] * _actionInputVocalness;

		_critic.process(inputWithActionf, criticOutput);

		_critic.backpropagate(inputWithActionf, std::vector<float>(1, pSample->_criticOutput), backpropAlphaCritic * sampleImportance, momentumCritic);
	}

	std::normal_distribution<float> policySearchDist(0.0f, policySearchStdDev);

	for (int s = 0; s < _backpropPassesActor; s++) {
		int replayIndex = sampleDist(generator);

		float sampleImportance = 1.0f;// std::pow(gamma, replayIndex);

		ReplaySample* pSample = pReplaySamples[replayIndex];

		for (int i = 0; i < _actor.getNumInputs(); i++)
			inputWithActionf[i] = inputf[i] = pSample->_actorInputsb[i] ? 1.0f : 0.0f;

		// Get action we now think is optimal
		_actor.process(inputf, pSample->_actorOutputsOptimal);

		// Clamp action
		for (int i = 0; i < _actor.getNumOutputs(); i++)
			pSample->_actorOutputsOptimal[i] = std::min(1.0f, std::max(-1.0f, pSample->_actorOutputsOptimal[i]));

		// Get Q at action we think is optimal and at exploratory action
		for (int i = 0; i < _actor.getNumOutputs(); i++)
			inputWithActionf[i + _actor.getNumInputs()] = pSample->_actorOutputsOptimal[i] * _actionInputVocalness;
		
		_critic.process(inputWithActionf, criticOutput);

		pSample->_optimalQ = criticOutput[0];

		for (int i = 0; i < _actor.getNumOutputs(); i++)
			inputWithActionf[i + _actor.getNumInputs()] = (actorOutputsExploratory[i] = std::min(1.0f, std::max(-1.0f, std::min(1.0f, std::max(-1.0f, pSample->_actorOutputsOptimal[i])) + policySearchDist(generator) + pSample->_prevDAction[i] * actionMomentum))) * _actionInputVocalness;

		_critic.process(inputWithActionf, criticOutput);

		float exploratoryQ = criticOutput[0];

		if (exploratoryQ > pSample->_optimalQ) {
			_actor.backpropagate(inputf, actorOutputsExploratory, backpropAlphaActor * sampleImportance, momentumActor);

			for (int i = 0; i < _actor.getNumOutputs(); i++)
				pSample->_prevDAction[i] = actorOutputsExploratory[i] - pSample->_actorOutputsOptimal[i];

			numActionsKept++;
		}
		else {
			_actor.backpropagate(inputf, pSample->_actorOutputsOptimal, backpropAlphaActor * sampleImportance, momentumActor);

			for (int i = 0; i < _actor.getNumOutputs(); i++)
				pSample->_prevDAction[i] = 0.0f;
		}
	}

	_prevLayerInputb = layerInput;
	_prevOutputs = _outputs;
	_prevExploratoryOutputs = _exploratoryOutputs;

	_prevMaxQ = nextMaxQ;
	_prevValue = nextValue;

	_variance = (1.0f - varianceDecay) * _variance + varianceDecay * std::abs(errorCritic);

	std::cout << errorCritic << " " << newAdv << " " << " " << _prevValue << " " << _outputs[0] << " " << numActionsKept << std::endl;
}