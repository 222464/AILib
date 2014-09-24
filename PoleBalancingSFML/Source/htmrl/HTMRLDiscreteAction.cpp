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

#include <htmrl/HTMRLDiscreteAction.h>

#include <iostream>

using namespace htmrl;

float htmrl::defaultBoostFunctionDiscreteAction(float active, float minimum) {
	return (1.0f - minimum) + std::max(0.0f, -(minimum - active));
}

HTMRLDiscreteAction::HTMRLDiscreteAction()
: _encodeBlobRadius(1), _replaySampleFrames(3), _maxReplayChainSize(400),
_backpropPassesCritic(200),
_prevMaxQAction(0), _prevChooseAction(0)
{}

void HTMRLDiscreteAction::createRandom(int inputWidth, int inputHeight, int inputDotsWidth, int inputDotsHeight, int numOutputs, int criticNumHiddenLayers, int criticNumNodesPerHiddenLayer, float criticInitWeightStdDev, const std::vector<RegionDesc> &regionDescs, std::mt19937 &generator) {
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

	_critic.createRandom(stateSize, numOutputs, criticNumHiddenLayers, criticNumNodesPerHiddenLayer, criticInitWeightStdDev, generator);

	_prevLayerInputb.clear();
	_prevLayerInputb.assign(stateSize, false);

	_prevQValues.clear();
	_prevQValues.assign(_critic.getNumOutputs(), 0.0f);
}

void HTMRLDiscreteAction::decodeInput() {
	for (int i = 0; i < _inputb.size(); i++)
		_inputb[i] = false;

	int inputBWidth = _inputWidth * _inputDotsWidth;

	for (int x = 0; x < _inputWidth; x++)
	for (int y = 0; y < _inputHeight; y++) {
		int bX = x * _inputDotsWidth;
		int bY = y * _inputDotsHeight;

		int eX = bX + _inputDotsWidth;
		int eY = bY + _inputDotsHeight;

		int numDotsX = static_cast<int>((_inputf[x + y * _inputWidth + 0 * _inputWidth * _inputHeight] * 0.5f + 0.5f) * _inputDotsWidth);
		int numDotsY = static_cast<int>((_inputf[x + y * _inputWidth + 1 * _inputWidth * _inputHeight] * 0.5f + 0.5f) * _inputDotsHeight);

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

int HTMRLDiscreteAction::step(float reward, float backpropAlphaCritic, float momentumCritic, float gamma, float lambda, float tauInv, float epsilon, float weightDecayMultiplier, std::mt19937 &generator) {
	_critic.decayWeights(weightDecayMultiplier);
	
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

	// Get maximum Q prediction from critic
	for (int i = 0; i < layerInputf.size(); i++)
		layerInputf[i] = layerInput[i] ? 1.0f : 0.0f;

	std::vector<float> criticOutput(_critic.getNumOutputs());

	_critic.process(layerInputf, criticOutput);

	int maxQActionIndex = 0;

	for (int i = 1; i < _critic.getNumOutputs(); i++) {
		if (criticOutput[i] > criticOutput[maxQActionIndex])
			maxQActionIndex = i;
	}

	float nextMaxQ = criticOutput[maxQActionIndex];

	// Generate exploratory action
	std::uniform_real_distribution<float> uniformDist(0.0f, 1.0f);
	
	int choosenAction;

	if (uniformDist(generator) < epsilon) {
		std::uniform_int_distribution<int> actionDist(0, _critic.getNumOutputs() - 1);

		choosenAction = actionDist(generator);
	}
	else
		choosenAction = maxQActionIndex;

	float nextValue = criticOutput[choosenAction];

	float newAdv = reward + gamma * nextMaxQ;

	float errorCritic = lambda * (newAdv - _prevQValues[_prevChooseAction]);

	// Add sample to chain
	ReplaySample sample;
	sample._actorInputsb = _prevLayerInputb;

	sample._actionExploratory = _prevChooseAction;
	sample._actionOptimal = _prevMaxQAction;

	sample._reward = reward;

	sample._actionQValues = _prevQValues;

	sample._actionQValues[_prevChooseAction] += errorCritic;

	float prevV = sample._actionQValues[_prevMaxQAction];

	for (std::list<ReplaySample>::iterator it = _replayChain.begin(); it != _replayChain.end(); it++) {
		float value = it->_reward + gamma * prevV;
		float error = lambda * (value - it->_actionQValues[it->_actionExploratory]);

		it->_actionQValues[it->_actionExploratory] += error;

		prevV = it->_actionQValues[it->_actionOptimal];
	}

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

	std::vector<float> inputf(_critic.getNumInputs());
	std::vector<float> criticTempOutput(_critic.getNumOutputs());

	for (int s = 0; s < _backpropPassesCritic; s++) {
		int replayIndex = sampleDist(generator);

		ReplaySample* pSample = pReplaySamples[replayIndex];

		for (int i = 0; i < _critic.getNumInputs(); i++)
			inputf[i] = pSample->_actorInputsb[i] ? 1.0f : 0.0f;

		_critic.process(inputf, criticTempOutput);

		_critic.backpropagate(inputf, pSample->_actionQValues, backpropAlphaCritic, momentumCritic);
	}

	_prevMaxQAction = maxQActionIndex;
	_prevChooseAction = choosenAction;

	_critic.process(layerInputf, criticOutput);

	_prevQValues = criticOutput;

	_prevLayerInputb = layerInput;

	std::cout << errorCritic << " " << _prevQValues[_prevMaxQAction] << " " << " " << _prevQValues[_prevChooseAction] << std::endl;

	return choosenAction;
}