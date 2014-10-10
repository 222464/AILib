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
: _encodeBlobRadius(0), _replaySampleFrames(3), _maxReplayChainSize(800),
_backpropPassesCritic(40), _minibatchSize(8),
_prevMaxQAction(0), _prevChooseAction(0),
_earlyStopError(0.0f), _averageAbsError(0.0f)
{}

void HTMRLDiscreteAction::createRandom(int inputWidth, int inputHeight, int inputDotsWidth, int inputDotsHeight, int condenseWidth, int condenseHeight, int numOutputs, int criticNumHidden, int criticNumPerHidden, float criticInitWeightStdDev, const std::vector<RegionDesc> &regionDescs, std::mt19937 &generator) {
	_inputWidth = inputWidth;
	_inputHeight = inputHeight;

	_inputDotsWidth = inputDotsWidth;
	_inputDotsHeight = inputDotsHeight;

	_condenseWidth = condenseWidth;
	_condenseHeight = condenseHeight;

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

	_condenseBufferWidth = std::ceil(static_cast<float>(dotsWidth) / _condenseWidth);
	_condenseBufferHeight = std::ceil(static_cast<float>(dotsHeight) / _condenseHeight);

	int stateSize = _condenseBufferWidth * _condenseBufferHeight;

	_inputCond.clear();
	_inputCond.assign(stateSize, 0.0f);

	_critic.createRandom(stateSize, numOutputs, criticNumHidden, criticNumPerHidden, criticInitWeightStdDev, generator);

	_prevLayerInputf.clear();
	_prevLayerInputf.assign(stateSize, false);

	_prevQValues.clear();
	_prevQValues.assign(numOutputs, 0.0f);
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

int HTMRLDiscreteAction::step(float reward, float qAlpha, float criticRMSDecay, float criticGradientAlpha, float criticGradientMomentum, float gamma, float lambda, float tauInv, float epsilon, float softmaxT, float kOut, float kHidden, float averageAbsErrorDecay, std::mt19937 &generator, std::vector<float> &condensed) {
	decodeInput();

	std::vector<bool> layerInput = _inputb;

	int dotsWidth = _inputWidth * _inputDotsWidth;
	int dotsHeight = _inputHeight * _inputDotsHeight;

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

		dotsWidth = _regionDescs[i]._regionWidth;
		dotsHeight = _regionDescs[i]._regionHeight;
	}

	// Condense
	std::vector<float> condensedInputf(_condenseBufferWidth * _condenseBufferHeight);

	float maxInv = 1.0f / (_condenseWidth * _condenseHeight);
	
	for (int x = 0; x < _condenseBufferWidth; x++)
	for (int y = 0; y < _condenseBufferHeight; y++) {

		float sum = 0.0f;

		for (int dx = 0; dx < _condenseWidth; dx++)
		for (int dy = 0; dy < _condenseHeight; dy++) {
			int bX = x * _condenseWidth + dx;
			int bY = y * _condenseHeight + dy;

			if (bX >= 0 && bX < dotsWidth && bY >= 0 && bY < dotsHeight)
				sum += (layerInput[bX + bY * dotsWidth] ? 1.0f : 0.0f);
		}

		sum *= maxInv;

		condensedInputf[x + y * _condenseBufferWidth] = sum;
	}

	condensed = condensedInputf;

	// Get maximum Q prediction from critics
	std::vector<float> criticOutputs(_critic.getNumOutputs());

	_critic.process(condensedInputf, criticOutputs);

	std::vector<float> originalOutputs = criticOutputs;

	int maxQActionIndex = 0;

	for (int i = 1; i < _critic.getNumOutputs(); i++) {
		if (criticOutputs[i] > criticOutputs[maxQActionIndex])
			maxQActionIndex = i;
	}

	int choosenAction;

	// Generate exploratory action
	std::uniform_real_distribution<float> uniformDist(0.0f, 1.0f);

	if (uniformDist(generator) < epsilon) {
		// Use softmax
		std::vector<float> softmaxValues(_critic.getNumOutputs());

		float sum = 0.0f;

		for (int i = 0; i < softmaxValues.size(); i++) {
			softmaxValues[i] = std::exp(criticOutputs[i] * softmaxT);

			sum += softmaxValues[i];
		}

		float sumSoFar = 0.0f;

		float cusp = uniformDist(generator) * sum;

		choosenAction = 0;

		for (; choosenAction < _critic.getNumOutputs() - 1; choosenAction++) {
			sumSoFar += softmaxValues[choosenAction];

			if (sumSoFar >= cusp)
				break;
		}
	}
	else
		choosenAction = maxQActionIndex;

	float newAdv = _prevQValues[_prevMaxQAction] + (reward + gamma * criticOutputs[choosenAction] - _prevQValues[_prevMaxQAction]) * tauInv;

	float errorCritic = newAdv - _prevQValues[_prevChooseAction];

	_averageAbsError = (1.0f - averageAbsErrorDecay) * _averageAbsError + averageAbsErrorDecay * std::abs(errorCritic);

	// Add sample to chain
	ReplaySample sample;
	sample._inputs = _prevLayerInputf;

	sample._actionExploratory = _prevChooseAction;
	sample._actionOptimal = _prevMaxQAction;

	sample._reward = reward;

	sample._actionQValues = _prevQValues;

	sample._actionQValues[_prevChooseAction] += qAlpha * errorCritic;

	sample._actionOptimal = 0;

	for (int i = 1; i < sample._actionQValues.size(); i++)
	if (sample._actionQValues[i] > sample._actionQValues[sample._actionOptimal])
		sample._actionOptimal = i;

	//float prevV = sample._actionQValues[sample._actionExploratory];

	float g = gamma;

	for (std::list<ReplaySample>::iterator it = _replayChain.begin(); it != _replayChain.end(); it++) {
		//float value = it->_actionQValues[it->_actionOptimal] + (it->_reward + gamma * prevV - it->_actionQValues[it->_actionOptimal]) * tauInv;
		//float error = (value - it->_actionQValues[it->_actionExploratory]);

		it->_actionQValues[it->_actionExploratory] += qAlpha * g * errorCritic;

		//it->_actionOptimal = 0;

		for (int i = 1; i < it->_actionQValues.size(); i++)
		if (it->_actionQValues[i] > it->_actionQValues[it->_actionOptimal])
			it->_actionOptimal = i;

		//prevV = it->_actionQValues[it->_actionExploratory];

		g *= gamma;
	}

	_replayChain.push_front(sample);

	while (_replayChain.size() > _maxReplayChainSize)
		_replayChain.pop_back();

	std::vector<ReplaySample*> pReplaySamples(_replayChain.size());

	int index = 0;

	for (std::list<ReplaySample>::iterator it = _replayChain.begin(); it != _replayChain.end(); it++, index++)
		pReplaySamples[index] = &(*it);

	// Rehearse
	std::uniform_int_distribution<int> sampleDist(0, pReplaySamples.size() - 1);

	float minibatchInv = 1.0f / _minibatchSize;

	//_critic.learnFeatures(condensedInputf, criticCenterAlpha, criticWidthAlpha, criticWidthScalar);

	for (int s = 0; s < _backpropPassesCritic; s++) {
		_critic.clearGradient();

		for (int b = 0; b < _minibatchSize; b++) {
			int r = sampleDist(generator);

			ReplaySample* pSample = pReplaySamples[r];

			_critic.process(pSample->_inputs, criticOutputs);

			float error = pSample->_actionQValues[pSample->_actionExploratory] - criticOutputs[pSample->_actionExploratory];

			if (std::abs(error) > _earlyStopError * _averageAbsError) {
				std::vector<float> target = criticOutputs;

				target[pSample->_actionExploratory] = pSample->_actionQValues[pSample->_actionExploratory];

				_critic.accumulateGradient(pSample->_inputs, target);
			}
		}

		_critic.scaleGradient(minibatchInv);

		_critic.moveAlongGradientRMS(criticRMSDecay, criticGradientAlpha, criticGradientMomentum, kOut, kHidden);
	}

	// Recompute samples
	/*for (std::list<ReplaySample>::iterator it = _replayChain.begin(); it != _replayChain.end(); it++) {
		_critic.getOutput(it->_inputs, criticOutputs);

		for (int i = 0; i < _critic.getNumOutputs(); i++)
		if (i != it->_actionExploratory)
			it->_actionQValues[i] = criticOutputs[i];

		it->_actionOptimal = 0;

		for (int i = 1; i < it->_actionQValues.size(); i++)
		if (it->_actionQValues[i] > it->_actionQValues[it->_actionOptimal])
			it->_actionOptimal = i;
	}*/

	_prevMaxQAction = maxQActionIndex;
	_prevChooseAction = choosenAction;

	_prevQValues = originalOutputs;

	_prevLayerInputf = condensedInputf;

	std::cout << errorCritic << " " << newAdv << " " << _prevQValues[_prevChooseAction] << std::endl;

	return choosenAction;
}