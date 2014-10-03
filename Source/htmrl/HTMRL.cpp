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
: _encodeBlobRadius(1), _replaySampleFrames(3), _maxReplayChainSize(600),
_backpropPassesActor(100),
_backpropPassesCritic(100),
_approachPasses(4),
_prevMaxQ(0.0f), _prevValue(0.0f), _actionInputVocalness(10.0f)
{}

void HTMRL::createRandom(int inputWidth, int inputHeight, int inputDotsWidth, int inputDotsHeight, int condenseWidth, int condenseHeight, int numOutputs, int numHidden, float weightStdDev, const std::vector<RegionDesc> &regionDescs, std::mt19937 &generator) {
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

	_ferl.createRandom(stateSize, numOutputs, numHidden, weightStdDev, generator);

	_output.clear();
	_output.assign(numOutputs, 0.0f);
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

void HTMRL::step(float reward, float qAlpha, float gamma, float lambdaGamma, float tauInv,
	int actionSearchIterations, int actionSearchSamples, float actionSearchAlpha,
	float breakChance, float perturbationStdDev,
	int maxNumReplaySamples, int replayIterations, float gradientAlpha, float gradientMomentum,
	std::mt19937 &generator, std::vector<float> &condensed)
{
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

	_ferl.step(condensedInputf, _output,
		reward, qAlpha, gamma, lambdaGamma, tauInv,
		actionSearchIterations, actionSearchSamples, actionSearchAlpha,
		breakChance, perturbationStdDev,
		maxNumReplaySamples, replayIterations, gradientAlpha, gradientMomentum,
		generator);
}