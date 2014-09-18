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
: _encodeBlobRadius(1)
{}

void HTMRL::createRandom(int inputWidth, int inputHeight, int inputDotsWidth, int inputDotsHeight, int numOutputs, int ferlNumHidden, float ferlInitWeightStdDev, const std::vector<RegionDesc> &regionDescs, std::mt19937 &generator) {
	_inputWidth = inputWidth;
	_inputHeight = inputHeight;

	_inputDotsWidth = inputDotsWidth;
	_inputDotsHeight = inputDotsHeight;

	_inputMax = _inputDotsWidth * _inputDotsHeight;

	_regionDescs = regionDescs;

	_inputf.resize(_inputWidth * _inputHeight);
	_inputb.resize(_inputf.size() * _inputMax);

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

		dotsWidth = _regionDescs[i]._regionWidth * _regionDescs[i]._columnSize;
		dotsHeight = _regionDescs[i]._regionHeight;
	}

	_ferl.createRandom(dotsWidth * dotsHeight, numOutputs, ferlNumHidden, ferlInitWeightStdDev, generator);

	_outputs.clear();
	_outputs.assign(numOutputs, 0.0f);
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

		int numDots = static_cast<int>((_inputf[x + y * _inputWidth] * 0.5f + 0.5f) * _inputMax);

		int dotX = bX + numDots % _inputDotsWidth;
		int dotY = bY + numDots / _inputDotsHeight;

		for (int dx = -_encodeBlobRadius; dx <= _encodeBlobRadius; dx++)
		for (int dy = -_encodeBlobRadius; dy <= _encodeBlobRadius; dy++) {
			int pX = dotX + dx;
			int pY = dotY + dy;

			if (pX >= bX && pX < eX && pY >= bY && pY < eY)
				_inputb[pX + pY * inputBWidth] = true;
		}
	}
}

void HTMRL::step(float reward, float qAlpha, float gamma, float lambdaGamma, float tauInv, int qSearchIterations, int qSearchSamples, int qSearchAlpha, float perturbationStdDev, float breakRate, std::mt19937 &generator) {
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

		layerInput.resize(_regions[i].getRegionWidth() * _regions[i].getRegionHeight() * _regionDescs[i]._columnSize);

		for (int x = 0; x < _regions[i].getRegionWidth(); x++)
		for (int y = 0; y < _regions[i].getRegionWidth(); y++)
		for (int c = 0; c < _regionDescs[i]._columnSize; c++) {
			layerInput[c + x * _regionDescs[i]._columnSize + y * _regionDescs[i]._columnSize * _regions[i].getRegionWidth()] = _regions[i].getColumn(x, y).getCell(c).getOutput();
		}
	}

	std::vector<float> layerInputf(layerInput.size());

	for (int i = 0; i < layerInputf.size(); i++)
		layerInputf[i] = layerInput[i] ? 1.0f : 0.0f;

	_ferl.step(layerInputf, _outputs, reward, qAlpha, gamma, lambdaGamma, tauInv, qSearchIterations, qSearchSamples, qSearchAlpha, breakRate, perturbationStdDev, generator);
}