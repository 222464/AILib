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
: _firstStep(true), _prevVariance(0.0f), _variance(0.0f), _explore(false), _prevError(0.0f)
{}

void HTMRL::createRandom(int inputWidth, int inputHeight, int inputDotsWidth, int inputDotsHeight, int numOutputs, const RegionDesc &regionDesc, float stateOutputWeightStdDev, std::mt19937 &generator) {
	_inputWidth = inputWidth;
	_inputHeight = inputHeight;

	_inputDotsWidth = inputDotsWidth;
	_inputDotsHeight = inputDotsHeight;

	_inputMax = _inputDotsWidth * _inputDotsHeight;

	_inputf.resize(_inputWidth * _inputHeight);
	_inputb.resize(_inputf.size() * _inputMax);

	for (int i = 0; i < _inputf.size(); i++)
		_inputf[i] = 0.0f;

	_region.createRandom(_inputWidth * _inputDotsWidth, _inputHeight * _inputDotsHeight, regionDesc._connectionRadius, regionDesc._initInhibitionRadius,
		regionDesc._initNumSegments, regionDesc._regionWidth, regionDesc._regionHeight, regionDesc._columnSize,
		regionDesc._permanenceDistanceBias, regionDesc._permanenceDistanceFalloff, regionDesc._permanenceBiasFloor,
		regionDesc._connectionPermanenceTarget, regionDesc._connectionPermanenceStdDev, generator);

	// Initialize FFNNs
	std::normal_distribution<float> weightDist(0.0f, stateOutputWeightStdDev);

	int numColumns = regionDesc._regionWidth * regionDesc._regionHeight;

	_criticOutput._cellWeights.resize(numColumns * regionDesc._columnSize);

	for (int i = 0; i < _criticOutput._cellWeights.size(); i++)
		_criticOutput._cellWeights[i] = weightDist(generator);

	_criticOutput._columnWeights.resize(numColumns);
	_criticOutput._columnBiases.resize(numColumns);
	_criticOutput._columnPrevPrevOutputs.resize(numColumns);
	_criticOutput._columnPrevOutputs.resize(numColumns);
	_criticOutput._columnOutputs.resize(numColumns);
	_criticOutput._columnErrors.resize(numColumns);

	for (int i = 0; i < numColumns; i++) {
		_criticOutput._columnWeights[i] = weightDist(generator);
		_criticOutput._columnBiases[i] = weightDist(generator);
		_criticOutput._columnPrevPrevOutputs[i] = 0.0f;
		_criticOutput._columnPrevOutputs[i] = 0.0f;
		_criticOutput._columnOutputs[i] = 0.0f;
		_criticOutput._columnErrors[i] = 0.0f;
	}

	_criticOutput._prevPrevOutput = 0.0f;
	_criticOutput._prevOutput = 0.0f;
	_criticOutput._output = 0.0f;
	_criticOutput._bias = weightDist(generator);
	_criticOutput._outputError = 0.0f;

	_actorOutputs.resize(numOutputs);

	for (int a = 0; a < numOutputs; a++) {
		_actorOutputs[a]._cellWeights.resize(numColumns * regionDesc._columnSize);

		for (int i = 0; i < _actorOutputs[a]._cellWeights.size(); i++)
			_actorOutputs[a]._cellWeights[i] = weightDist(generator);

		_actorOutputs[a]._columnWeights.resize(numColumns);
		_actorOutputs[a]._columnBiases.resize(numColumns);
		_actorOutputs[a]._columnPrevPrevOutputs.resize(numColumns);
		_actorOutputs[a]._columnPrevOutputs.resize(numColumns);
		_actorOutputs[a]._columnOutputs.resize(numColumns);
		_actorOutputs[a]._columnErrors.resize(numColumns);

		for (int i = 0; i < numColumns; i++) {
			_actorOutputs[a]._columnWeights[i] = weightDist(generator);
			_actorOutputs[a]._columnBiases[i] = weightDist(generator);
			_actorOutputs[a]._columnPrevPrevOutputs[i] = 0.0f;
			_actorOutputs[a]._columnPrevOutputs[i] = 0.0f;
			_actorOutputs[a]._columnOutputs[i] = 0.0f;
			_actorOutputs[a]._columnErrors[i] = 0.0f;
		}

		_actorOutputs[a]._prevPrevOutput = 0.0f;
		_actorOutputs[a]._prevOutput = 0.0f;
		_actorOutputs[a]._output = 0.0f;
		_actorOutputs[a]._bias = weightDist(generator);
		_actorOutputs[a]._outputError = 0.0f;
	}

	_prevPrevState.resize(numColumns * regionDesc._columnSize);
	_prevState.resize(numColumns * regionDesc._columnSize);

	for (int i = 0; i < _prevState.size(); i++) {
		_prevPrevState[i] = false;
		_prevState[i] = false;
	}
	
	_prevPrevOutputs.assign(numOutputs, 0.0f);
	_prevOutputs.assign(numOutputs, 0.0f);
	_outputs.assign(numOutputs, 0.0f);
	_outputOffsets.assign(numOutputs, 0.0f);
}

void HTMRL::decodeInput() {
	int inputBWidth = _inputWidth * _inputDotsWidth;

	for (int x = 0; x < _inputWidth; x++)
	for (int y = 0; y < _inputHeight; y++) {
		int bX = x * _inputDotsWidth;
		int bY = y * _inputDotsHeight;

		int numDots = static_cast<int>((_inputf[x + y * _inputWidth] * 0.5f + 0.5f) * _inputMax);

		int dotCount = 0;

		for (int dx = 0; dx < _inputDotsWidth; dx++)
		for (int dy = 0; dy < _inputDotsHeight; dy++) {
			_inputb[(bX + dx) + (bY + dy) * inputBWidth] = dotCount < numDots;

			dotCount++;
		}
	}
}

void HTMRL::step(float reward, float gamma, float qAlpha, float hebbianAlphaActor, float backpropAlphaCritic, RegionDesc &regionDesc, float outputPerturbationStdDev, float breakRate, float exploreErrorTolerance, std::mt19937 &generator) {
	// ------------------------------------- Discretize State -------------------------------------

	decodeInput();

	_region.stepBegin();

	_region.spatialPooling(_inputb, regionDesc._minPermanence, regionDesc._minOverlap, regionDesc._desiredLocalActivity,
		regionDesc._spatialPermanenceIncrease, regionDesc._spatialPermanenceDecrease, regionDesc._minDutyCycleRatio, regionDesc._activeDutyCycleDecay,
		regionDesc._overlapDutyCycleDecay, regionDesc._subOverlapPermanenceIncrease, regionDesc._boostFunction);

	_region.temporalPoolingLearn(regionDesc._minPermanence, regionDesc._learningRadius, regionDesc._minLearningThreshold,
		regionDesc._activationThreshold, regionDesc._newNumConnections, regionDesc._temporalPermanenceIncrease,
		regionDesc._temporalPermanenceDecrease, regionDesc._newConnectionPermanence, regionDesc._maxSteps, generator);

	// ----------------------------------------- Get Next Q -----------------------------------------

	int numColumns = regionDesc._regionWidth * regionDesc._regionHeight;

	float columnSum = _criticOutput._bias;

	for (int i = 0; i < numColumns; i++) {
		int weightOffset = i * regionDesc._columnSize;

		float sum = _criticOutput._columnBiases[i];

		for (int j = 0; j < regionDesc._columnSize; j++)
			sum += _criticOutput._cellWeights[j + weightOffset] * (_region.getColumn(i).getCell(j).isActive() ? 1.0f : -1.0f);

		_criticOutput._columnOutputs[i] = sigmoid(sum);

		columnSum += _criticOutput._columnWeights[i] * _criticOutput._columnOutputs[i];
	}

	float nextQ = _criticOutput._output = columnSum;

	float newQ = reward + gamma * nextQ;

	float tdError;

	if (_firstStep) {
		tdError = 0.0f;

		_firstStep = false;
	}
	else {
		tdError = newQ - _prevNextQ;

		// Backpropagate previous state
		float updateQ = _prevNextQ + qAlpha * tdError;

		_criticOutput._outputError = updateQ - _criticOutput._prevOutput;

		for (int i = 0; i < numColumns; i++)
			_criticOutput._columnErrors[i] = _criticOutput._outputError * _criticOutput._columnWeights[i] * _criticOutput._columnPrevOutputs[i] * (1.0f - _criticOutput._columnPrevOutputs[i]);

		// Move along gradient
		_criticOutput._bias += backpropAlphaCritic * _criticOutput._outputError;

		for (int i = 0; i < numColumns; i++) {
			_criticOutput._columnWeights[i] += backpropAlphaCritic * _criticOutput._outputError * _criticOutput._columnPrevOutputs[i];

			_criticOutput._columnBiases[i] += backpropAlphaCritic * _criticOutput._columnErrors[i];

			for (int j = 0; j < regionDesc._columnSize; j++) {
				int cellIndex = j + i * regionDesc._columnSize;
				_criticOutput._cellWeights[cellIndex] += backpropAlphaCritic * _criticOutput._columnErrors[i] * (_prevState[cellIndex] ? 1.0f : -1.0f);
			}
		}
	}

	float alphaError = hebbianAlphaActor * tdError;

	for (int a = 0; a < _actorOutputs.size(); a++) {
		// Move along gradient
		_actorOutputs[a]._bias += alphaError * _prevOutputs[a];

		for (int i = 0; i < numColumns; i++) {
			_actorOutputs[a]._columnWeights[i] += alphaError * _prevOutputs[a] * _actorOutputs[a]._columnPrevOutputs[i];

			_actorOutputs[a]._columnBiases[i] += alphaError * (_actorOutputs[a]._columnPrevOutputs[i] * 2.0f - 1.0f);

			for (int j = 0; j < regionDesc._columnSize; j++) {
				int cellIndex = j + i * regionDesc._columnSize;
				_actorOutputs[a]._cellWeights[cellIndex] += alphaError * (_actorOutputs[a]._columnPrevOutputs[i] * 2.0f - 1.0f) * (_prevState[cellIndex] ? 1.0f : -1.0f);
			}
		}
	}

	// Find new output
	std::normal_distribution<float> outputPertDist(0.0f, outputPerturbationStdDev);
	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

	for (int a = 0; a < _actorOutputs.size(); a++) {
		columnSum = _actorOutputs[a]._bias;

		for (int i = 0; i < numColumns; i++) {
			int weightOffset = i * regionDesc._columnSize;

			float sum = _actorOutputs[a]._columnBiases[i];

			for (int j = 0; j < regionDesc._columnSize; j++)
				sum += _actorOutputs[a]._cellWeights[j + weightOffset] * (_region.getColumn(i).getCell(j).isActive() ? 1.0f : -1.0f);

			_actorOutputs[a]._columnOutputs[i] = sigmoid(sum);

			columnSum += _actorOutputs[a]._columnWeights[i] * _actorOutputs[a]._columnOutputs[i];
		}

		float output = _actorOutputs[a]._output = columnSum;

		// Perturb and randomly break
		if (dist01(generator) < breakRate)
			_outputs[a] = dist01(generator) * 2.0f - 1.0f;
		else
			_outputs[a] = std::min(1.0f, std::max(-1.0f, std::min(1.0f, std::max(-1.0f, output)) + outputPertDist(generator)));

		_outputOffsets[a] = _outputs[a] - output;
	}

	// ------------------------------------- Update Prevs -------------------------------------

	_criticOutput._prevPrevOutput = _criticOutput._prevOutput;
	_criticOutput._prevOutput = _criticOutput._output;

	for (int i = 0; i < numColumns; i++) {
		_criticOutput._columnPrevPrevOutputs[i] = _criticOutput._columnPrevOutputs[i];
		_criticOutput._columnPrevOutputs[i] = _criticOutput._columnOutputs[i];
	}

	for (int a = 0; a < _actorOutputs.size(); a++) {
		_actorOutputs[a]._prevPrevOutput = _actorOutputs[a]._prevOutput;
		_actorOutputs[a]._prevOutput = _actorOutputs[a]._output;

		for (int i = 0; i < numColumns; i++) {
			_actorOutputs[a]._columnPrevPrevOutputs[i] = _actorOutputs[a]._columnPrevOutputs[i];
			_actorOutputs[a]._columnPrevOutputs[i] = _actorOutputs[a]._columnOutputs[i];
		}
	}

	_prevNextQ = nextQ;
	_prevQ = newQ;

	_prevPrevOutputs = _prevOutputs;
	_prevOutputs = _outputs;

	_prevError = tdError;

	_prevVariance = _variance;
	_variance = tdError;

	// Copy to previous states
	_prevPrevState = _prevState;

	for (int i = 0; i < numColumns; i++)
	for (int j = 0; j < regionDesc._columnSize; j++)
		_prevState[j + i * regionDesc._columnSize] = _region.getColumn(i).getCell(j).isActive();
}