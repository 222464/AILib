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

#include <chtm/CHTMRL.h>

#include <algorithm>

#include <iostream>

#include <assert.h>

using namespace chtm;

void CHTMRL::createRandom(int inputWidth, int inputHeight, int numOutputs, int columnsWidth, int columnsHeight, int cellsPerColumn, int receptiveRadius, int cellRadius,
	float minCenter, float maxCenter, float minWidth, float maxWidth, float minInputWeight, float maxInputWeight,
	float minCellWeight, float maxCellWeight, float minOutputWeight, float maxOutputWeight, std::mt19937 &generator)
{
	assert(inputWidth * inputHeight > numOutputs);

	_region.createRandom(inputWidth, inputHeight, columnsWidth, columnsHeight, cellsPerColumn, receptiveRadius, cellRadius, numOutputs + 1,
		minCenter, maxCenter, minWidth, maxWidth, minInputWeight, maxInputWeight, minCellWeight, maxCellWeight, minOutputWeight, maxOutputWeight, generator);

	_prevActionUnclamped.clear();
	_prevActionUnclamped.assign(numOutputs, 0.0f);

	_prevActionPerturbed.clear();
	_prevActionPerturbed.assign(numOutputs, 0.0f);
}

void CHTMRL::step(float reward, const std::vector<float> &input, std::vector<float> &action, int inhibitionRadius, float sparsity, float cellIntensity, float predictionIntensity, float weightAlphaQ, float weightAlphaAction, float centerAlpha, float widthAlpha, float widthScalar,
	float minDistance, float minLearningThreshold, float cellAlpha, float qAlpha, float gamma, float lambda, float tauInv, float breakRate, float perturbationStdDev, std::mt19937 &generator)
{
	_region.stepBegin();

	std::vector<float> output(_prevActionUnclamped.size() + 1);

	_region.getOutput(input, output, inhibitionRadius, sparsity, cellIntensity, predictionIntensity, generator);

	float newAdv = _prevValue + (reward + gamma * output[0] - _prevValue) * tauInv;

	float tdError = newAdv - _prevValue;

	_prevValue = output[0];

	std::vector<float> error(_prevActionUnclamped.size() + 1);

	error[0] = tdError;

	if (tdError > 0.0f) {
		for (int i = 0; i < _prevActionPerturbed.size(); i++)
			error[i + 1] = _prevActionPerturbed[i] - _prevActionUnclamped[i];
	}
	else {
		for (int i = 0; i < _prevActionUnclamped.size(); i++)
			error[i + 1] = 0.0f;
	}

	std::vector<float> outputLambdas(_prevActionUnclamped.size() + 1, 0.0f);
	outputLambdas[0] = lambda;

	std::vector<float> weightAlphas(_prevActionUnclamped.size() + 1, weightAlphaAction);
	weightAlphas[0] = weightAlphaQ;

	_region.learnTraces(input, output, error, weightAlphas, centerAlpha, widthAlpha, widthScalar, minDistance, minLearningThreshold, cellAlpha, outputLambdas);

	std::uniform_real_distribution<float> uniformDist(0.0f, 1.0f);
	std::normal_distribution<float> perturbationDist(0.0f, perturbationStdDev);

	for (int i = 0; i < _prevActionUnclamped.size(); i++) {
		_prevActionUnclamped[i] = output[i + 1];

		if (uniformDist(generator) < breakRate)
			_prevActionPerturbed[i] = uniformDist(generator) * 2.0f - 1.0f;
		else
			_prevActionPerturbed[i] = std::min(1.0f, std::max(-1.0f, std::min(1.0f, std::max(-1.0f, _prevActionUnclamped[i])) + perturbationDist(generator)));
	}

	action = _prevActionPerturbed;

	std::cout << output[0] << std::endl;
}
