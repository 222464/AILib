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

void CHTMRL::createRandom(int inputWidth, int inputHeight, int columnsWidth, int columnsHeight, int cellsPerColumn, int receptiveRadius, int cellRadius,
	float minCenter, float maxCenter, float minWidth, float maxWidth, float minInputWeight, float maxInputWeight, float minReconWeight, float maxReconWeight,
	float minCellWeight, float maxCellWeight, float minOutputWeight, float maxOutputWeight, std::mt19937 &generator)
{
	_region.createRandom(inputWidth, inputHeight, columnsWidth, columnsHeight, cellsPerColumn, receptiveRadius, cellRadius, 1,
		minCenter, maxCenter, minWidth, maxWidth, minInputWeight, maxInputWeight, minReconWeight, maxReconWeight, minCellWeight, maxCellWeight, minOutputWeight, maxOutputWeight, generator);
}

void CHTMRL::step(float reward, const std::vector<float> &input, const std::vector<bool> &actionMask, std::vector<float> &action, float optimizationAlpha, int optimizationSteps, float optimizationPerturbationStdDev, float optimizationDecay, float indecisivnessIntensity, float perturbationIntensity, float intentSparsity, float intentIntensity, int inhibitionRadius, float sparsity, float cellIntensity, float predictionIntensity, float weightAlphaQ, float reconAlpha, float centerAlpha, float widthAlpha, float widthScalar,
	float minDistance, float minLearningThreshold, float cellAlpha, float qAlpha, float gamma, float lambda, float tauInv, std::mt19937 &generator)
{
	_region.stepBegin();

	std::vector<float> output(1);

	_region.getOutputAction(input, output, action, indecisivnessIntensity, perturbationIntensity, intentSparsity, intentIntensity, inhibitionRadius, sparsity, cellIntensity, predictionIntensity, optimizationSteps, optimizationAlpha, optimizationPerturbationStdDev, optimizationDecay, reconAlpha, generator);

	float newAdv = _prevValue + (reward + gamma * output[0] - _prevValue) * tauInv;

	float tdError = newAdv - _prevValue;

	_prevValue = output[0];

	std::vector<float> error(1, tdError);

	std::vector<float> outputLambdas(1, lambda);

	std::vector<float> weightAlphas(1, weightAlphaQ);

	_region.learnTraces(input, output, error, weightAlphas, centerAlpha, widthAlpha, widthScalar, minDistance, minLearningThreshold, cellAlpha, predictionIntensity, outputLambdas);

	std::cout << output[0] << " " << action[4] << std::endl;
}
