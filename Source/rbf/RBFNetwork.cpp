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

#include <rbf/RBFNetwork.h>

#include <algorithm>

#include <iostream>

using namespace rbf;

void RBFNetwork::createRandom(int numInputs, int numRBF, int numOutputs, float minCenter, float maxCenter, float minWidth, float maxWidth, float minWeight, float maxWeight, std::mt19937 &generator) {
	std::uniform_real_distribution<float> centerDist(minCenter, maxCenter);
	std::uniform_real_distribution<float> widthDist(minWidth, maxWidth);
	std::uniform_real_distribution<float> weightDist(minWeight, maxWeight);

	_rbfNodes.resize(numRBF);

	for (int i = 0; i < _rbfNodes.size(); i++) {
		_rbfNodes[i]._center.resize(numInputs);

		for (int j = 0; j < _rbfNodes[i]._center.size(); j++)
			_rbfNodes[i]._center[j] = centerDist(generator);

		_rbfNodes[i]._width = widthDist(generator);
	}

	_outputNodes.resize(numOutputs);

	for (int i = 0; i < _outputNodes.size(); i++) {
		_outputNodes[i]._weights.resize(_rbfNodes.size());

		for (int j = 0; j < _outputNodes[i]._weights.size(); j++)
			_outputNodes[i]._weights[j] = weightDist(generator);

		_outputNodes[i]._bias = weightDist(generator);
	}
}

void RBFNetwork::getOutput(const std::vector<float> &input, std::vector<float> &rbfOutputs, std::vector<float> &output) {
	if (rbfOutputs.size() != _rbfNodes.size())
		rbfOutputs.resize(_rbfNodes.size());
	
	for (int i = 0; i < _rbfNodes.size(); i++) {
		float dist2 = 0.0f;

		for (int j = 0; j < _rbfNodes[i]._center.size(); j++) {
			float delta = input[j] - _rbfNodes[i]._center[j];
			dist2 += delta * delta;
		}

		rbfOutputs[i] = std::exp(-_rbfNodes[i]._width * dist2);
	}

	if (output.size() != _outputNodes.size())
		output.resize(_outputNodes.size());

	for (int i = 0; i < _outputNodes.size(); i++) {
		float sum = _outputNodes[i]._bias;

		for (int j = 0; j < _outputNodes[i]._weights.size(); j++)
			sum += rbfOutputs[j] * _outputNodes[i]._weights[j];

		output[i] = sum;
	}
}

void RBFNetwork::update(const std::vector<float> &input, const std::vector<float> &target, float centerAlpha, float widthAlpha, float weightAlpha) {
	// Find closest node
	float minDist2 = 999999.0f;
	int minDist2Index = 0;

	for (int i = 0; i < _rbfNodes.size(); i++) {
		float dist2 = 0.0f;

		for (int j = 0; j < _rbfNodes[i]._center.size(); j++) {
			float delta = input[j] - _rbfNodes[i]._center[j];
			dist2 += delta * delta;
		}

		if (dist2 < minDist2) {
			minDist2 = dist2;

			minDist2Index = i;
		}
	}

	// Move minimum distance node towards this input
	float dist2 = 0.0f;

	for (int i = 0; i < _rbfNodes[minDist2Index]._center.size(); i++) {
		float delta = input[i] - _rbfNodes[minDist2Index]._center[i];

		_rbfNodes[minDist2Index]._center[i] += centerAlpha * delta;

		dist2 += delta * delta;
	}

	_rbfNodes[minDist2Index]._width += widthAlpha * (std::sqrt(dist2) - _rbfNodes[minDist2Index]._width);

	// Get new output
	std::vector<float> output;
	std::vector<float> rbfOutputs;

	getOutput(input, rbfOutputs, output);

	// Update output node weights
	for (int i = 0; i < _outputNodes.size(); i++) {
		float alphaError = weightAlpha * (target[i] - output[i]);

		for (int j = 0; j < _outputNodes[i]._weights.size(); j++)
			_outputNodes[i]._weights[j] += alphaError * rbfOutputs[j];
		
		_outputNodes[i]._bias += alphaError;
	}
}