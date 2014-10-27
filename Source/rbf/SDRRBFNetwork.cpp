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

#include <rbf/SDRRBFNetwork.h>

#include <algorithm>

#include <iostream>

#include <assert.h>

using namespace rbf;

void SDRRBFNetwork::createRandom(int inputWidth, int inputHeight, int rbfWidth, int rbfHeight, int receptiveRadius, int numOutputs, float minCenter, float maxCenter, float minWidth, float maxWidth, float minWeight, float maxWeight, std::mt19937 &generator) {
	std::uniform_real_distribution<float> centerDist(minCenter, maxCenter);
	std::uniform_real_distribution<float> widthDist(minWidth, maxWidth);
	std::uniform_real_distribution<float> weightDist(minWeight, maxWeight);

	_inputWidth = inputWidth;
	_inputHeight = inputHeight;

	_rbfWidth = rbfWidth;
	_rbfHeight = rbfHeight;

	_receptiveRadius = receptiveRadius;

	int numInputs = _inputWidth * _inputHeight;
	int numRBF = _rbfWidth * _rbfHeight;
	int numRBFWeights = std::pow(receptiveRadius * 2 + 1, 2);

	_rbfNodes.resize(numRBF);

	for (int i = 0; i < _rbfNodes.size(); i++) {
		_rbfNodes[i]._center.resize(numRBFWeights);

		for (int j = 0; j < _rbfNodes[i]._center.size(); j++)
			_rbfNodes[i]._center[j] = centerDist(generator);

		_rbfNodes[i]._width = widthDist(generator);
	}

	_outputNodes.resize(numOutputs);

	for (int i = 0; i < _outputNodes.size(); i++) {
		_outputNodes[i]._connections.resize(_rbfNodes.size());

		for (int j = 0; j < _outputNodes[i]._connections.size(); j++)
			_outputNodes[i]._connections[j]._weight = weightDist(generator);

		_outputNodes[i]._bias._weight = weightDist(generator);
	}
}

void SDRRBFNetwork::getOutput(const std::vector<float> &input, std::vector<float> &output, int inhibitionRadius, float sparsity, float minActivation) {
	float inputWidthInv = 1.0f / _inputWidth;
	float inputHeightInv = 1.0f / _inputHeight;

	float rbfWidthInv = 1.0f / _rbfWidth;
	float rbfHeightInv = 1.0f / _rbfHeight;

	for (int rx = 0; rx < _rbfWidth; rx++)
	for (int ry = 0; ry < _rbfHeight; ry++) {
		int i = rx + ry * _rbfWidth;

		float rxn = rx * rbfWidthInv;
		float ryn = ry * rbfHeightInv;

		float dist2 = 0.0f;

		int weightIndex = 0;

		for (int dx = -_receptiveRadius; dx <= _receptiveRadius; dx++)
		for (int dy = -_receptiveRadius; dy <= _receptiveRadius; dy++) {
			float xn = rxn + dx * inputWidthInv;
			float yn = ryn + dy * inputHeightInv;

			if (xn >= 0.0f && xn < 1.0f && yn >= 0.0f && yn < 1.0f) {
				int x = xn * _inputWidth;
				int y = yn * _inputHeight;

				int j = x + y * _inputWidth;

				float delta = input[j] - _rbfNodes[i]._center[weightIndex];

				dist2 += delta * delta;
			}

			weightIndex++;
		}

		_rbfNodes[i]._activation = std::exp(-_rbfNodes[i]._width * dist2);
	}

	// Sparsify
	for (int rx = 0; rx < _rbfWidth; rx++)
	for (int ry = 0; ry < _rbfHeight; ry++) {
		int i = rx + ry * _rbfWidth;

		float maximum = 0.0f;
		float average = 0.0f;

		int count = 0;

		for (int dx = -inhibitionRadius; dx <= inhibitionRadius; dx++)
		for (int dy = -inhibitionRadius; dy <= inhibitionRadius; dy++) {
			int x = rx + dx;
			int y = ry + dy;

			if (x >= 0 && x < _rbfWidth && y >= 0 && y < _rbfHeight) {
				int j = x + y * _rbfWidth;

				maximum = std::max(maximum, _rbfNodes[j]._activation);

				average += _rbfNodes[j]._activation;
				count++;
			}
		}

		average /= count;

		_rbfNodes[i]._output = std::exp((_rbfNodes[i]._activation - maximum) / (minActivation + maximum - average) * sparsity);

		assert(_rbfNodes[i]._output >= 0.0f && _rbfNodes[i]._output <= 1.0f);
	}

	if (output.size() != _outputNodes.size())
		output.resize(_outputNodes.size());

	for (int i = 0; i < _outputNodes.size(); i++) {
		float sum = _outputNodes[i]._bias._weight;

		for (int j = 0; j < _outputNodes[i]._connections.size(); j++)
			sum += _rbfNodes[j]._output * _outputNodes[i]._connections[j]._weight;

		output[i] = sum;
	}
}

void SDRRBFNetwork::update(const std::vector<float> &input, std::vector<float> &output, const std::vector<float> &target, float weightAlpha, float centerAlpha, float widthAlpha, float widthScalar) {
	// Update output node weights
	for (int i = 0; i < _outputNodes.size(); i++) {
		float alphaError = weightAlpha * (target[i] - output[i]);

		for (int j = 0; j < _outputNodes[i]._connections.size(); j++)
			_outputNodes[i]._connections[j]._weight += alphaError * _rbfNodes[j]._output;
		
		_outputNodes[i]._bias._weight += alphaError;
	}

	float inputWidthInv = 1.0f / _inputWidth;
	float inputHeightInv = 1.0f / _inputHeight;

	float rbfWidthInv = 1.0f / _rbfWidth;
	float rbfHeightInv = 1.0f / _rbfHeight;

	for (int rx = 0; rx < _rbfWidth; rx++)
	for (int ry = 0; ry < _rbfHeight; ry++) {
		int i = rx + ry * _rbfWidth;

		float rxn = rx * rbfWidthInv;
		float ryn = ry * rbfHeightInv;

		float dist2 = 0.0f;

		int weightIndex = 0;

		for (int dx = -_receptiveRadius; dx <= _receptiveRadius; dx++)
		for (int dy = -_receptiveRadius; dy <= _receptiveRadius; dy++) {
			float xn = rxn + dx * inputWidthInv;
			float yn = ryn + dy * inputHeightInv;

			if (xn >= 0.0f && xn < 1.0f && yn >= 0.0f && yn < 1.0f) {
				int x = xn * _inputWidth;
				int y = yn * _inputHeight;

				int j = x + y * _inputWidth;

				_rbfNodes[i]._center[weightIndex] += centerAlpha * _rbfNodes[i]._output * (input[j] - _rbfNodes[i]._center[weightIndex]);

				float delta = input[j] - _rbfNodes[i]._center[weightIndex];

				dist2 += delta * delta;
			}

			weightIndex++;
		}

		_rbfNodes[i]._width = std::max(0.0f, _rbfNodes[i]._width + widthAlpha * _rbfNodes[i]._output * (dist2 / (2.0f * widthScalar * widthScalar) - _rbfNodes[i]._width));
	}
}