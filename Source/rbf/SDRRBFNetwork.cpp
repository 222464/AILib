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

void SDRRBFNetwork::createRandom(int inputWidth, int inputHeight, const std::vector<LayerDesc> &layerDescs, int numOutputs, float minCenter, float maxCenter, float minWidth, float maxWidth, float minWeight, float maxWeight, std::mt19937 &generator) {
	std::uniform_real_distribution<float> centerDist(minCenter, maxCenter);
	std::uniform_real_distribution<float> widthDist(minWidth, maxWidth);
	std::uniform_real_distribution<float> weightDist(minWeight, maxWeight);

	_inputWidth = inputWidth;
	_inputHeight = inputHeight;

	int numInputs = _inputWidth * _inputHeight;

	_layerDescs = layerDescs;

	_layers.resize(_layerDescs.size());

	int totalOutputConnections = 0;

	for (int l = 0; l < _layers.size(); l++) {
		int numRBF = _layerDescs[l]._rbfWidth * _layerDescs[l]._rbfHeight;
		int numRBFWeights = std::pow(_layerDescs[l]._receptiveRadius * 2 + 1, 2);

		totalOutputConnections += numRBF;

		_layers[l]._rbfNodes.resize(numRBF);

		for (int i = 0; i < _layers[l]._rbfNodes.size(); i++) {
			_layers[l]._rbfNodes[i]._center.resize(numRBFWeights);

			for (int j = 0; j < _layers[l]._rbfNodes[i]._center.size(); j++)
				_layers[l]._rbfNodes[i]._center[j] = centerDist(generator);

			_layers[l]._rbfNodes[i]._width = widthDist(generator);
		}
	}

	_outputNodes.resize(numOutputs);

	for (int i = 0; i < _outputNodes.size(); i++) {
		_outputNodes[i]._connections.resize(totalOutputConnections);

		for (int j = 0; j < _outputNodes[i]._connections.size(); j++)
			_outputNodes[i]._connections[j]._weight = weightDist(generator);

		_outputNodes[i]._bias._weight = weightDist(generator);
	}
}

void SDRRBFNetwork::getOutput(const std::vector<float> &input, std::vector<float> &output, float localActivity, float activationIntensity, float outputIntensity, float minDutyCycleRatio, float dutyCycleDecay, float randomFireChance, float randomFireStrength, std::mt19937 &generator) {
	int prevLayerWidth = _inputWidth;
	int prevLayerHeight = _inputHeight;

	std::vector<float> prevLayerOutput = input;

	std::uniform_real_distribution<float> uniformDist(0.0f, 1.0f);
	
	for (int l = 0; l < _layers.size(); l++) {
		float inputWidthInv = 1.0f / prevLayerWidth;
		float inputHeightInv = 1.0f / prevLayerHeight;

		float rbfWidthInv = 1.0f / _layerDescs[l]._rbfWidth;
		float rbfHeightInv = 1.0f / _layerDescs[l]._rbfHeight;

		std::uniform_real_distribution<float> uniformDist(0.0f, 1.0f);

		for (int rx = 0; rx < _layerDescs[l]._rbfWidth; rx++)
		for (int ry = 0; ry < _layerDescs[l]._rbfHeight; ry++) {
			int i = rx + ry * _layerDescs[l]._rbfWidth;

			float rxn = rx * rbfWidthInv;
			float ryn = ry * rbfHeightInv;

			float dist2 = 0.0f;

			int weightIndex = 0;

			for (int dx = -_layerDescs[l]._receptiveRadius; dx <= _layerDescs[l]._receptiveRadius; dx++)
			for (int dy = -_layerDescs[l]._receptiveRadius; dy <= _layerDescs[l]._receptiveRadius; dy++) {
				float xn = rxn + dx * inputWidthInv;
				float yn = ryn + dy * inputHeightInv;

				if (xn >= 0.0f && xn < 1.0f && yn >= 0.0f && yn < 1.0f) {
					int x = xn * _inputWidth;
					int y = yn * _inputHeight;

					int j = x + y * _inputWidth;

					float delta = input[j] - _layers[l]._rbfNodes[i]._center[weightIndex];

					dist2 += delta * delta;
				}

				weightIndex++;
			}

			_layers[l]._rbfNodes[i]._activation = std::exp(-dist2 * activationIntensity);
		}

		// Sparsify
		for (int rx = 0; rx < _layerDescs[l]._rbfWidth; rx++)
		for (int ry = 0; ry < _layerDescs[l]._rbfHeight; ry++) {
			int i = rx + ry * _layerDescs[l]._rbfWidth;

			int numHigher = 0;

			//float maxNeighborhoodDutyCycle = 0.0f;

			for (int dx = -_layerDescs[l]._inhibitionRadius; dx <= _layerDescs[l]._inhibitionRadius; dx++)
			for (int dy = -_layerDescs[l]._inhibitionRadius; dy <= _layerDescs[l]._inhibitionRadius; dy++) {
				int x = rx + dx;
				int y = ry + dy;

				if (x >= 0 && x < _layerDescs[l]._rbfWidth && y >= 0 && y < _layerDescs[l]._rbfHeight) {
					int j = x + y * _layerDescs[l]._rbfWidth;

					if (_layers[l]._rbfNodes[j]._activation > _layers[l]._rbfNodes[i]._activation)
						numHigher++;

					//maxNeighborhoodDutyCycle = std::max(maxNeighborhoodDutyCycle, _layers[l]._rbfNodes[j]._dutyCycle);
				}
			}

			//_layers[l]._rbfNodes[i]._minDutyCycle = minDutyCycleRatio * maxNeighborhoodDutyCycle;

			_layers[l]._rbfNodes[i]._output = sigmoid((localActivity - numHigher) * outputIntensity);

			_layers[l]._rbfNodes[i]._output = std::min(1.0f, _layers[l]._rbfNodes[i]._output + (uniformDist(generator) < randomFireChance ? randomFireStrength : 0.0f));

			//_layers[l]._rbfNodes[i]._dutyCycle = (1.0f - dutyCycleDecay) * _layers[l]._rbfNodes[i]._dutyCycle + dutyCycleDecay * _layers[l]._rbfNodes[i]._output;
		}

		prevLayerOutput.resize(_layers[l]._rbfNodes.size());

		for (int i = 0; i < _layers[l]._rbfNodes.size(); i++)
			prevLayerOutput[i] = _layers[l]._rbfNodes[i]._output;

		prevLayerWidth = _layerDescs[l]._rbfWidth;
		prevLayerHeight = _layerDescs[l]._rbfHeight;
	}

	if (output.size() != _outputNodes.size())
		output.resize(_outputNodes.size());

	for (int i = 0; i < _outputNodes.size(); i++) {
		float sum = _outputNodes[i]._bias._weight;

		int ci = 0;

		for (int l = 0; l < _layers.size(); l++)
		for (int j = 0; j < _layers[l]._rbfNodes.size(); j++)
			sum += _layers[l]._rbfNodes[j]._output * _layerDescs[l]._outputMultiplier * _outputNodes[i]._connections[ci++]._weight;

		output[i] = sum;
	}
}

void SDRRBFNetwork::update(const std::vector<float> &input, std::vector<float> &output, const std::vector<float> &target, float weightAlpha, float centerAlpha, float widthAlpha, float widthScalar, float minDistance, float minLearningThreshold) {
	for (int i = 0; i < _outputNodes.size(); i++) {
		float alphaError = weightAlpha * (target[i] - output[i]);

		int ci = 0;

		for (int l = 0; l < _layers.size(); l++)
		for (int j = 0; j < _layers[l]._rbfNodes.size(); j++)
			_outputNodes[i]._connections[ci++]._weight += alphaError * _layers[l]._rbfNodes[j]._output * _layerDescs[l]._outputMultiplier;

		_outputNodes[i]._bias._weight += alphaError;
	}

	for (int l = 0; l < _layers.size(); l++) {
		float inputWidthInv = 1.0f / _inputWidth;
		float inputHeightInv = 1.0f / _inputHeight;

		float rbfWidthInv = 1.0f / _layerDescs[l]._rbfWidth;
		float rbfHeightInv = 1.0f / _layerDescs[l]._rbfHeight;

		for (int rx = 0; rx < _layerDescs[l]._rbfWidth; rx++)
		for (int ry = 0; ry < _layerDescs[l]._rbfHeight; ry++) {
			int i = rx + ry * _layerDescs[l]._rbfWidth;

			float rxn = rx * rbfWidthInv;
			float ryn = ry * rbfHeightInv;

			//float dist2 = 0.0f;

			int weightIndex = 0;

			float learnScalar = std::max(0.0f, _layers[l]._rbfNodes[i]._output - minLearningThreshold);

			for (int dx = -_layerDescs[l]._receptiveRadius; dx <= _layerDescs[l]._receptiveRadius; dx++)
			for (int dy = -_layerDescs[l]._receptiveRadius; dy <= _layerDescs[l]._receptiveRadius; dy++) {
				float xn = rxn + dx * inputWidthInv;
				float yn = ryn + dy * inputHeightInv;

				if (xn >= 0.0f && xn < 1.0f && yn >= 0.0f && yn < 1.0f) {
					int x = xn * _inputWidth;
					int y = yn * _inputHeight;

					int j = x + y * _inputWidth;

					_layers[l]._rbfNodes[i]._center[weightIndex] += centerAlpha * learnScalar * (input[j] - _layers[l]._rbfNodes[i]._center[weightIndex]);

					float delta = input[j] - _layers[l]._rbfNodes[i]._center[weightIndex];

					//dist2 += delta * delta;
				}

				weightIndex++;
			}

			//_layers[l]._rbfNodes[i]._width = std::max(0.0f, _layers[l]._rbfNodes[i]._width + widthAlpha * learnScalar * (widthScalar / std::max(minDistance, dist2) - _layers[l]._rbfNodes[i]._width));
		}
	}
}