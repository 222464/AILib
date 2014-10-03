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
		_outputNodes[i]._connections.resize(_rbfNodes.size());

		for (int j = 0; j < _outputNodes[i]._connections.size(); j++)
			_outputNodes[i]._connections[j]._weight = weightDist(generator);

		_outputNodes[i]._bias._weight = weightDist(generator);
	}
}

void RBFNetwork::getOutput(const std::vector<float> &input, std::vector<float> &output) {
	for (int i = 0; i < _rbfNodes.size(); i++) {
		float dist2 = 0.0f;

		for (int j = 0; j < _rbfNodes[i]._center.size(); j++) {
			float delta = input[j] - _rbfNodes[i]._center[j];
			dist2 += delta * delta;
		}

		_rbfNodes[i]._output = std::exp(-_rbfNodes[i]._width * dist2);
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

bool RBFNetwork::getPrediction(const std::vector<float> &input, std::vector<float> &output, float threshold) {
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

	bool certain = std::sqrt(minDist2) < threshold;
	
	std::vector<float> rbfOutputs(_rbfNodes.size());

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
		float sum = _outputNodes[i]._bias._weight;

		for (int j = 0; j < _outputNodes[i]._connections.size(); j++)
			sum += rbfOutputs[j] * _outputNodes[i]._connections[j]._weight;

		output[i] = sum;
	}

	return certain;
}

void RBFNetwork::update(const std::vector<float> &input, std::vector<float> &output, const std::vector<float> &target, float centerAlpha, float widthAlpha, float weightAlpha) {
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

	// Find error of this node
	float minDist2NodeError = 0.0f;

	for (int i = 0; i < _outputNodes.size(); i++)
		minDist2NodeError += (target[i] - output[i]) * _outputNodes[i]._connections[minDist2Index]._weight;

	float errorScaledCenterAlpha = std::abs(minDist2NodeError) * centerAlpha;

	// Move minimum distance node towards this input
	float dist2 = 0.0f;

	for (int i = 0; i < _rbfNodes[minDist2Index]._center.size(); i++) {
		float delta = input[i] - _rbfNodes[minDist2Index]._center[i];

		_rbfNodes[minDist2Index]._center[i] += errorScaledCenterAlpha * delta;

		dist2 += delta * delta;
	}

	_rbfNodes[minDist2Index]._width = std::max(0.0f, _rbfNodes[minDist2Index]._width + std::abs(minDist2NodeError) * widthAlpha * (std::sqrt(dist2) - _rbfNodes[minDist2Index]._width));

	// Get new output
	std::vector<float> output;

	getOutput(input, output);

	// Update output node weights
	for (int i = 0; i < _outputNodes.size(); i++) {
		float alphaError = weightAlpha * (target[i] - output[i]);

		for (int j = 0; j < _outputNodes[i]._connections.size(); j++)
			_outputNodes[i]._connections[j]._weight += alphaError * _rbfNodes[j]._output;
		
		_outputNodes[i]._bias._weight += alphaError;
	}
}

void RBFNetwork::learnFeatures(const std::vector<float> &input, float centerAlpha, float widthAlpha) {
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

	_rbfNodes[minDist2Index]._width = std::max(0.0f, _rbfNodes[minDist2Index]._width + widthAlpha * (std::sqrt(dist2) - _rbfNodes[minDist2Index]._width));
}

int RBFNetwork::step(const std::vector<float> &input, float reward, float alpha, float gamma, float lambda, float tauInv, float epsilon, int prevAction, std::mt19937 &generator) {
	std::vector<float> qValues(_outputNodes.size());

	float prevMaxQ = -999999.0f;
	
	for (int i = 0; i < _outputNodes.size(); i++) {
		float sum = _outputNodes[i]._bias._weight;

		for (int j = 0; j < _outputNodes[i]._connections.size(); j++)
			sum += _rbfNodes[j]._output * _outputNodes[i]._connections[j]._weight;

		qValues[i] = sum;

		if (qValues[i] > prevMaxQ)
			prevMaxQ = qValues[i];
	}

	float prevValue = qValues[prevAction];

	std::uniform_real_distribution<float> uniformDist(0.0f, 1.0f);

	// Get new features
	std::vector<float> nextRBFOutputs(_rbfNodes.size());

	for (int i = 0; i < _rbfNodes.size(); i++) {
		float dist2 = 0.0f;

		for (int j = 0; j < _rbfNodes[i]._center.size(); j++) {
			float delta = input[j] - _rbfNodes[i]._center[j];
			dist2 += delta * delta;
		}

		nextRBFOutputs[i] = std::exp(-_rbfNodes[i]._width * dist2);
	}

	int action;

	float nextQ;

	if (uniformDist(generator) < epsilon) {
		std::uniform_int_distribution<int> actionDist(0, _outputNodes.size() - 1);

		action = actionDist(generator);

		float sum = _outputNodes[action]._bias._weight;

		for (int j = 0; j < _outputNodes[action]._connections.size(); j++)
			sum += nextRBFOutputs[j] * _outputNodes[action]._connections[j]._weight;

		nextQ = sum;
	}
	else {
		nextQ = -999999.0f;

		for (int a = 0; a < _outputNodes.size(); a++) {
			float sum = _outputNodes[a]._bias._weight;

			for (int j = 0; j < _outputNodes[a]._connections.size(); j++)
				sum += nextRBFOutputs[j] * _outputNodes[a]._connections[j]._weight;

			if (sum > nextQ) {
				nextQ = sum;
				action = a;
			}
		}
	}

	float tdError = prevMaxQ + (reward + gamma * nextQ - prevMaxQ) * tauInv - prevValue;

	// Update parameters
	for (int i = 0; i < _outputNodes.size(); i++) {
		_outputNodes[i]._bias._weight += alpha * tdError * _outputNodes[i]._bias._eligibility;

		for (int j = 0; j < _outputNodes[i]._connections.size(); j++)
			_outputNodes[i]._connections[j]._weight += alpha * tdError * _outputNodes[i]._connections[j]._eligibility;
	}

	// Decay eligibilities
	float lambdaGamma = lambda * gamma;

	for (int i = 0; i < _outputNodes.size(); i++) {
		_outputNodes[i]._bias._eligibility *= lambdaGamma;

		for (int j = 0; j < _outputNodes[i]._connections.size(); j++)
			_outputNodes[i]._connections[j]._eligibility *= lambdaGamma;
	}

	// Pass on features to storage
	for (int i = 0; i < _rbfNodes.size(); i++)
		_rbfNodes[i]._output = nextRBFOutputs[i];

	// Update eligibilities
	_outputNodes[action]._bias._eligibility += 1.0f;

	for (int j = 0; j < _outputNodes[action]._connections.size(); j++)
		_outputNodes[action]._connections[j]._eligibility += _rbfNodes[j]._output;

	std::cout << tdError << " " << nextQ << std::endl;

	return action;
}