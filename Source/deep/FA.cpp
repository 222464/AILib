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

#include <deep/FA.h>

#include <list>

using namespace deep;

void FA::createRandom(int numInputs, int numOutputs, int numHiddenLayers, int numNeuronsPerHiddenLayer, float weightStdDev, std::mt19937 &generator) {
	std::normal_distribution<float> distWeight(0.0f, weightStdDev);

	if (numHiddenLayers > 0) {
		_hiddenLayers.resize(numHiddenLayers);

		// First hidden layer
		_hiddenLayers[0].resize(numNeuronsPerHiddenLayer);

		for (int n = 0; n < _hiddenLayers[0].size(); n++) {
			_hiddenLayers[0][n]._connections.resize(numInputs);

			for (int w = 0; w < _hiddenLayers[0][n]._connections.size(); w++)
				_hiddenLayers[0][n]._connections[w]._weight = distWeight(generator);

			_hiddenLayers[0][n]._bias._weight = distWeight(generator);
		}

		// All other hidden layers
		for (int l = 1; l < _hiddenLayers.size(); l++) {
			_hiddenLayers[l].resize(numNeuronsPerHiddenLayer);

			for (int n = 0; n < _hiddenLayers[l].size(); n++) {
				_hiddenLayers[l][n]._connections.resize(numNeuronsPerHiddenLayer);

				for (int w = 0; w < _hiddenLayers[l][n]._connections.size(); w++)
					_hiddenLayers[l][n]._connections[w]._weight = distWeight(generator);

				_hiddenLayers[l][n]._bias._weight = distWeight(generator);
			}
		}

		_outputLayer.resize(numOutputs);

		for (int n = 0; n < _outputLayer.size(); n++) {
			_outputLayer[n]._connections.resize(numNeuronsPerHiddenLayer);

			for (int w = 0; w < _outputLayer[n]._connections.size(); w++)
				_outputLayer[n]._connections[w]._weight = distWeight(generator);

			_outputLayer[n]._bias._weight = distWeight(generator);
		}
	}
	else {
		_outputLayer.resize(numOutputs);

		for (int n = 0; n < _outputLayer.size(); n++) {
			_outputLayer[n]._connections.resize(numInputs);

			for (int w = 0; w < _outputLayer[n]._connections.size(); w++)
				_outputLayer[n]._connections[w]._weight = distWeight(generator);

			_outputLayer[n]._bias._weight = distWeight(generator);
		}
	}
}

float FA::crossoverChooseWeight(float w1, float w2, float averageChance, std::mt19937 &generator) {
	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

	if (dist01(generator) < averageChance)
		return (w1 + w2) * 0.5f;
	
	return dist01(generator) < 0.5f ? w1 : w2;
}

void FA::createFromParents(const FA &parent1, const FA &parent2, float averageChance, std::mt19937 &generator) {
	int numInputs = parent1.getNumInputs();
	int numOutputs = parent1.getNumOutputs();
	int numHiddenLayers = parent1.getNumHiddenLayers();
	int numNeuronsPerHiddenLayer = parent1.getNumNeuronsPerHiddenLayer();

	if (numHiddenLayers > 0) {
		_hiddenLayers.resize(numHiddenLayers);

		// First hidden layer
		_hiddenLayers[0].resize(numNeuronsPerHiddenLayer);

		for (int n = 0; n < _hiddenLayers[0].size(); n++) {
			_hiddenLayers[0][n]._connections.resize(numInputs);

			for (int w = 0; w < _hiddenLayers[0][n]._connections.size(); w++)
				_hiddenLayers[0][n]._connections[w]._weight = crossoverChooseWeight(parent1._hiddenLayers[0][n]._connections[w]._weight, parent2._hiddenLayers[0][n]._connections[w]._weight, averageChance, generator);

			_hiddenLayers[0][n]._bias._weight = crossoverChooseWeight(parent1._hiddenLayers[0][n]._bias._weight, parent2._hiddenLayers[0][n]._bias._weight, averageChance, generator);
		}

		// All other hidden layers
		for (int l = 1; l < _hiddenLayers.size(); l++) {
			_hiddenLayers[l].resize(numNeuronsPerHiddenLayer);

			for (int n = 0; n < _hiddenLayers[l].size(); n++) {
				_hiddenLayers[l][n]._connections.resize(numNeuronsPerHiddenLayer);

				for (int w = 0; w < _hiddenLayers[l][n]._connections.size(); w++)
					_hiddenLayers[l][n]._connections[w]._weight = crossoverChooseWeight(parent1._hiddenLayers[l][n]._connections[w]._weight, parent2._hiddenLayers[l][n]._connections[w]._weight, averageChance, generator);

				_hiddenLayers[l][n]._bias._weight = crossoverChooseWeight(parent1._hiddenLayers[l][n]._bias._weight, parent2._hiddenLayers[l][n]._bias._weight, averageChance, generator);
			}
		}

		_outputLayer.resize(numOutputs);

		for (int n = 0; n < _outputLayer.size(); n++) {
			_outputLayer[n]._connections.resize(numNeuronsPerHiddenLayer);

			for (int w = 0; w < _outputLayer[n]._connections.size(); w++)
				_outputLayer[n]._connections[w]._weight = crossoverChooseWeight(parent1._outputLayer[n]._connections[w]._weight, parent2._outputLayer[n]._connections[w]._weight, averageChance, generator);

			_outputLayer[n]._bias._weight = crossoverChooseWeight(parent1._outputLayer[n]._bias._weight, parent2._outputLayer[n]._bias._weight, averageChance, generator);
		}
	}
	else {
		_outputLayer.resize(numOutputs);

		for (int n = 0; n < _outputLayer.size(); n++) {
			_outputLayer[n]._connections.resize(numInputs);

			for (int w = 0; w < _outputLayer[n]._connections.size(); w++)
				_outputLayer[n]._connections[w]._weight = crossoverChooseWeight(parent1._outputLayer[n]._connections[w]._weight, parent2._outputLayer[n]._connections[w]._weight, averageChance, generator);

			_outputLayer[n]._bias._weight = crossoverChooseWeight(parent1._outputLayer[n]._bias._weight, parent2._outputLayer[n]._bias._weight, averageChance, generator);
		}
	}
}

void FA::mutate(float perturbationChance, float perturbationStdDev, std::mt19937 &generator) {
	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
	std::normal_distribution<float> distPerturbation(0.0f, perturbationStdDev);

	for (int l = 0; l < _hiddenLayers.size(); l++)
	for (int n = 0; n < _hiddenLayers[l].size(); n++) {
		for (int w = 0; w < _hiddenLayers[l][n]._connections.size(); w++)
			_hiddenLayers[l][n]._connections[w]._weight += dist01(generator) < perturbationChance ? distPerturbation(generator) : 0.0f;

		_hiddenLayers[l][n]._bias._weight += dist01(generator) < perturbationChance ? distPerturbation(generator) : 0.0f;
	}

	for (int n = 0; n < _outputLayer.size(); n++) {
		for (int w = 0; w < _outputLayer[n]._connections.size(); w++)
			_outputLayer[n]._connections[w]._weight += dist01(generator) < perturbationChance ? distPerturbation(generator) : 0.0f;

		_outputLayer[n]._bias._weight += dist01(generator) < perturbationChance ? distPerturbation(generator) : 0.0f;
	}
}

int FA::createFromWeightsVector(int numInputs, int numOutputs, int numHiddenLayers, int numNeuronsPerHiddenLayer, const std::vector<float> &weights, int startIndex) {
	int weightIndex = startIndex;
	
	if (numHiddenLayers > 0) {
		_hiddenLayers.resize(numHiddenLayers);

		// First hidden layer
		_hiddenLayers[0].resize(numNeuronsPerHiddenLayer);

		for (int n = 0; n < _hiddenLayers[0].size(); n++) {
			_hiddenLayers[0][n]._connections.resize(numInputs);

			for (int w = 0; w < _hiddenLayers[0][n]._connections.size(); w++)
				_hiddenLayers[0][n]._connections[w]._weight = weights[weightIndex++];

			_hiddenLayers[0][n]._bias._weight = weights[weightIndex++];
		}

		// All other hidden layers
		for (int l = 1; l < _hiddenLayers.size(); l++) {
			_hiddenLayers[l].resize(numNeuronsPerHiddenLayer);

			for (int n = 0; n < _hiddenLayers[l].size(); n++) {
				_hiddenLayers[l][n]._connections.resize(numNeuronsPerHiddenLayer);

				for (int w = 0; w < _hiddenLayers[l][n]._connections.size(); w++)
					_hiddenLayers[l][n]._connections[w]._weight = weights[weightIndex++];

				_hiddenLayers[l][n]._bias._weight = weights[weightIndex++];
			}
		}

		_outputLayer.resize(numOutputs);

		for (int n = 0; n < _outputLayer.size(); n++) {
			_outputLayer[n]._connections.resize(numNeuronsPerHiddenLayer);

			for (int w = 0; w < _outputLayer[n]._connections.size(); w++)
				_outputLayer[n]._connections[w]._weight = weights[weightIndex++];

			_outputLayer[n]._bias._weight = weights[weightIndex++];
		}
	}
	else {
		_outputLayer.resize(numOutputs);

		for (int n = 0; n < _outputLayer.size(); n++) {
			_outputLayer[n]._connections.resize(numInputs);

			for (int w = 0; w < _outputLayer[n]._connections.size(); w++)
				_outputLayer[n]._connections[w]._weight = weights[weightIndex++];

			_outputLayer[n]._bias._weight = weights[weightIndex++];
		}
	}

	return weightIndex;
}

void FA::getWeightsVector(std::vector<float> &weights) {
	for (int l = 0; l < _hiddenLayers.size(); l++)
	for (int n = 0; n < _hiddenLayers[l].size(); n++) {
		for (int w = 0; w < _hiddenLayers[l][n]._connections.size(); w++)
			weights.push_back(_hiddenLayers[l][n]._connections[w]._weight);

		weights.push_back(_hiddenLayers[l][n]._bias._weight);
	}

	for (int n = 0; n < _outputLayer.size(); n++) {
		for (int w = 0; w < _outputLayer[n]._connections.size(); w++)
			weights.push_back(_outputLayer[n]._connections[w]._weight);

		weights.push_back(_outputLayer[n]._bias._weight);
	}
}

void FA::process(const std::vector<float> &inputs, std::vector<float> &outputs) {
	if (!_hiddenLayers.empty()) {
		// First hidden layer
		for (int n = 0; n < _hiddenLayers[0].size(); n++) {
			float sum = _hiddenLayers[0][n]._bias._weight;

			for (int w = 0; w < _hiddenLayers[0][n]._connections.size(); w++)
				sum += inputs[w] * _hiddenLayers[0][n]._connections[w]._weight;

			_hiddenLayers[0][n]._output = sigmoid(sum);
		}

		if (_hiddenLayers.size() > 1) {
			// All other hidden layers
			for (int l = 1; l < _hiddenLayers.size(); l++) {
				int prevLayerIndex = l - 1;

				for (int n = 0; n < _hiddenLayers[l].size(); n++) {
					float sum = _hiddenLayers[l][n]._bias._weight;

					for (int w = 0; w < _hiddenLayers[l][n]._connections.size(); w++)
						sum += _hiddenLayers[prevLayerIndex][w]._output * _hiddenLayers[l][n]._connections[w]._weight;

					_hiddenLayers[l][n]._output = sigmoid(sum);
				}
			}
		}

		// Output layer
		for (int n = 0; n < _outputLayer.size(); n++) {
			float sum = _outputLayer[n]._bias._weight;

			for (int w = 0; w < _outputLayer[n]._connections.size(); w++)
				sum += _hiddenLayers.back()[w]._output * _outputLayer[n]._connections[w]._weight;

			outputs[n] = _outputLayer[n]._output = sum; // Linear activation
		}
	}
	else {
		// Output layer
		for (int n = 0; n < _outputLayer.size(); n++) {
			float sum = _outputLayer[n]._bias._weight;

			for (int w = 0; w < _outputLayer[n]._connections.size(); w++)
				sum += inputs[w] * _outputLayer[n]._connections[w]._weight;

			outputs[n] = _outputLayer[n]._output = sum; // Linear activation
		}
	}
}

void FA::backpropagate(const std::vector<float> &inputs, const std::vector<float> &targetOutputs, float alpha, float momentum) {
	// Output layer error
	for (int n = 0; n < _outputLayer.size(); n++)
		_outputLayer[n]._error = targetOutputs[n] - _outputLayer[n]._output;

	if (!_hiddenLayers.empty())  {
		// Last hidden layer
		for (int n = 0; n < _hiddenLayers.back().size(); n++) {
			float sum = 0.0f;

			for (int w = 0; w < _outputLayer.size(); w++)
				sum += _outputLayer[w]._error * _outputLayer[w]._connections[n]._weight;

			_hiddenLayers.back()[n]._error = sum * _hiddenLayers.back()[n]._output * (1.0f - _hiddenLayers.back()[n]._output);
		}

		for (int l = static_cast<int>(_hiddenLayers.size()) - 2; l >= 0; l--) {
			int prevLayerIndex = l + 1;

			for (int n = 0; n < _hiddenLayers[l].size(); n++) {
				float sum = 0.0f;

				for (int w = 0; w < _hiddenLayers[prevLayerIndex].size(); w++)
					sum += _hiddenLayers[prevLayerIndex][w]._error * _hiddenLayers[prevLayerIndex][w]._connections[n]._weight;

				_hiddenLayers[l][n]._error = sum * _hiddenLayers[l][n]._output * (1.0f - _hiddenLayers[l][n]._output);
			}
		}

		// Move along gradient
		for (int n = 0; n < _outputLayer.size(); n++) {
			for (int w = 0; w < _outputLayer[n]._connections.size(); w++) {
				float dWeight = alpha * _outputLayer[n]._error * _hiddenLayers.back()[w]._output + momentum * _outputLayer[n]._connections[w]._prevDWeight;
				_outputLayer[n]._connections[w]._weight += dWeight;
				_outputLayer[n]._connections[w]._prevDWeight = dWeight;
			}

			float dBias = alpha * _outputLayer[n]._error + momentum * _outputLayer[n]._bias._prevDWeight;
			_outputLayer[n]._bias._weight += dBias;
			_outputLayer[n]._bias._prevDWeight = dBias;
		}

		for (int l = static_cast<int>(_hiddenLayers.size()) - 1; l >= 1; l--) {
			int prevLayerIndex = l - 1;

			for (int n = 0; n < _hiddenLayers[l].size(); n++) {
				for (int w = 0; w < _hiddenLayers[l][n]._connections.size(); w++) {
					float dWeight = alpha * _hiddenLayers[l][n]._error * _hiddenLayers[prevLayerIndex][w]._output + momentum * _hiddenLayers[l][n]._connections[w]._prevDWeight;
					_hiddenLayers[l][n]._connections[w]._weight += dWeight;
					_hiddenLayers[l][n]._connections[w]._prevDWeight = dWeight;
				}

				float dBias = alpha * _hiddenLayers[l][n]._error + momentum * _hiddenLayers[l][n]._bias._prevDWeight;
				_hiddenLayers[l][n]._bias._weight += dBias;
				_hiddenLayers[l][n]._bias._prevDWeight = dBias;
			}
		}

		for (int n = 0; n < _hiddenLayers[0].size(); n++) {
			for (int w = 0; w < _hiddenLayers[0][n]._connections.size(); w++) {
				float dWeight = alpha * _hiddenLayers[0][n]._error * inputs[w] + momentum * _hiddenLayers[0][n]._connections[w]._prevDWeight;
				_hiddenLayers[0][n]._connections[w]._weight += dWeight;
				_hiddenLayers[0][n]._connections[w]._prevDWeight = dWeight;
			}

			float dBias = alpha * _hiddenLayers[0][n]._error + momentum * _hiddenLayers[0][n]._bias._prevDWeight;
			_hiddenLayers[0][n]._bias._weight += dBias;
			_hiddenLayers[0][n]._bias._prevDWeight = dBias;
		}
	}
	else {
		// Move along gradient
		for (int n = 0; n < _outputLayer.size(); n++) {
			for (int w = 0; w < _outputLayer[n]._connections.size(); w++) {
				float dWeight = alpha * _outputLayer[n]._error * inputs[w] + momentum * _outputLayer[n]._connections[w]._prevDWeight;
				_outputLayer[n]._connections[w]._weight += dWeight;
				_outputLayer[n]._connections[w]._prevDWeight = dWeight;
			}

			float dBias = alpha * _outputLayer[n]._error + momentum * _outputLayer[n]._bias._prevDWeight;
			_outputLayer[n]._bias._weight += dBias;
			_outputLayer[n]._bias._prevDWeight = dBias;
		}
	}
}

void FA::clearGradient() {
	for (int l = 0; l < _hiddenLayers.size(); l++)
	for (int n = 0; n < _hiddenLayers[l].size(); n++) {
		for (int w = 0; w < _hiddenLayers[l][n]._connections.size(); w++)
			_hiddenLayers[l][n]._connections[w]._grad = 0.0f;

		_hiddenLayers[l][n]._bias._grad = 0.0f;
	}

	for (int n = 0; n < _outputLayer.size(); n++) {
		for (int w = 0; w < _outputLayer[n]._connections.size(); w++)
			_outputLayer[n]._connections[w]._grad = 0.0f;

		_outputLayer[n]._bias._grad = 0.0f;
	}
}

void FA::scaleGradient(float scalar) {
	for (int l = 0; l < _hiddenLayers.size(); l++)
	for (int n = 0; n < _hiddenLayers[l].size(); n++) {
		for (int w = 0; w < _hiddenLayers[l][n]._connections.size(); w++)
			_hiddenLayers[l][n]._connections[w]._grad *= scalar;

		_hiddenLayers[l][n]._bias._grad *= scalar;
	}

	for (int n = 0; n < _outputLayer.size(); n++) {
		for (int w = 0; w < _outputLayer[n]._connections.size(); w++)
			_outputLayer[n]._connections[w]._grad *= scalar;

		_outputLayer[n]._bias._grad *= scalar;
	}
}

void FA::accumulateGradient(const std::vector<float> &inputs, const std::vector<float> &targetOutputs) {
	// Output layer error
	for (int n = 0; n < _outputLayer.size(); n++)
		_outputLayer[n]._error = targetOutputs[n] - _outputLayer[n]._output;

	if (!_hiddenLayers.empty())  {
		// Last hidden layer
		for (int n = 0; n < _hiddenLayers.back().size(); n++) {
			float sum = 0.0f;

			for (int w = 0; w < _outputLayer.size(); w++)
				sum += _outputLayer[w]._error * _outputLayer[w]._connections[n]._weight;

			_hiddenLayers.back()[n]._error = sum * _hiddenLayers.back()[n]._output * (1.0f - _hiddenLayers.back()[n]._output);
		}

		for (int l = static_cast<int>(_hiddenLayers.size()) - 2; l >= 0; l--) {
			int prevLayerIndex = l + 1;

			for (int n = 0; n < _hiddenLayers[l].size(); n++) {
				float sum = 0.0f;

				for (int w = 0; w < _hiddenLayers[prevLayerIndex].size(); w++)
					sum += _hiddenLayers[prevLayerIndex][w]._error * _hiddenLayers[prevLayerIndex][w]._connections[n]._weight;

				_hiddenLayers[l][n]._error = sum * _hiddenLayers[l][n]._output * (1.0f - _hiddenLayers[l][n]._output);
			}
		}

		// Get gradient
		for (int n = 0; n < _outputLayer.size(); n++) {
			for (int w = 0; w < _outputLayer[n]._connections.size(); w++)
				_outputLayer[n]._connections[w]._grad += _outputLayer[n]._error * _hiddenLayers.back()[w]._output;
			
			_outputLayer[n]._bias._grad += _outputLayer[n]._error;
		}

		for (int l = static_cast<int>(_hiddenLayers.size()) - 1; l >= 1; l--) {
			int prevLayerIndex = l - 1;

			for (int n = 0; n < _hiddenLayers[l].size(); n++) {
				for (int w = 0; w < _hiddenLayers[l][n]._connections.size(); w++)
					_hiddenLayers[l][n]._connections[w]._grad += _hiddenLayers[l][n]._error * _hiddenLayers[prevLayerIndex][w]._output;

				_hiddenLayers[l][n]._bias._grad += _hiddenLayers[l][n]._error;
			}
		}

		for (int n = 0; n < _hiddenLayers[0].size(); n++) {
			for (int w = 0; w < _hiddenLayers[0][n]._connections.size(); w++)
				_hiddenLayers[0][n]._connections[w]._grad += _hiddenLayers[0][n]._error * inputs[w];

			_hiddenLayers[0][n]._bias._grad += _hiddenLayers[0][n]._error;
		}
	}
	else {
		// Get gradient
		for (int n = 0; n < _outputLayer.size(); n++) {
			for (int w = 0; w < _outputLayer[n]._connections.size(); w++)
				_outputLayer[n]._connections[w]._grad += _outputLayer[n]._error * inputs[w];

			_outputLayer[n]._bias._grad += _outputLayer[n]._error;
		}
	}
}

void FA::moveAlongGradientRMS(float rmsDecay, float alpha, float momentum) {
	if (!_hiddenLayers.empty())  {
		// Move along gradient
		for (int n = 0; n < _outputLayer.size(); n++) {
			for (int w = 0; w < _outputLayer[n]._connections.size(); w++) {
				float grad = _outputLayer[n]._connections[w]._grad;

				_outputLayer[n]._connections[w]._learningRate = (1.0f - rmsDecay) * _outputLayer[n]._connections[w]._learningRate + rmsDecay * grad * grad;

				float dWeight = alpha * grad / std::sqrt(_outputLayer[n]._connections[w]._learningRate) + momentum * _outputLayer[n]._connections[w]._prevDWeight;
				_outputLayer[n]._connections[w]._weight += dWeight;
				_outputLayer[n]._connections[w]._prevDWeight = dWeight;
			}

			float grad = _outputLayer[n]._bias._grad;

			_outputLayer[n]._bias._learningRate = (1.0f - rmsDecay) * _outputLayer[n]._bias._learningRate + rmsDecay * grad * grad;

			float dBias = alpha * grad / std::sqrt(_outputLayer[n]._bias._learningRate) + momentum * _outputLayer[n]._bias._prevDWeight;
			_outputLayer[n]._bias._weight += dBias;
			_outputLayer[n]._bias._prevDWeight = dBias;
		}

		for (int l = static_cast<int>(_hiddenLayers.size()) - 1; l >= 0; l--) {
			for (int n = 0; n < _hiddenLayers[l].size(); n++) {
				for (int w = 0; w < _hiddenLayers[l][n]._connections.size(); w++) {
					float grad = _hiddenLayers[l][n]._connections[w]._grad;

					_hiddenLayers[l][n]._connections[w]._learningRate = (1.0f - rmsDecay) * _hiddenLayers[l][n]._connections[w]._learningRate + rmsDecay * grad * grad;

					float dWeight = alpha * grad / std::sqrt(_hiddenLayers[l][n]._connections[w]._learningRate) + momentum * _hiddenLayers[l][n]._connections[w]._prevDWeight;
					_hiddenLayers[l][n]._connections[w]._weight += dWeight;
					_hiddenLayers[l][n]._connections[w]._prevDWeight = dWeight;
				}

				float grad = _hiddenLayers[l][n]._bias._grad;

				_hiddenLayers[l][n]._bias._learningRate = (1.0f - rmsDecay) * _hiddenLayers[l][n]._bias._learningRate + rmsDecay * grad * grad;

				float dBias = alpha * grad / std::sqrt(_hiddenLayers[l][n]._bias._learningRate) + momentum * _hiddenLayers[l][n]._bias._prevDWeight;
				_hiddenLayers[l][n]._bias._weight += dBias;
				_hiddenLayers[l][n]._bias._prevDWeight = dBias;
			}
		}
	}
	else {
		// Move along gradient
		for (int n = 0; n < _outputLayer.size(); n++) {
			for (int w = 0; w < _outputLayer[n]._connections.size(); w++) {
				float grad = _outputLayer[n]._connections[w]._grad;

				_outputLayer[n]._connections[w]._learningRate = (1.0f - rmsDecay) * _outputLayer[n]._connections[w]._learningRate + rmsDecay * grad * grad;

				float dWeight = alpha * grad / std::sqrt(_outputLayer[n]._connections[w]._learningRate) + momentum * _outputLayer[n]._connections[w]._prevDWeight;
				_outputLayer[n]._connections[w]._weight += dWeight;
				_outputLayer[n]._connections[w]._prevDWeight = dWeight;
			}

			float grad = _outputLayer[n]._bias._grad;

			_outputLayer[n]._bias._learningRate = (1.0f - rmsDecay) * _outputLayer[n]._bias._learningRate + rmsDecay * grad * grad;

			float dBias = alpha * grad / std::sqrt(_outputLayer[n]._bias._learningRate) + momentum * _outputLayer[n]._bias._prevDWeight;
			_outputLayer[n]._bias._weight += dBias;
			_outputLayer[n]._bias._prevDWeight = dBias;
		}
	}
}

void FA::adapt(const std::vector<float> &inputs, const std::vector<float> &targetOutputs, float alpha, float error, float eligibilityDecay, float momentum) {
	// Move along eligibility traces
	for (int n = 0; n < _outputLayer.size(); n++) {
		for (int w = 0; w < _outputLayer[n]._connections.size(); w++) {
			float dWeight = error * _outputLayer[n]._connections[w]._eligibility + momentum * _outputLayer[n]._connections[w]._prevDWeight;
			_outputLayer[n]._connections[w]._weight += dWeight;
			_outputLayer[n]._connections[w]._prevDWeight = dWeight;
		}

		float dBias = error * _outputLayer[n]._bias._eligibility + momentum * _outputLayer[n]._bias._prevDWeight;
		_outputLayer[n]._bias._weight += dBias;
		_outputLayer[n]._bias._prevDWeight = dBias;
	}

	for (int l = 0; l < _hiddenLayers.size(); l++)
	for (int n = 0; n < _hiddenLayers[l].size(); n++) {
		for (int w = 0; w < _hiddenLayers[l][n]._connections.size(); w++) {
			float dWeight = error * _hiddenLayers[l][n]._connections[w]._eligibility + momentum * _hiddenLayers[l][n]._connections[w]._prevDWeight;
			_hiddenLayers[l][n]._connections[w]._weight += dWeight;
			_hiddenLayers[l][n]._connections[w]._prevDWeight = dWeight;
		}

		float dBias = error * _hiddenLayers[l][n]._bias._eligibility + momentum * _hiddenLayers[l][n]._bias._prevDWeight;
		_hiddenLayers[l][n]._bias._weight += dBias;
		_hiddenLayers[l][n]._bias._prevDWeight = dBias;
	}

	// Output layer error
	for (int n = 0; n < _outputLayer.size(); n++)
		_outputLayer[n]._error = targetOutputs[n] - _outputLayer[n]._output;

	if (!_hiddenLayers.empty())  {
		// Last hidden layer
		for (int n = 0; n < _hiddenLayers.back().size(); n++) {
			float sum = 0.0f;

			for (int w = 0; w < _outputLayer.size(); w++)
				sum += _outputLayer[w]._error * _outputLayer[w]._connections[n]._weight;

			_hiddenLayers.back()[n]._error = sum * _hiddenLayers.back()[n]._output * (1.0f - _hiddenLayers.back()[n]._output);
		}

		for (int l = static_cast<int>(_hiddenLayers.size()) - 2; l >= 0; l--) {
			int prevLayerIndex = l + 1;

			for (int n = 0; n < _hiddenLayers[l].size(); n++) {
				float sum = 0.0f;

				for (int w = 0; w < _hiddenLayers[prevLayerIndex].size(); w++)
					sum += _hiddenLayers[prevLayerIndex][w]._error * _hiddenLayers[prevLayerIndex][w]._connections[n]._weight;

				_hiddenLayers[l][n]._error = sum * _hiddenLayers[l][n]._output * (1.0f - _hiddenLayers[l][n]._output);
			}
		}

		// Move along gradient
		for (int n = 0; n < _outputLayer.size(); n++) {
			for (int w = 0; w < _outputLayer[n]._connections.size(); w++)
				_outputLayer[n]._connections[w]._eligibility += -eligibilityDecay * _outputLayer[n]._connections[w]._eligibility + alpha * _outputLayer[n]._error * _hiddenLayers.back()[w]._output;

			_outputLayer[n]._bias._eligibility += -eligibilityDecay * _outputLayer[n]._bias._eligibility + alpha * _outputLayer[n]._error;
		}

		for (int l = static_cast<int>(_hiddenLayers.size()) - 1; l >= 1; l--) {
			int prevLayerIndex = l - 1;

			for (int n = 0; n < _hiddenLayers[l].size(); n++) {
				for (int w = 0; w < _hiddenLayers[l][n]._connections.size(); w++)
					_hiddenLayers[l][n]._connections[w]._eligibility += -eligibilityDecay * _hiddenLayers[l][n]._connections[w]._eligibility + alpha * _hiddenLayers[l][n]._error * _hiddenLayers[prevLayerIndex][w]._output;

				_hiddenLayers[l][n]._bias._eligibility += -eligibilityDecay * _hiddenLayers[l][n]._bias._eligibility + alpha * _hiddenLayers[l][n]._error;
			}
		}

		for (int n = 0; n < _hiddenLayers[0].size(); n++) {
			for (int w = 0; w < _hiddenLayers[0][n]._connections.size(); w++)
				_hiddenLayers[0][n]._connections[w]._eligibility += -eligibilityDecay * _hiddenLayers[0][n]._connections[w]._eligibility + alpha * _hiddenLayers[0][n]._error * inputs[w];

			_hiddenLayers[0][n]._bias._eligibility += -eligibilityDecay * _hiddenLayers[0][n]._bias._eligibility + alpha * _hiddenLayers[0][n]._error;
		}
	}
	else {
		// Move along gradient
		for (int n = 0; n < _outputLayer.size(); n++) {
			for (int w = 0; w < _outputLayer[n]._connections.size(); w++)
				_outputLayer[n]._connections[w]._eligibility += -eligibilityDecay * _outputLayer[n]._connections[w]._eligibility + alpha * _outputLayer[n]._error * inputs[w];

			_outputLayer[n]._bias._eligibility += -eligibilityDecay * _outputLayer[n]._bias._eligibility + alpha * _outputLayer[n]._error;
		}
	}
}

void FA::decayWeights(float decayMultiplier) {
	for (int l = 0; l < _hiddenLayers.size(); l++)
	for (int n = 0; n < _hiddenLayers[l].size(); n++) {
		for (int w = 0; w < _hiddenLayers[l][n]._connections.size(); w++)
			_hiddenLayers[l][n]._connections[w]._weight *= decayMultiplier;

		_hiddenLayers[l][n]._bias._weight *= decayMultiplier;
	}

	for (int n = 0; n < _outputLayer.size(); n++) {
		for (int w = 0; w < _outputLayer[n]._connections.size(); w++)
			_outputLayer[n]._connections[w]._weight *= decayMultiplier;

		_outputLayer[n]._bias._weight *= decayMultiplier;
	}
}

void FA::writeToStream(std::ostream &os) const {
	os << getNumInputs() << " " << getNumOutputs() << " " << getNumHiddenLayers() << " " << getNumNeuronsPerHiddenLayer() << std::endl;

	for (int l = 0; l < _hiddenLayers.size(); l++)
	for (int n = 0; n < _hiddenLayers[l].size(); n++) {
		for (int w = 0; w < _hiddenLayers[l][n]._connections.size(); w++)
			os << _hiddenLayers[l][n]._connections[w]._weight << " ";

		os << _hiddenLayers[l][n]._bias._weight << std::endl;
	}

	for (int n = 0; n < _outputLayer.size(); n++) {
		for (int w = 0; w < _outputLayer[n]._connections.size(); w++)
			os << _outputLayer[n]._connections[w]._weight << " ";

		os << _outputLayer[n]._bias._weight << std::endl;
	}
}

void FA::readFromStream(std::istream &is) {
	int numInputs, numOutputs, numHiddenLayers, numNeuronsPerHiddenLayer;

	is >> numInputs >> numOutputs >> numHiddenLayers >> numNeuronsPerHiddenLayer;

	if (numHiddenLayers > 0) {
		_hiddenLayers.resize(numHiddenLayers);

		// First hidden layer
		_hiddenLayers[0].resize(numNeuronsPerHiddenLayer);

		for (int n = 0; n < _hiddenLayers[0].size(); n++) {
			_hiddenLayers[0][n]._connections.resize(numInputs);

			for (int w = 0; w < _hiddenLayers[0][n]._connections.size(); w++)
				is >> _hiddenLayers[0][n]._connections[w]._weight;

			is >> _hiddenLayers[0][n]._bias._weight;
		}

		// All other hidden layers
		for (int l = 1; l < _hiddenLayers.size(); l++) {
			_hiddenLayers[l].resize(numNeuronsPerHiddenLayer);

			for (int n = 0; n < _hiddenLayers[l].size(); n++) {
				_hiddenLayers[l][n]._connections.resize(numNeuronsPerHiddenLayer);

				for (int w = 0; w < _hiddenLayers[l][n]._connections.size(); w++)
					is >> _hiddenLayers[l][n]._connections[w]._weight;

				is >> _hiddenLayers[l][n]._bias._weight;
			}
		}

		_outputLayer.resize(numOutputs);

		for (int n = 0; n < _outputLayer.size(); n++) {
			_outputLayer[n]._connections.resize(numNeuronsPerHiddenLayer);

			for (int w = 0; w < _outputLayer[n]._connections.size(); w++)
				is >> _outputLayer[n]._connections[w]._weight;

			is >> _outputLayer[n]._bias._weight;
		}
	}
	else {
		_outputLayer.resize(numOutputs);

		for (int n = 0; n < _outputLayer.size(); n++) {
			_outputLayer[n]._connections.resize(numInputs);

			for (int w = 0; w < _outputLayer[n]._connections.size(); w++)
				is >> _outputLayer[n]._connections[w]._weight;

			is >> _outputLayer[n]._bias._weight;
		}
	}
}