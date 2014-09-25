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

#include <hypernet/FunctionApproximator.h>

using namespace hn;

void FunctionApproximator::createRandom(size_t numInputs, size_t numOutputs, size_t numHiddenLayers, size_t numNeuronsPerHiddenLayer, float minWeight, float maxWeight, std::mt19937 &generator) {
	std::uniform_real_distribution<float> distWeight(minWeight, maxWeight);

	if (numHiddenLayers > 0) {
		_hiddenLayers.resize(numHiddenLayers);

		// First hidden layer
		_hiddenLayers[0].resize(numNeuronsPerHiddenLayer);

		for (size_t n = 0; n < _hiddenLayers[0].size(); n++) {
			_hiddenLayers[0][n]._weights.resize(numInputs);

			for (size_t w = 0; w < _hiddenLayers[0][n]._weights.size(); w++)
				_hiddenLayers[0][n]._weights[w] = distWeight(generator);

			_hiddenLayers[0][n]._bias = distWeight(generator);
		}

		// All other hidden layers
		for (size_t l = 1; l < _hiddenLayers.size(); l++) {
			_hiddenLayers[l].resize(numNeuronsPerHiddenLayer);

			for (size_t n = 0; n < _hiddenLayers[l].size(); n++) {
				_hiddenLayers[l][n]._weights.resize(numNeuronsPerHiddenLayer);

				for (size_t w = 0; w < _hiddenLayers[l][n]._weights.size(); w++)
					_hiddenLayers[l][n]._weights[w] = distWeight(generator);

				_hiddenLayers[l][n]._bias = distWeight(generator);
			}
		}

		_outputLayer.resize(numOutputs);

		for (size_t n = 0; n < _outputLayer.size(); n++) {
			_outputLayer[n]._weights.resize(numNeuronsPerHiddenLayer);

			for (size_t w = 0; w < _outputLayer[n]._weights.size(); w++)
				_outputLayer[n]._weights[w] = distWeight(generator);

			_outputLayer[n]._bias = distWeight(generator);
		}
	}
	else {
		_outputLayer.resize(numOutputs);

		for (size_t n = 0; n < _outputLayer.size(); n++) {
			_outputLayer[n]._weights.resize(numInputs);

			for (size_t w = 0; w < _outputLayer[n]._weights.size(); w++)
				_outputLayer[n]._weights[w] = distWeight(generator);

			_outputLayer[n]._bias = distWeight(generator);
		}
	}
}

float FunctionApproximator::crossoverChooseWeight(float w1, float w2, float averageChance, std::mt19937 &generator) {
	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

	if (dist01(generator) < averageChance)
		return (w1 + w2) * 0.5f;
	
	return dist01(generator) < 0.5f ? w1 : w2;
}

void FunctionApproximator::createFromParents(const FunctionApproximator &parent1, const FunctionApproximator &parent2, float averageChance, std::mt19937 &generator) {
	size_t numInputs = parent1.getNumInputs();
	size_t numOutputs = parent1.getNumOutputs();
	size_t numHiddenLayers = parent1.getNumHiddenLayers();
	size_t numNeuronsPerHiddenLayer = parent1.getNumNeuronsPerHiddenLayer();

	if (numHiddenLayers > 0) {
		_hiddenLayers.resize(numHiddenLayers);

		// First hidden layer
		_hiddenLayers[0].resize(numNeuronsPerHiddenLayer);

		for (size_t n = 0; n < _hiddenLayers[0].size(); n++) {
			_hiddenLayers[0][n]._weights.resize(numInputs);

			for (size_t w = 0; w < _hiddenLayers[0][n]._weights.size(); w++)
				_hiddenLayers[0][n]._weights[w] = crossoverChooseWeight(parent1._hiddenLayers[0][n]._weights[w], parent2._hiddenLayers[0][n]._weights[w], averageChance, generator);

			_hiddenLayers[0][n]._bias = crossoverChooseWeight(parent1._hiddenLayers[0][n]._bias, parent2._hiddenLayers[0][n]._bias, averageChance, generator);
		}

		// All other hidden layers
		for (size_t l = 1; l < _hiddenLayers.size(); l++) {
			_hiddenLayers[l].resize(numNeuronsPerHiddenLayer);

			for (size_t n = 0; n < _hiddenLayers[l].size(); n++) {
				_hiddenLayers[l][n]._weights.resize(numNeuronsPerHiddenLayer);

				for (size_t w = 0; w < _hiddenLayers[l][n]._weights.size(); w++)
					_hiddenLayers[l][n]._weights[w] = crossoverChooseWeight(parent1._hiddenLayers[l][n]._weights[w], parent2._hiddenLayers[l][n]._weights[w], averageChance, generator);

				_hiddenLayers[l][n]._bias = crossoverChooseWeight(parent1._hiddenLayers[l][n]._bias, parent2._hiddenLayers[l][n]._bias, averageChance, generator);
			}
		}

		_outputLayer.resize(numOutputs);

		for (size_t n = 0; n < _outputLayer.size(); n++) {
			_outputLayer[n]._weights.resize(numNeuronsPerHiddenLayer);

			for (size_t w = 0; w < _outputLayer[n]._weights.size(); w++)
				_outputLayer[n]._weights[w] = crossoverChooseWeight(parent1._outputLayer[n]._weights[w], parent2._outputLayer[n]._weights[w], averageChance, generator);

			_outputLayer[n]._bias = crossoverChooseWeight(parent1._outputLayer[n]._bias, parent2._outputLayer[n]._bias, averageChance, generator);
		}
	}
	else {
		_outputLayer.resize(numOutputs);

		for (size_t n = 0; n < _outputLayer.size(); n++) {
			_outputLayer[n]._weights.resize(numInputs);

			for (size_t w = 0; w < _outputLayer[n]._weights.size(); w++)
				_outputLayer[n]._weights[w] = crossoverChooseWeight(parent1._outputLayer[n]._weights[w], parent2._outputLayer[n]._weights[w], averageChance, generator);

			_outputLayer[n]._bias = crossoverChooseWeight(parent1._outputLayer[n]._bias, parent2._outputLayer[n]._bias, averageChance, generator);
		}
	}
}

void FunctionApproximator::mutate(float perturbationChance, float perturbationStdDev, std::mt19937 &generator) {
	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
	std::normal_distribution<float> distPerturbation(0.0f, perturbationStdDev);

	for (size_t l = 0; l < _hiddenLayers.size(); l++)
	for (size_t n = 0; n < _hiddenLayers[l].size(); n++) {
		for (size_t w = 0; w < _hiddenLayers[l][n]._weights.size(); w++)
			_hiddenLayers[l][n]._weights[w] += dist01(generator) < perturbationChance ? distPerturbation(generator) : 0.0f;

		_hiddenLayers[l][n]._bias += dist01(generator) < perturbationChance ? distPerturbation(generator) : 0.0f;
	}

	for (size_t n = 0; n < _outputLayer.size(); n++) {
		for (size_t w = 0; w < _outputLayer[n]._weights.size(); w++)
			_outputLayer[n]._weights[w] += dist01(generator) < perturbationChance ? distPerturbation(generator) : 0.0f;

		_outputLayer[n]._bias += dist01(generator) < perturbationChance ? distPerturbation(generator) : 0.0f;
	}
}

size_t FunctionApproximator::createFromWeightsVector(size_t numInputs, size_t numOutputs, size_t numHiddenLayers, size_t numNeuronsPerHiddenLayer, const std::vector<float> &weights, size_t startIndex) {
	size_t weightIndex = startIndex;
	
	if (numHiddenLayers > 0) {
		_hiddenLayers.resize(numHiddenLayers);

		// First hidden layer
		_hiddenLayers[0].resize(numNeuronsPerHiddenLayer);

		for (size_t n = 0; n < _hiddenLayers[0].size(); n++) {
			_hiddenLayers[0][n]._weights.resize(numInputs);

			for (size_t w = 0; w < _hiddenLayers[0][n]._weights.size(); w++)
				_hiddenLayers[0][n]._weights[w] = weights[weightIndex++];

			_hiddenLayers[0][n]._bias = weights[weightIndex++];
		}

		// All other hidden layers
		for (size_t l = 1; l < _hiddenLayers.size(); l++) {
			_hiddenLayers[l].resize(numNeuronsPerHiddenLayer);

			for (size_t n = 0; n < _hiddenLayers[l].size(); n++) {
				_hiddenLayers[l][n]._weights.resize(numNeuronsPerHiddenLayer);

				for (size_t w = 0; w < _hiddenLayers[l][n]._weights.size(); w++)
					_hiddenLayers[l][n]._weights[w] = weights[weightIndex++];

				_hiddenLayers[l][n]._bias = weights[weightIndex++];
			}
		}

		_outputLayer.resize(numOutputs);

		for (size_t n = 0; n < _outputLayer.size(); n++) {
			_outputLayer[n]._weights.resize(numNeuronsPerHiddenLayer);

			for (size_t w = 0; w < _outputLayer[n]._weights.size(); w++)
				_outputLayer[n]._weights[w] = weights[weightIndex++];

			_outputLayer[n]._bias = weights[weightIndex++];
		}
	}
	else {
		_outputLayer.resize(numOutputs);

		for (size_t n = 0; n < _outputLayer.size(); n++) {
			_outputLayer[n]._weights.resize(numInputs);

			for (size_t w = 0; w < _outputLayer[n]._weights.size(); w++)
				_outputLayer[n]._weights[w] = weights[weightIndex++];

			_outputLayer[n]._bias = weights[weightIndex++];
		}
	}

	return weightIndex;
}

void FunctionApproximator::getWeightsVector(std::vector<float> &weights) {
	for (size_t l = 0; l < _hiddenLayers.size(); l++)
	for (size_t n = 0; n < _hiddenLayers[l].size(); n++) {
		for (size_t w = 0; w < _hiddenLayers[l][n]._weights.size(); w++)
			weights.push_back(_hiddenLayers[l][n]._weights[w]);

		weights.push_back(_hiddenLayers[l][n]._bias);
	}

	for (size_t n = 0; n < _outputLayer.size(); n++) {
		for (size_t w = 0; w < _outputLayer[n]._weights.size(); w++)
			weights.push_back(_outputLayer[n]._weights[w]);

		weights.push_back(_outputLayer[n]._bias);
	}
}

void FunctionApproximator::process(const std::vector<float> &inputs, std::vector<float> &outputs, float activationMultiplier) {
	if (!_hiddenLayers.empty()) {
		std::vector<float> tempInputs;
		
		tempInputs.resize(_hiddenLayers[0].size());

		// First hidden layer
		for (size_t n = 0; n < _hiddenLayers[0].size(); n++) {
			float sum = _hiddenLayers[0][n]._bias;

			for (size_t w = 0; w < _hiddenLayers[0][n]._weights.size(); w++)
				sum += inputs[w] * _hiddenLayers[0][n]._weights[w];

			tempInputs[n] = sigmoid(sum * activationMultiplier);
		}

		if (_hiddenLayers.size() > 1) {
			std::vector<float> tempOutputs;

			tempOutputs.resize(_hiddenLayers[0].size());

			// All other hidden layers
			for (size_t l = 1; l < _hiddenLayers.size(); l++) {
				for (size_t n = 0; n < _hiddenLayers[l].size(); n++) {
					float sum = _hiddenLayers[l][n]._bias;

					for (size_t w = 0; w < _hiddenLayers[l][n]._weights.size(); w++)
						sum += tempInputs[w] * _hiddenLayers[l][n]._weights[w];

					tempOutputs[n] = sigmoid(sum * activationMultiplier);
				}

				tempInputs = tempOutputs;
			}
		}

		// Output layer
		for (size_t n = 0; n < _outputLayer.size(); n++) {
			float sum = _outputLayer[n]._bias;

			for (size_t w = 0; w < _outputLayer[n]._weights.size(); w++)
				sum += tempInputs[w] * _outputLayer[n]._weights[w];

			outputs[n] = sum * activationMultiplier; // Linear activation
		}
	}
	else {
		// Output layer
		for (size_t n = 0; n < _outputLayer.size(); n++) {
			float sum = _outputLayer[n]._bias;

			for (size_t w = 0; w < _outputLayer[n]._weights.size(); w++)
				sum += inputs[w] * _outputLayer[n]._weights[w];

			outputs[n] = sum * activationMultiplier; // Linear activation
		}
	}
}

void FunctionApproximator::process(const std::vector<float> &inputs, std::vector<std::vector<float>> &layerOutputs, float activationMultiplier) {
	layerOutputs.resize(_hiddenLayers.size() + 1);
	
	if (!_hiddenLayers.empty()) {
		layerOutputs[0].resize(_hiddenLayers[0].size());

		// First hidden layer
		for (size_t n = 0; n < _hiddenLayers[0].size(); n++) {
			float sum = _hiddenLayers[0][n]._bias;

			for (size_t w = 0; w < _hiddenLayers[0][n]._weights.size(); w++)
				sum += inputs[w] * _hiddenLayers[0][n]._weights[w];

			layerOutputs[0][n] = sigmoid(sum * activationMultiplier);
		}

		// All other hidden layers
		for (size_t l = 1; l < _hiddenLayers.size(); l++) {
			layerOutputs[l].resize(_hiddenLayers[l].size());

			for (size_t n = 0; n < _hiddenLayers[l].size(); n++) {
				float sum = _hiddenLayers[l][n]._bias;

				for (size_t w = 0; w < _hiddenLayers[l][n]._weights.size(); w++)
					sum += layerOutputs[l - 1][w] * _hiddenLayers[l][n]._weights[w];

				layerOutputs[l][n] = sigmoid(sum * activationMultiplier);
			}
		}

		// Output layer
		layerOutputs.back().resize(_outputLayer.size());

		for (size_t n = 0; n < _outputLayer.size(); n++) {
			float sum = _outputLayer[n]._bias;

			for (size_t w = 0; w < _outputLayer[n]._weights.size(); w++)
				sum += layerOutputs[_hiddenLayers.size() - 1][w] * _outputLayer[n]._weights[w];

			layerOutputs.back()[n] = sum * activationMultiplier; // Linear activation
		}
	}
	else {
		layerOutputs[0].resize(_outputLayer.size());

		// Output layer
		for (size_t n = 0; n < _outputLayer.size(); n++) {
			float sum = _outputLayer[n]._bias;

			for (size_t w = 0; w < _outputLayer[n]._weights.size(); w++)
				sum += inputs[w] * _outputLayer[n]._weights[w];

			layerOutputs[0][n] = sum * activationMultiplier; // Linear activation
		}
	}
}

void FunctionApproximator::backpropagate(const std::vector<float> &inputs, const std::vector<std::vector<float>> &layerOutputs, const std::vector<float> &targetOutputs, float alpha) {
	// Output layer error
	std::vector<std::vector<float>> errors = layerOutputs;

	for (size_t n = 0; n < _outputLayer.size(); n++)
		errors.back()[n] = targetOutputs[n] - layerOutputs.back()[n];

	// Last hidden layer
	if (!_hiddenLayers.empty())  {
		for (size_t n = 0; n < _hiddenLayers.back().size(); n++) {
			float sum = 0.0f;

			for (size_t w = 0; w < _outputLayer.size(); w++)
				sum += errors.back()[w] * _outputLayer[w]._weights[n];

			errors[errors.size() - 2][n] = sum * layerOutputs[layerOutputs.size() - 2][n] * (1.0f - layerOutputs[layerOutputs.size() - 2][n]);
		}

		for (int l = static_cast<int>(_hiddenLayers.size()) - 2; l >= 0; l--) {
			for (size_t n = 0; n < _hiddenLayers[l].size(); n++) {
				float sum = 0.0f;

				for (size_t w = 0; w < _hiddenLayers[l + 1].size(); w++)
					sum += errors[l + 1][w] * _hiddenLayers[l + 1][w]._weights[n];

				errors[l][n] = sum * layerOutputs[l][n] * (1.0f - layerOutputs[l][n]);
			}
		}

		// Move along gradient
		for (size_t n = 0; n < _outputLayer.size(); n++) {
			for (size_t w = 0; w < _outputLayer[n]._weights.size(); w++)
				_outputLayer[n]._weights[w] += alpha * errors.back()[n] * layerOutputs[layerOutputs.size() - 2][w];

			_outputLayer[n]._bias += alpha * errors.back()[n];
		}

		for (int l = static_cast<int>(_hiddenLayers.size()) - 1; l >= 1; l--)
		for (size_t n = 0; n < _hiddenLayers[l].size(); n++) {
			for (size_t w = 0; w < _hiddenLayers[l][n]._weights.size(); w++)
				_hiddenLayers[l][n]._weights[w] += alpha * errors[l][n] * layerOutputs[l - 1][w];
			
			_hiddenLayers[l][n]._bias += alpha * errors[l][n];
		}

		for (size_t n = 0; n < _hiddenLayers[0].size(); n++) {
			for (size_t w = 0; w < _hiddenLayers[0][n]._weights.size(); w++)
				_hiddenLayers[0][n]._weights[w] += alpha * errors[0][n] * inputs[w];

			_hiddenLayers[0][n]._bias += alpha * errors[0][n];
		}
	}
	else {
		// Move along gradient
		for (size_t n = 0; n < _outputLayer.size(); n++) {
			for (size_t w = 0; w < _outputLayer[n]._weights.size(); w++)
				_outputLayer[n]._weights[w] += alpha * errors.back()[n] * inputs[w];

			_outputLayer[n]._bias += alpha * errors.back()[n];
		}
	}
}

void FunctionApproximator::writeToStream(std::ostream &os) const {
	os << getNumInputs() << " " << getNumOutputs() << " " << getNumHiddenLayers() << " " << getNumNeuronsPerHiddenLayer() << std::endl;

	for (size_t l = 0; l < _hiddenLayers.size(); l++)
	for (size_t n = 0; n < _hiddenLayers[l].size(); n++) {
		for (size_t w = 0; w < _hiddenLayers[l][n]._weights.size(); w++)
			os << _hiddenLayers[l][n]._weights[w] << " ";

		os << _hiddenLayers[l][n]._bias << std::endl;
	}

	for (size_t n = 0; n < _outputLayer.size(); n++) {
		for (size_t w = 0; w < _outputLayer[n]._weights.size(); w++)
			os << _outputLayer[n]._weights[w] << " ";

		os << _outputLayer[n]._bias << std::endl;
	}
}

void FunctionApproximator::readFromStream(std::istream &is) {
	size_t numInputs, numOutputs, numHiddenLayers, numNeuronsPerHiddenLayer;

	is >> numInputs >> numOutputs >> numHiddenLayers >> numNeuronsPerHiddenLayer;

	if (numHiddenLayers > 0) {
		_hiddenLayers.resize(numHiddenLayers);

		// First hidden layer
		_hiddenLayers[0].resize(numNeuronsPerHiddenLayer);

		for (size_t n = 0; n < _hiddenLayers[0].size(); n++) {
			_hiddenLayers[0][n]._weights.resize(numInputs);

			for (size_t w = 0; w < _hiddenLayers[0][n]._weights.size(); w++)
				is >> _hiddenLayers[0][n]._weights[w];

			is >> _hiddenLayers[0][n]._bias;
		}

		// All other hidden layers
		for (size_t l = 1; l < _hiddenLayers.size(); l++) {
			_hiddenLayers[l].resize(numNeuronsPerHiddenLayer);

			for (size_t n = 0; n < _hiddenLayers[l].size(); n++) {
				_hiddenLayers[l][n]._weights.resize(numNeuronsPerHiddenLayer);

				for (size_t w = 0; w < _hiddenLayers[l][n]._weights.size(); w++)
					is >> _hiddenLayers[l][n]._weights[w];

				is >> _hiddenLayers[l][n]._bias;
			}
		}

		_outputLayer.resize(numOutputs);

		for (size_t n = 0; n < _outputLayer.size(); n++) {
			_outputLayer[n]._weights.resize(numNeuronsPerHiddenLayer);

			for (size_t w = 0; w < _outputLayer[n]._weights.size(); w++)
				is >> _outputLayer[n]._weights[w];

			is >> _outputLayer[n]._bias;
		}
	}
	else {
		_outputLayer.resize(numOutputs);

		for (size_t n = 0; n < _outputLayer.size(); n++) {
			_outputLayer[n]._weights.resize(numInputs);

			for (size_t w = 0; w < _outputLayer[n]._weights.size(); w++)
				is >> _outputLayer[n]._weights[w];

			is >> _outputLayer[n]._bias;
		}
	}
}