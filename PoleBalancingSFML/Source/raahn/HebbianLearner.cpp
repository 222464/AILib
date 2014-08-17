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

#include <raahn/HebbianLearner.h>

#include <algorithm>
#include <iostream>

using namespace raahn;

void HebbianLearner::createRandom(size_t numInputs, size_t numOutputs, size_t numHiddenLayers, size_t numNeuronsPerHiddenLayer, float minWeight, float maxWeight, std::mt19937 &generator) {
	std::uniform_real_distribution<float> distWeight(minWeight, maxWeight);

	if (numHiddenLayers > 0) {
		_hiddenLayers.resize(numHiddenLayers);

		// First hidden layer
		_hiddenLayers[0].resize(numNeuronsPerHiddenLayer);

		for (size_t n = 0; n < _hiddenLayers[0].size(); n++) {
			_hiddenLayers[0][n]._weights.resize(numInputs);

			for (size_t w = 0; w < _hiddenLayers[0][n]._weights.size(); w++) {
				_hiddenLayers[0][n]._weights[w]._weight = distWeight(generator);
				_hiddenLayers[0][n]._weights[w]._trace = 0.0f;
			}

			_hiddenLayers[0][n]._output = 0.5f;
			_hiddenLayers[0][n]._outputTrace = 0.5f;
	
			_hiddenLayers[0][n]._bias._weight = distWeight(generator);
			_hiddenLayers[0][n]._bias._trace = 0.0f;
		}

		// All other hidden layers
		for (size_t l = 1; l < _hiddenLayers.size(); l++) {
			_hiddenLayers[l].resize(numNeuronsPerHiddenLayer);

			for (size_t n = 0; n < _hiddenLayers[l].size(); n++) {
				_hiddenLayers[l][n]._weights.resize(numNeuronsPerHiddenLayer);

				for (size_t w = 0; w < _hiddenLayers[l][n]._weights.size(); w++) {
					_hiddenLayers[l][n]._weights[w]._weight = distWeight(generator);
					_hiddenLayers[l][n]._weights[w]._trace = 0.0f;
				}

				_hiddenLayers[l][n]._output = 0.5f;
				_hiddenLayers[l][n]._outputTrace = 0.5f;
		
				_hiddenLayers[l][n]._bias._weight = distWeight(generator);
				_hiddenLayers[l][n]._bias._trace = 0.0f;
			}
		}

		_outputLayer.resize(numOutputs);

		for (size_t n = 0; n < _outputLayer.size(); n++) {
			_outputLayer[n]._weights.resize(numNeuronsPerHiddenLayer);

			for (size_t w = 0; w < _outputLayer[n]._weights.size(); w++) {
				_outputLayer[n]._weights[w]._weight = distWeight(generator);
				_outputLayer[n]._weights[w]._trace = 0.0f;
			}

			_outputLayer[n]._output = 0.0f;
			_outputLayer[n]._outputTrace = 0.0f;

			_outputLayer[n]._bias._weight = distWeight(generator);
			_outputLayer[n]._bias._trace = 0.0f;
		}
	}
	else {
		_outputLayer.resize(numOutputs);

		for (size_t n = 0; n < _outputLayer.size(); n++) {
			_outputLayer[n]._weights.resize(numInputs);

			for (size_t w = 0; w < _outputLayer[n]._weights.size(); w++) {
				_outputLayer[n]._weights[w]._weight = distWeight(generator);
				_outputLayer[n]._weights[w]._trace = 0.0f;
			}

			_outputLayer[n]._output = 0.0f;
			_outputLayer[n]._outputTrace = 0.0f;

			_outputLayer[n]._bias._weight = distWeight(generator);
			_outputLayer[n]._bias._trace = 0.0f;
		}
	}
}

float HebbianLearner::crossoverChooseWeight(float w1, float w2, float averageChance, std::mt19937 &generator) {
	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

	if (dist01(generator) < averageChance)
		return (w1 + w2) * 0.5f;

	return dist01(generator) < 0.5f ? w1 : w2;
}

void HebbianLearner::createFromParents(const HebbianLearner &parent1, const HebbianLearner &parent2, float averageChance, std::mt19937 &generator) {
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

			for (size_t w = 0; w < _hiddenLayers[0][n]._weights.size(); w++) {
				_hiddenLayers[0][n]._weights[w]._weight = crossoverChooseWeight(parent1._hiddenLayers[0][n]._weights[w]._weight, parent2._hiddenLayers[0][n]._weights[w]._weight, averageChance, generator);
				_hiddenLayers[0][n]._weights[w]._trace = 0.0f;
			}

			_hiddenLayers[0][n]._output = 0.5f;
			_hiddenLayers[0][n]._outputTrace = 0.5f;

			_hiddenLayers[0][n]._bias._weight = crossoverChooseWeight(parent1._hiddenLayers[0][n]._bias._weight, parent2._hiddenLayers[0][n]._bias._weight, averageChance, generator);
			_hiddenLayers[0][n]._bias._trace = 0.0f;
		}

		// All other hidden layers
		for (size_t l = 1; l < _hiddenLayers.size(); l++) {
			_hiddenLayers[l].resize(numNeuronsPerHiddenLayer);

			for (size_t n = 0; n < _hiddenLayers[l].size(); n++) {
				_hiddenLayers[l][n]._weights.resize(numNeuronsPerHiddenLayer);

				for (size_t w = 0; w < _hiddenLayers[l][n]._weights.size(); w++) {
					_hiddenLayers[l][n]._weights[w]._weight = crossoverChooseWeight(parent1._hiddenLayers[l][n]._weights[w]._weight, parent2._hiddenLayers[l][n]._weights[w]._weight, averageChance, generator);
					_hiddenLayers[l][n]._weights[w]._trace = 0.0f;
				}

				_hiddenLayers[l][n]._output = 0.5f;
				_hiddenLayers[l][n]._outputTrace = 0.5f;

				_hiddenLayers[l][n]._bias._weight = crossoverChooseWeight(parent1._hiddenLayers[l][n]._bias._weight, parent2._hiddenLayers[l][n]._bias._weight, averageChance, generator);
				_hiddenLayers[l][n]._bias._trace = 0.0f;
			}
		}

		_outputLayer.resize(numOutputs);

		for (size_t n = 0; n < _outputLayer.size(); n++) {
			_outputLayer[n]._weights.resize(numNeuronsPerHiddenLayer);

			for (size_t w = 0; w < _outputLayer[n]._weights.size(); w++) {
				_outputLayer[n]._weights[w]._weight = crossoverChooseWeight(parent1._outputLayer[n]._weights[w]._weight, parent2._outputLayer[n]._weights[w]._weight, averageChance, generator);
				_outputLayer[n]._weights[w]._trace = 0.0f;
			}

			_outputLayer[n]._output = 0.0f;
			_outputLayer[n]._outputTrace = 0.0f;

			_outputLayer[n]._bias._weight = crossoverChooseWeight(parent1._outputLayer[n]._bias._weight, parent2._outputLayer[n]._bias._weight, averageChance, generator);
			_outputLayer[n]._bias._trace = 0.0f;
		}
	}
	else {
		_outputLayer.resize(numOutputs);

		for (size_t n = 0; n < _outputLayer.size(); n++) {
			_outputLayer[n]._weights.resize(numInputs);

			for (size_t w = 0; w < _outputLayer[n]._weights.size(); w++) {
				_outputLayer[n]._weights[w]._weight = crossoverChooseWeight(parent1._outputLayer[n]._weights[w]._weight, parent2._outputLayer[n]._weights[w]._weight, averageChance, generator);
				_outputLayer[n]._weights[w]._trace = 0.0f;
			}

			_outputLayer[n]._output = 0.0f;
			_outputLayer[n]._outputTrace = 0.0f;

			_outputLayer[n]._bias._weight = crossoverChooseWeight(parent1._outputLayer[n]._bias._weight, parent2._outputLayer[n]._bias._weight, averageChance, generator);
			_outputLayer[n]._bias._trace = 0.0f;
		}
	}
}

void HebbianLearner::mutate(float perturbationChance, float perturbationStdDev, std::mt19937 &generator) {
	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
	std::normal_distribution<float> distPerturbation(0.0f, perturbationStdDev);

	for (size_t l = 0; l < _hiddenLayers.size(); l++)
	for (size_t n = 0; n < _hiddenLayers[l].size(); n++) {
		for (size_t w = 0; w < _hiddenLayers[l][n]._weights.size(); w++)
			_hiddenLayers[l][n]._weights[w]._weight += dist01(generator) < perturbationChance ? distPerturbation(generator) : 0.0f;

		_hiddenLayers[l][n]._bias._weight += dist01(generator) < perturbationChance ? distPerturbation(generator) : 0.0f;
	}

	for (size_t n = 0; n < _outputLayer.size(); n++) {
		for (size_t w = 0; w < _outputLayer[n]._weights.size(); w++)
			_outputLayer[n]._weights[w]._weight += dist01(generator) < perturbationChance ? distPerturbation(generator) : 0.0f;

		_outputLayer[n]._bias._weight += dist01(generator) < perturbationChance ? distPerturbation(generator) : 0.0f;
	}
}

void HebbianLearner::process(const std::vector<float> &inputs, std::vector<float> &outputs, float modulation, float traceDecay, float outputDecay, float breakRate, std::mt19937 &generator) {
	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
	
	if (!_hiddenLayers.empty()) {
		// First hidden layer
		for (size_t n = 0; n < _hiddenLayers[0].size(); n++) {
			_hiddenLayers[0][n]._bias._weight += modulation * _hiddenLayers[0][n]._bias._trace;

			float sum = _hiddenLayers[0][n]._bias._weight;

			for (size_t w = 0; w < _hiddenLayers[0][n]._weights.size(); w++) {
				_hiddenLayers[0][n]._weights[w]._weight += modulation * _hiddenLayers[0][n]._weights[w]._trace;
				sum += inputs[w] * _hiddenLayers[0][n]._weights[w]._weight;
			}

			float output = sigmoid(sum);

			_hiddenLayers[0][n]._output = dist01(generator) < breakRate ? dist01(generator) : output;

			float delta = _hiddenLayers[0][n]._output - output;

			_hiddenLayers[0][n]._bias._trace = (1.0f - traceDecay) * _hiddenLayers[0][n]._bias._trace + delta;

			for (size_t w = 0; w < _hiddenLayers[0][n]._weights.size(); w++)
				_hiddenLayers[0][n]._weights[w]._trace = (1.0f - traceDecay) * _hiddenLayers[0][n]._weights[w]._trace + delta * inputs[w];

			_hiddenLayers[0][n]._outputTrace += (output - _hiddenLayers[0][n]._outputTrace) * outputDecay;
		}

		if (_hiddenLayers.size() > 1) {
			// All other hidden layers
			for (size_t l = 1; l < _hiddenLayers.size(); l++) {
				for (size_t n = 0; n < _hiddenLayers[l].size(); n++) {
					_hiddenLayers[l][n]._bias._weight += modulation * _hiddenLayers[l][n]._bias._trace;

					float sum = _hiddenLayers[l][n]._bias._weight;

					for (size_t w = 0; w < _hiddenLayers[l][n]._weights.size(); w++) {
						_hiddenLayers[l][n]._weights[w]._weight += modulation * _hiddenLayers[l][n]._weights[w]._trace;
						sum += _hiddenLayers[l - 1][w]._output * _hiddenLayers[l][n]._weights[w]._weight;
					}

					float output = sigmoid(sum);

					_hiddenLayers[l][n]._output = dist01(generator) < breakRate ? dist01(generator) : output;

					float delta = _hiddenLayers[l][n]._output - output;

					_hiddenLayers[l][n]._bias._trace = (1.0f - traceDecay) * _hiddenLayers[l][n]._bias._trace + delta;

					for (size_t w = 0; w < _hiddenLayers[l][n]._weights.size(); w++)
						_hiddenLayers[l][n]._weights[w]._trace = (1.0f - traceDecay) * _hiddenLayers[l][n]._weights[w]._trace + delta * _hiddenLayers[l - 1][w]._output;

					_hiddenLayers[l][n]._outputTrace += (output - _hiddenLayers[0][n]._outputTrace) * outputDecay;
				}
			}
		}

		// Output layer
		for (size_t n = 0; n < _outputLayer.size(); n++) {
			_outputLayer[n]._bias._weight += modulation * _outputLayer[n]._bias._trace;
			std::cout << _outputLayer[n]._bias._weight << std::endl;
			float sum = _outputLayer[n]._bias._weight;

			for (size_t w = 0; w < _outputLayer[n]._weights.size(); w++) {
				_outputLayer[n]._weights[w]._weight += modulation * _outputLayer[n]._weights[w]._trace;
				sum += _hiddenLayers.back()[w]._output * _outputLayer[n]._weights[w]._weight;
			}

			float output = sigmoid(sum);

			outputs[n] = _outputLayer[n]._output = dist01(generator) < breakRate ? dist01(generator) : output;

			float delta = _outputLayer[n]._output - output;

			_outputLayer[n]._bias._trace = (1.0f - traceDecay) * _outputLayer[n]._bias._trace + delta;
			
			for (size_t w = 0; w < _outputLayer[n]._weights.size(); w++)
				_outputLayer[n]._weights[w]._trace = (1.0f - traceDecay) * _outputLayer[n]._weights[w]._trace + delta * _hiddenLayers.back()[w]._output;

			_outputLayer[n]._outputTrace += (output - _outputLayer[n]._outputTrace) * outputDecay;
		}
	}
	else {
		// Output layer
		for (size_t n = 0; n < _outputLayer.size(); n++) {
			_outputLayer[n]._bias._weight += modulation * _outputLayer[n]._bias._trace;

			float sum = _outputLayer[n]._bias._weight;

			for (size_t w = 0; w < _outputLayer[n]._weights.size(); w++) {
				_outputLayer[n]._weights[w]._weight += modulation * _outputLayer[n]._weights[w]._trace;
				sum += inputs[w] * _outputLayer[n]._weights[w]._weight;
			}

			float output = sigmoid(sum);

			outputs[n] = _outputLayer[n]._output = dist01(generator) < breakRate ? dist01(generator) : output;

			float delta = _outputLayer[n]._output - output;

			_outputLayer[n]._bias._trace = (1.0f - traceDecay) * _outputLayer[n]._bias._trace + delta;
			
			for (size_t w = 0; w < _outputLayer[n]._weights.size(); w++)
				_outputLayer[n]._weights[w]._trace = (1.0f - traceDecay) * _outputLayer[n]._weights[w]._trace + delta * inputs[w];

			_outputLayer[n]._outputTrace += (output - _outputLayer[n]._outputTrace) * outputDecay;
		}
	}
}

void HebbianLearner::process(const std::vector<float> &inputs, std::vector<float> &outputs, float activationMultiplier, float modulation, float traceDecay, float outputDecay, float breakRate, std::mt19937 &generator) {
	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

	if (!_hiddenLayers.empty()) {
		// First hidden layer
		for (size_t n = 0; n < _hiddenLayers[0].size(); n++) {
			_hiddenLayers[0][n]._bias._weight += modulation * _hiddenLayers[0][n]._bias._trace;

			float sum = _hiddenLayers[0][n]._bias._weight;

			for (size_t w = 0; w < _hiddenLayers[0][n]._weights.size(); w++) {
				_hiddenLayers[0][n]._weights[w]._weight += modulation * _hiddenLayers[0][n]._weights[w]._trace;
				sum += inputs[w] * _hiddenLayers[0][n]._weights[w]._weight;
			}

			float output = sigmoid(sum * activationMultiplier);

			_hiddenLayers[0][n]._output = dist01(generator) < breakRate ? dist01(generator) : output;

			float delta = _hiddenLayers[0][n]._output - output;

			_hiddenLayers[0][n]._bias._trace = (1.0f - traceDecay) * _hiddenLayers[0][n]._bias._trace + delta;

			for (size_t w = 0; w < _hiddenLayers[0][n]._weights.size(); w++)
				_hiddenLayers[0][n]._weights[w]._trace = (1.0f - traceDecay) * _hiddenLayers[0][n]._weights[w]._trace + delta * inputs[w];

			_hiddenLayers[0][n]._outputTrace += (output - _hiddenLayers[0][n]._outputTrace) * outputDecay;
		}

		if (_hiddenLayers.size() > 1) {
			// All other hidden layers
			for (size_t l = 1; l < _hiddenLayers.size(); l++) {
				for (size_t n = 0; n < _hiddenLayers[l].size(); n++) {
					_hiddenLayers[l][n]._bias._weight += modulation * _hiddenLayers[l][n]._bias._trace;

					float sum = _hiddenLayers[l][n]._bias._weight;

					for (size_t w = 0; w < _hiddenLayers[l][n]._weights.size(); w++) {
						_hiddenLayers[l][n]._weights[w]._weight += modulation * _hiddenLayers[l][n]._weights[w]._trace;
						sum += _hiddenLayers[l - 1][w]._output * _hiddenLayers[l][n]._weights[w]._weight;
					}

					float output = sigmoid(sum * activationMultiplier);

					_hiddenLayers[l][n]._output = dist01(generator) < breakRate ? dist01(generator) : output;

					float delta = _hiddenLayers[l][n]._output - output;

					_hiddenLayers[l][n]._bias._trace = (1.0f - traceDecay) * _hiddenLayers[l][n]._bias._trace + delta;

					for (size_t w = 0; w < _hiddenLayers[l][n]._weights.size(); w++)
						_hiddenLayers[l][n]._weights[w]._trace = (1.0f - traceDecay) * _hiddenLayers[l][n]._weights[w]._trace + delta * _hiddenLayers[l - 1][w]._output;

					_hiddenLayers[l][n]._outputTrace += (output - _hiddenLayers[l][n]._outputTrace) * outputDecay;
				}
			}
		}

		// Output layer
		for (size_t n = 0; n < _outputLayer.size(); n++) {
			_outputLayer[n]._bias._weight += modulation * _outputLayer[n]._bias._trace;

			float sum = _outputLayer[n]._bias._weight;

			for (size_t w = 0; w < _outputLayer[n]._weights.size(); w++) {
				_outputLayer[n]._weights[w]._weight += modulation * _outputLayer[n]._weights[w]._trace;
				sum += _hiddenLayers.back()[w]._output * _outputLayer[n]._weights[w]._weight;
			}

			float output = sigmoid(sum * activationMultiplier);

			outputs[n] = _outputLayer[n]._output = dist01(generator) < breakRate ? dist01(generator) : output;

			float delta = _outputLayer[n]._output - output;

			_outputLayer[n]._bias._trace = (1.0f - traceDecay) * _outputLayer[n]._bias._trace + delta;

			for (size_t w = 0; w < _outputLayer[n]._weights.size(); w++)
				_outputLayer[n]._weights[w]._trace = (1.0f - traceDecay) * _outputLayer[n]._weights[w]._trace + delta * _hiddenLayers.back()[w]._output;

			_outputLayer[n]._outputTrace += (output - _outputLayer[n]._outputTrace) * outputDecay;
		}
	}
	else {
		// Output layer
		for (size_t n = 0; n < _outputLayer.size(); n++) {
			_outputLayer[n]._bias._weight += modulation * _outputLayer[n]._bias._trace;

			float sum = _outputLayer[n]._bias._weight;

			for (size_t w = 0; w < _outputLayer[n]._weights.size(); w++) {
				_outputLayer[n]._weights[w]._weight += modulation * _outputLayer[n]._weights[w]._trace;
				sum += inputs[w] * _outputLayer[n]._weights[w]._weight;
			}

			float output = sigmoid(sum * activationMultiplier);

			outputs[n] = _outputLayer[n]._output = dist01(generator) < breakRate ? dist01(generator) : output;

			float delta = _outputLayer[n]._output - output;

			_outputLayer[n]._bias._trace = (1.0f - traceDecay) * _outputLayer[n]._bias._trace + delta;

			for (size_t w = 0; w < _outputLayer[n]._weights.size(); w++)
				_outputLayer[n]._weights[w]._trace = (1.0f - traceDecay) * _outputLayer[n]._weights[w]._trace + delta * inputs[w];

			_outputLayer[n]._outputTrace += (output - _outputLayer[n]._outputTrace) * outputDecay;
		}
	}
}