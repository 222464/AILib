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

#include <nn/FeedForwardNeuralNetwork.h>

#include <algorithm>
#include <random>

using namespace nn;

void FeedForwardNeuralNetwork::Gradient::decayTowards(float decay, const Gradient &other) {
	for (size_t g = 0; g < _outputGradient.size(); g++)
		_outputGradient[g] += (other._outputGradient[g] - _outputGradient[g]) * decay;

	for (size_t l = 0; l < _hiddenLayersGradient.size(); l++)
	for (size_t g = 0; g < _hiddenLayersGradient[l].size(); g++)
		_hiddenLayersGradient[l][g] += (other._hiddenLayersGradient[l][g] - _hiddenLayersGradient[l][g]) * decay;
}

void FeedForwardNeuralNetwork::Gradient::setValue(float value) {
	for (size_t g = 0; g < _outputGradient.size(); g++)
		_outputGradient[g] = value;

	for (size_t l = 0; l < _hiddenLayersGradient.size(); l++)
	for (size_t g = 0; g < _hiddenLayersGradient[l].size(); g++)
		_hiddenLayersGradient[l][g] = value;
}

void FeedForwardNeuralNetwork::BrownianPerturbationSet::update(float dt) {
	for (BrownianPerturbation &p : _outputPerturbations)
		p.update(_generator, dt);

	for (size_t l = 0; l < _hiddenLayerPerturbations.size(); l++)
	for (BrownianPerturbation &p : _hiddenLayerPerturbations[l])
		p.update(_generator, dt);
}

void FeedForwardNeuralNetwork::BrownianPerturbationSet::setAll(const BrownianPerturbation &perturbation) {
	for (BrownianPerturbation &p : _outputPerturbations)
		p = perturbation;

	for (size_t l = 0; l < _hiddenLayerPerturbations.size(); l++)
	for (BrownianPerturbation &p : _hiddenLayerPerturbations[l])
		p = perturbation;
}

void FeedForwardNeuralNetwork::BrownianPerturbationSet::reset() {
	for (BrownianPerturbation &p : _outputPerturbations)
		p.reset();

	for (size_t l = 0; l < _hiddenLayerPerturbations.size(); l++)
	for (BrownianPerturbation &p : _hiddenLayerPerturbations[l])
		p.reset();
}

const FeedForwardNeuralNetwork &FeedForwardNeuralNetwork::operator=(const FeedForwardNeuralNetwork &other) {
	_activationMultiplier = other._activationMultiplier;
	_outputTraceDecay = other._outputTraceDecay;
	_weightTraceDecay = other._weightTraceDecay;

	_inputs.resize(other.getNumInputs());
	_outputs.resize(other.getNumOutputs());

	if (other.getNumHiddenLayers() == 0) {
		// Connect outputs directly to inputs
		for (size_t n = 0; n < _outputs.size(); n++) {
			_outputs[n]._output = other._outputs[n]._output;
			_outputs[n]._outputTrace = other._outputs[n]._outputTrace;

			_outputs[n]._bias = other._outputs[n]._bias;

			_outputs[n]._synapses.resize(_inputs.size());

			for (size_t i = 0; i < _inputs.size(); i++) {
				_outputs[n]._synapses[i]._pInput = &_inputs[i];

				_outputs[n]._synapses[i]._trace = other._outputs[n]._synapses[i]._trace;

				_outputs[n]._synapses[i]._weight = other._outputs[n]._synapses[i]._weight;
			}
		}
	} else {
		_hidden.resize(other.getNumHiddenLayers());

		// First hidden layer
		_hidden[0].resize(other.getNumNeuronsPerHiddenLayer());

		for (size_t n = 0; n < _hidden[0].size(); n++) {
			_hidden[0][n]._output = other._hidden[0][n]._output;
			_hidden[0][n]._outputTrace = other._hidden[0][n]._outputTrace;

			_hidden[0][n]._bias = other._hidden[0][n]._bias;

			_hidden[0][n]._synapses.resize(_inputs.size());

			for (size_t i = 0; i < _inputs.size(); i++) {
				_hidden[0][n]._synapses[i]._pInput = &_inputs[i];

				_hidden[0][n]._synapses[i]._trace = other._hidden[0][n]._synapses[i]._trace;

				_hidden[0][n]._synapses[i]._weight = other._hidden[0][n]._synapses[i]._weight;
			}
		}

		// All other hidden layers
		for (size_t l = 1; l < other.getNumHiddenLayers(); l++) {
			size_t prevLayerIndex = l - 1;

			_hidden[l].resize(other.getNumNeuronsPerHiddenLayer());

			for (size_t n = 0; n < _hidden[l].size(); n++) {
				_hidden[l][n]._output = other._hidden[l][n]._output;
				_hidden[l][n]._outputTrace = other._hidden[l][n]._outputTrace;

				_hidden[l][n]._bias = other._hidden[l][n]._bias;

				_hidden[l][n]._synapses.resize(other.getNumNeuronsPerHiddenLayer());

				for (size_t i = 0; i < other.getNumNeuronsPerHiddenLayer(); i++) {
					_hidden[l][n]._synapses[i]._pInput = &_hidden[prevLayerIndex][i];

					_hidden[l][n]._synapses[i]._trace = other._hidden[l][n]._synapses[i]._trace;

					_hidden[l][n]._synapses[i]._weight = other._hidden[l][n]._synapses[i]._weight;
				}
			}
		}

		for (size_t n = 0; n < _outputs.size(); n++) {
			_outputs[n]._output = other._outputs[n]._output;
			_outputs[n]._outputTrace = other._outputs[n]._outputTrace;

			_outputs[n]._bias = other._outputs[n]._bias;

			_outputs[n]._synapses.resize(other.getNumNeuronsPerHiddenLayer());

			for (size_t i = 0; i < other.getNumNeuronsPerHiddenLayer(); i++) {
				_outputs[n]._synapses[i]._pInput = &_hidden.back()[i];

				_outputs[n]._synapses[i]._trace = other._outputs[n]._synapses[i]._trace;

				_outputs[n]._synapses[i]._weight = other._outputs[n]._synapses[i]._weight;
			}
		}
	}

	return *this;
}

void FeedForwardNeuralNetwork::createRandom(size_t numInputs, size_t numOutputs,
	size_t numHiddenLayers, size_t numNeuronsPerHiddenLayer,
	float minWeight, float maxWeight, std::mt19937 &generator)
{
	_inputs.resize(numInputs);
	_outputs.resize(numOutputs);

	std::uniform_real_distribution<float> distribution(minWeight, maxWeight);

	if (numHiddenLayers == 0) {
		// Connect outputs directly to inputs
		for (Neuron &neuron : _outputs) {
			neuron._bias = distribution(generator);

			neuron._synapses.resize(numInputs);

			for (size_t i = 0; i < numInputs; i++) {
				neuron._synapses[i]._pInput = &_inputs[i];
				neuron._synapses[i]._weight = distribution(generator);
			}
		}
	} else {
		_hidden.resize(numHiddenLayers);

		// First hidden layer
		_hidden[0].resize(numNeuronsPerHiddenLayer);

		for (Neuron &neuron : _hidden[0]) {
			neuron._bias = distribution(generator);

			neuron._synapses.resize(numInputs);

			for (size_t i = 0; i < numInputs; i++) {
				neuron._synapses[i]._pInput = &_inputs[i];
				neuron._synapses[i]._weight = distribution(generator);
			}
		}

		// All other hidden layers
		for (size_t l = 1; l < numHiddenLayers; l++) {
			size_t prevLayerIndex = l - 1;

			_hidden[l].resize(numNeuronsPerHiddenLayer);

			for (Neuron &neuron : _hidden[l]) {
				neuron._bias = distribution(generator);

				neuron._synapses.resize(numNeuronsPerHiddenLayer);

				for (size_t i = 0; i < numNeuronsPerHiddenLayer; i++) {
					neuron._synapses[i]._pInput = &_hidden[prevLayerIndex][i];
					neuron._synapses[i]._weight = distribution(generator);
				}
			}
		}

		for (Neuron &neuron : _outputs) {
			neuron._bias = distribution(generator);

			neuron._synapses.resize(numNeuronsPerHiddenLayer);

			for (size_t i = 0; i < numNeuronsPerHiddenLayer; i++) {
				neuron._synapses[i]._pInput = &_hidden.back()[i];
				neuron._synapses[i]._weight = distribution(generator);
			}
		}
	}
}

void FeedForwardNeuralNetwork::createFromParents(const FeedForwardNeuralNetwork &parent1, const FeedForwardNeuralNetwork &parent2,
	float averageWeightsChance, unsigned long seed)
{
	std::mt19937 generator;
	generator.seed(seed);
	std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

	_activationMultiplier = parent1._activationMultiplier;
	_outputTraceDecay = parent1._outputTraceDecay;
	_weightTraceDecay = parent1._weightTraceDecay;

	_inputs.resize(parent1.getNumInputs());
	_outputs.resize(parent1.getNumOutputs());

	if (parent1.getNumHiddenLayers() == 0) {
		// Connect outputs directly to inputs
		for (size_t n = 0; n < _outputs.size(); n++) {
			_outputs[n]._bias = distribution(generator) < averageWeightsChance ?
				(parent1._outputs[n]._bias + parent2._outputs[n]._bias) * 0.5f :
				(distribution(generator) < 0.5f ? parent1._outputs[n]._bias : parent2._outputs[n]._bias);

			_outputs[n]._synapses.resize(_inputs.size());

			for (size_t i = 0; i < _inputs.size(); i++) {
				_outputs[n]._synapses[i]._pInput = &_inputs[i];

				_outputs[n]._synapses[i]._weight = distribution(generator) < averageWeightsChance ?
					(parent1._outputs[n]._synapses[i]._weight + parent2._outputs[n]._synapses[i]._weight) * 0.5f :
					(distribution(generator) < 0.5f ? parent1._outputs[n]._synapses[i]._weight : parent2._outputs[n]._synapses[i]._weight);
			}
		}
	} else {
		_hidden.resize(parent1.getNumHiddenLayers());

		// First hidden layer
		_hidden[0].resize(parent1.getNumNeuronsPerHiddenLayer());

		for (size_t n = 0; n < _hidden[0].size(); n++) {
			_hidden[0][n]._bias = distribution(generator) < averageWeightsChance ?
				(parent1._hidden[0][n]._bias + parent2._hidden[0][n]._bias) * 0.5f :
				(distribution(generator) < 0.5f ? parent1._hidden[0][n]._bias : parent2._hidden[0][n]._bias);

			_hidden[0][n]._synapses.resize(_inputs.size());

			for (size_t i = 0; i < _inputs.size(); i++) {
				_hidden[0][n]._synapses[i]._pInput = &_inputs[i];

				_hidden[0][n]._synapses[i]._weight = distribution(generator) < averageWeightsChance ?
					(parent1._hidden[0][n]._synapses[i]._weight + parent2._hidden[0][n]._synapses[i]._weight) * 0.5f :
					(distribution(generator) < 0.5f ? parent1._hidden[0][n]._synapses[i]._weight : parent2._hidden[0][n]._synapses[i]._weight);
			}
		}

		// All other hidden layers
		for (size_t i = 1; i < parent1.getNumHiddenLayers(); i++) {
			size_t prevLayerIndex = i - 1;

			_hidden[i].resize(parent1.getNumNeuronsPerHiddenLayer());

			for (size_t n = 0; n < _hidden[i].size(); n++) {
				_hidden[i][n]._bias = distribution(generator) < averageWeightsChance ?
					(parent1._hidden[i][n]._bias + parent2._hidden[i][n]._bias) * 0.5f :
					(distribution(generator) < 0.5f ? parent1._hidden[i][n]._bias : parent2._hidden[i][n]._bias);


				_hidden[i][n]._synapses.resize(parent1.getNumNeuronsPerHiddenLayer());

				for (size_t i = 0; i < parent1.getNumNeuronsPerHiddenLayer(); i++) {
					_hidden[i][n]._synapses[i]._pInput = &_hidden[prevLayerIndex][i];

					_hidden[i][n]._synapses[i]._weight = distribution(generator) < averageWeightsChance ?
						(parent1._hidden[i][n]._synapses[i]._weight + parent2._hidden[i][n]._synapses[i]._weight) * 0.5f :
						(distribution(generator) < 0.5f ? parent1._hidden[i][n]._synapses[i]._weight : parent2._hidden[i][n]._synapses[i]._weight);
				}
			}
		}

		for (size_t n = 0; n < _outputs.size(); n++) {
			_outputs[n]._bias = distribution(generator) < averageWeightsChance ?
				(parent1._outputs[n]._bias + parent2._outputs[n]._bias) * 0.5f :
				(distribution(generator) < 0.5f ? parent1._outputs[n]._bias : parent2._outputs[n]._bias);

			_outputs[n]._synapses.resize(parent1.getNumNeuronsPerHiddenLayer());

			for (size_t i = 0; i < parent1.getNumNeuronsPerHiddenLayer(); i++) {
				_outputs[n]._synapses[i]._pInput = &_hidden.back()[i];

				_outputs[n]._synapses[i]._weight = distribution(generator) < averageWeightsChance ?
					(parent1._outputs[n]._synapses[i]._weight + parent2._outputs[n]._synapses[i]._weight) * 0.5f :
					(distribution(generator) < 0.5f ? parent1._outputs[n]._synapses[i]._weight : parent2._outputs[n]._synapses[i]._weight);
			}
		}
	}
}

void FeedForwardNeuralNetwork::mutate(float weightMutationChance, float maxWeightPerturbation, unsigned long seed) {
	std::mt19937 generator;
	generator.seed(seed);
	std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
	std::uniform_real_distribution<float> distributionPerturb(-maxWeightPerturbation, maxWeightPerturbation);

	for (size_t l = 0; l < getNumHiddenLayers(); l++)
	for (Neuron &n : _hidden[l]) {
		n._bias += distribution(generator) < weightMutationChance ? distributionPerturb(generator) : 0.0f;

		for (Neuron::Synapse &s : n._synapses)
			s._weight += distribution(generator) < weightMutationChance ? distributionPerturb(generator) : 0.0f;
	}

	for (Neuron &n : _outputs) {
		n._bias += distribution(generator) < weightMutationChance ? distributionPerturb(generator) : 0.0f;

		for (Neuron::Synapse &s : n._synapses)
			s._weight += distribution(generator) < weightMutationChance ? distributionPerturb(generator) : 0.0f;
	}
}

void FeedForwardNeuralNetwork::activate() {
	for (Sensor &s : _inputs)
		s.activate(_activationMultiplier, _outputTraceDecay);

	for (size_t l = 0; l < getNumHiddenLayers(); l++)
	for (Neuron &n : _hidden[l])
		n.activate(_activationMultiplier, _outputTraceDecay);

	for (Neuron &n : _outputs)
		n.activate(_activationMultiplier, _outputTraceDecay);
}

void FeedForwardNeuralNetwork::activateTraceless() {
	for (size_t l = 0; l < getNumHiddenLayers(); l++)
	for (Neuron &n : _hidden[l])
		n.activateTraceless(_activationMultiplier);

	for (Neuron &n : _outputs)
		n.activateTraceless(_activationMultiplier);
}

void FeedForwardNeuralNetwork::activateAndReinforce(float error) {
	for (Sensor &s : _inputs)
		s.activate(_activationMultiplier, _outputTraceDecay);

	for (size_t l = 0; l < getNumHiddenLayers(); l++)
	for (Neuron &n : _hidden[l])
		n.activateAndReinforce(_activationMultiplier, _outputTraceDecay, _weightTraceDecay, error);

	for (Neuron &n : _outputs)
		n.activateAndReinforce(_activationMultiplier, _outputTraceDecay, _weightTraceDecay, error);
}

void FeedForwardNeuralNetwork::activateAndReinforceTraceless(float error) {
	for (size_t l = 0; l < getNumHiddenLayers(); l++)
	for (Neuron &n : _hidden[l])
		n.activateAndReinforceTraceless(_activationMultiplier, error);

	for (Neuron &n : _outputs)
		n.activateAndReinforceTraceless(_activationMultiplier, error);
}

void FeedForwardNeuralNetwork::activateLinearOutputLayer() {
	for (Sensor &s : _inputs)
		s.activate(_activationMultiplier, _outputTraceDecay);

	for (size_t l = 0; l < getNumHiddenLayers(); l++)
	for (Neuron &n : _hidden[l])
		n.activate(_activationMultiplier, _outputTraceDecay);

	for (Neuron &n : _outputs)
		n.activateLinear(_activationMultiplier);
}

void FeedForwardNeuralNetwork::activateArp(std::mt19937 &generator) {
	for (Sensor &s : _inputs)
		s.activateArp(_activationMultiplier, _outputTraceDecay);

	for (size_t l = 0; l < getNumHiddenLayers(); l++)
	for (size_t n = 0; n < getNumNeuronsPerHiddenLayer(); n++)
		_hidden[l][n].activateArp(_activationMultiplier, _outputTraceDecay, generator);

	for (size_t n = 0; n < getNumOutputs(); n++)
		_outputs[n].activateArp(_activationMultiplier, _outputTraceDecay, generator);
}

void FeedForwardNeuralNetwork::reinforce(float error) {
	for (size_t l = 0; l < getNumHiddenLayers(); l++)
	for (Neuron &n : _hidden[l])
		n.reinforce(error, _weightTraceDecay);

	for (Neuron &n : _outputs)
		n.reinforce(error, _weightTraceDecay);
}

void FeedForwardNeuralNetwork::reinforceTraceless(float error) {
	for (size_t l = 0; l < getNumHiddenLayers(); l++)
	for (Neuron &n : _hidden[l])
		n.reinforceTraceless(error);

	for (Neuron &n : _outputs)
		n.reinforceTraceless(error);
}

void FeedForwardNeuralNetwork::reinforceArp(float reward, float alpha, float lambda) {
	for (size_t l = 0; l < getNumHiddenLayers(); l++)
	for (Neuron &n : _hidden[l])
		n.reinforceArp(reward, alpha, lambda);

	for (Neuron &n : _outputs)
		n.reinforceArp(reward, alpha, lambda);
}

void FeedForwardNeuralNetwork::reinforceArpWithTraces(float reward, float alpha, float lambda) {
	for (size_t l = 0; l < getNumHiddenLayers(); l++)
	for (Neuron &n : _hidden[l])
		n.reinforceArpWithTraces(reward, alpha, lambda, _weightTraceDecay);

	for (Neuron &n : _outputs)
		n.reinforceArpWithTraces(reward, alpha, lambda, _weightTraceDecay);
}

void FeedForwardNeuralNetwork::reinforceArpMomentum(float reward, float alpha, float lambda, float momentum) {
	for (size_t l = 0; l < getNumHiddenLayers(); l++)
	for (Neuron &n : _hidden[l])
		n.reinforceArpMomentum(reward, alpha, lambda, momentum);

	for (Neuron &n : _outputs)
		n.reinforceArpMomentum(reward, alpha, lambda, momentum);
}

void FeedForwardNeuralNetwork::zeroTraces() {
	for (Sensor &s : _inputs)
		s._outputTrace = 0.0f;

	for (size_t l = 0; l < getNumHiddenLayers(); l++)
	for (Neuron &n : _hidden[l]) {
		n._outputTrace = 0.0f;

		for (Neuron::Synapse &s : n._synapses)
			s._trace = 0.0f;
	}

	for (Neuron &n : _outputs) {
		n._outputTrace = 0.0f;

		for (Neuron::Synapse &s : n._synapses)
			s._trace = 0.0f;
	}
}

void FeedForwardNeuralNetwork::getGradient(const std::vector<float> &targets, Gradient &grad) {
	std::vector<float> error(targets.size());

	for (size_t n = 0; n < getNumOutputs(); n++)
		error[n] = targets[n] - _outputs[n]._output;

	getGradientFromError(error, grad);
}

void FeedForwardNeuralNetwork::getGradientLinearOutputLayer(const std::vector<float> &targets, Gradient &grad) {

	std::vector<float> error(targets.size());

	for (size_t n = 0; n < getNumOutputs(); n++)
		error[n] = targets[n] - _outputs[n]._output;

	getGradientFromError(error, grad);
}

void FeedForwardNeuralNetwork::getGradientFromError(const std::vector<float> &error, Gradient &grad) {
	grad._outputGradient.resize(_outputs.size());

	for (size_t n = 0; n < getNumOutputs(); n++)
		grad._outputGradient[n] = error[n];

	if (!_hidden.empty()) {
		grad._hiddenLayersGradient.resize(_hidden.size());

		for (size_t l = 0; l < getNumHiddenLayers(); l++)
			grad._hiddenLayersGradient[l].resize(getNumNeuronsPerHiddenLayer());

		size_t lastHiddenLayerIndex = _hidden.size() - 1;

		// Last hidden layer
		for (size_t n = 0; n < getNumNeuronsPerHiddenLayer(); n++) {
			float sum = 0.0f;

			for (size_t c = 0; c < getNumOutputs(); c++)
				sum += grad._outputGradient[c] * _outputs[c]._synapses[n]._weight;

			grad._hiddenLayersGradient[lastHiddenLayerIndex][n] = sum * _hidden[lastHiddenLayerIndex][n]._output * (1.0f - _hidden[lastHiddenLayerIndex][n]._output);
		}

		// All other hidden layers
		for (int l = static_cast<int>(getNumHiddenLayers()) - 2; l >= 0; l--) {
			int nextLayerIndex = l + 1;

			for (size_t n = 0; n < getNumNeuronsPerHiddenLayer(); n++) {
				float sum = 0.0f;

				for (size_t c = 0; c < getNumNeuronsPerHiddenLayer(); c++)
					sum += grad._hiddenLayersGradient[nextLayerIndex][c] * _hidden[nextLayerIndex][c]._synapses[n]._weight;

				grad._hiddenLayersGradient[l][n] = sum * _hidden[l][n]._output * (1.0f - _hidden[l][n]._output);
			}
		}
	}
}

void FeedForwardNeuralNetwork::getEmptyGradient(Gradient &grad) {
	grad._outputGradient.resize(_outputs.size());

	if (!_hidden.empty()) {
		grad._hiddenLayersGradient.resize(_hidden.size());

		for (size_t l = 0; l < getNumHiddenLayers(); l++)
			grad._hiddenLayersGradient[l].resize(getNumNeuronsPerHiddenLayer());
	}
}

void FeedForwardNeuralNetwork::moveAlongGradient(const Gradient &grad, float alpha) {
	// Update weights
	for (size_t l = 0; l < getNumHiddenLayers(); l++)
	for (size_t n = 0; n < getNumNeuronsPerHiddenLayer(); n++) {
		// Update bias
		_hidden[l][n]._bias += alpha * grad._hiddenLayersGradient[l][n];

		for (size_t c = 0; c < _hidden[l][n]._synapses.size(); c++)
			_hidden[l][n]._synapses[c]._weight += alpha * grad._hiddenLayersGradient[l][n] * _hidden[l][n]._synapses[c]._pInput->_output;
	}

	// Update weights
	for (size_t n = 0; n < getNumOutputs(); n++) {
		// Update bias
		_outputs[n]._bias += alpha * grad._outputGradient[n];

		for (size_t c = 0; c < _outputs[n]._synapses.size(); c++)
			_outputs[n]._synapses[c]._weight += alpha * grad._outputGradient[n] * _outputs[n]._synapses[c]._pInput->_output;
	}
}

void FeedForwardNeuralNetwork::moveAlongGradientSign(const Gradient &grad, float alpha) {
	// Update weights
	for (size_t l = 0; l < getNumHiddenLayers(); l++)
	for (size_t n = 0; n < getNumNeuronsPerHiddenLayer(); n++) {
		// Update bias
		_hidden[l][n]._bias += alpha * grad._hiddenLayersGradient[l][n];

		for (size_t c = 0; c < _hidden[l][n]._synapses.size(); c++)
			_hidden[l][n]._synapses[c]._weight += grad._hiddenLayersGradient[l][n] * _hidden[l][n]._synapses[c]._pInput->_output > 0.0f ? alpha : -alpha;
	}

	// Update weights
	for (size_t n = 0; n < getNumOutputs(); n++) {
		// Update bias
		_outputs[n]._bias += alpha * grad._outputGradient[n];

		for (size_t c = 0; c < _outputs[n]._synapses.size(); c++)
			_outputs[n]._synapses[c]._weight += grad._outputGradient[n] * _outputs[n]._synapses[c]._pInput->_output > 0.0f ? alpha : -alpha;
	}
}

void FeedForwardNeuralNetwork::moveAlongGradientMomentum(const Gradient &grad, float alpha, float momentum) {
	// Update weights
	for (size_t l = 0; l < getNumHiddenLayers(); l++)
	for (size_t n = 0; n < getNumNeuronsPerHiddenLayer(); n++) {
		// Update bias
		float dBias = alpha * grad._hiddenLayersGradient[l][n] + momentum * _hidden[l][n]._biasTrace;

		_hidden[l][n]._bias += dBias;
		_hidden[l][n]._biasTrace = dBias;

		for (size_t c = 0; c < _hidden[l][n]._synapses.size(); c++) {
			float dWeight = alpha * grad._hiddenLayersGradient[l][n] * _hidden[l][n]._synapses[c]._pInput->_output + momentum * _hidden[l][n]._synapses[c]._trace;

			_hidden[l][n]._synapses[c]._weight += dWeight;
			_hidden[l][n]._synapses[c]._trace = dWeight;
		}
	}

	// Update weights
	for (size_t n = 0; n < getNumOutputs(); n++) {
		// Update bias
		float dBias = alpha * grad._outputGradient[n] + momentum * _outputs[n]._biasTrace;

		_outputs[n]._bias += dBias;
		_outputs[n]._biasTrace = dBias;

		for (size_t c = 0; c < _outputs[n]._synapses.size(); c++) {
			float dWeight = alpha * grad._outputGradient[n] * _outputs[n]._synapses[c]._pInput->_output + momentum * _outputs[n]._synapses[c]._trace;

			_outputs[n]._synapses[c]._weight += dWeight;
			_outputs[n]._synapses[c]._trace = dWeight;
		}
	}
}

void FeedForwardNeuralNetwork::storeCurrentOutputsInTraces() {
	for (size_t l = 0; l < getNumHiddenLayers(); l++)
	for (size_t n = 0; n < getNumNeuronsPerHiddenLayer(); n++)
		_hidden[l][n]._outputTrace = _hidden[l][n]._output;

	for (size_t n = 0; n < getNumOutputs(); n++)
		_outputs[n]._outputTrace = _outputs[n]._output;
}

void FeedForwardNeuralNetwork::updateValueFunction(float tdError, float alpha, float traceDecay) {
	Gradient grad;
	getGradientFromError(std::vector<float>(1, 1.0f), grad);

	// Update weights
	for (size_t n = 0; n < getNumOutputs(); n++) {
		// Update bias
		_outputs[n]._biasTrace += -traceDecay * _outputs[n]._biasTrace + grad._outputGradient[n];
		_outputs[n]._bias += alpha * tdError * _outputs[n]._biasTrace;

		for (size_t c = 0; c < _outputs[n]._synapses.size(); c++) {
			_outputs[n]._synapses[c]._trace += -traceDecay * _outputs[n]._synapses[c]._trace + grad._outputGradient[n] * _outputs[n]._synapses[c]._pInput->_output;
			_outputs[n]._synapses[c]._weight += alpha * tdError * _outputs[n]._synapses[c]._trace;
		}
	}

	size_t lastHiddenLayerIndex = _hidden.size() - 1;

	// Last hidden layer
	for (size_t n = 0; n < getNumNeuronsPerHiddenLayer(); n++) {
		// Update bias
		_hidden[lastHiddenLayerIndex][n]._biasTrace += -traceDecay * _hidden[lastHiddenLayerIndex][n]._biasTrace + grad._hiddenLayersGradient[lastHiddenLayerIndex][n];
		_hidden[lastHiddenLayerIndex][n]._bias += alpha * tdError * _hidden[lastHiddenLayerIndex][n]._biasTrace;

		for (size_t c = 0; c < _hidden[lastHiddenLayerIndex][n]._synapses.size(); c++) {
			_hidden[lastHiddenLayerIndex][n]._synapses[c]._trace += -traceDecay * _hidden[lastHiddenLayerIndex][n]._synapses[c]._trace + grad._hiddenLayersGradient[lastHiddenLayerIndex][n] * _hidden[lastHiddenLayerIndex][n]._synapses[c]._pInput->_output;
			_hidden[lastHiddenLayerIndex][n]._synapses[c]._weight += alpha * tdError * _hidden[lastHiddenLayerIndex][n]._synapses[c]._trace;
		}
	}

	// All other hidden layers
	for (int l = static_cast<int>(getNumHiddenLayers()) - 2; l >= 0; l--) {
		int nextLayerIndex = l + 1;

		for (size_t n = 0; n < getNumNeuronsPerHiddenLayer(); n++) {
			// Update bias
			_hidden[l][n]._biasTrace += -traceDecay * _hidden[l][n]._biasTrace + grad._hiddenLayersGradient[l][n];
			_hidden[l][n]._bias += alpha * tdError * _hidden[l][n]._biasTrace;

			for (size_t c = 0; c < _hidden[lastHiddenLayerIndex][n]._synapses.size(); c++) {
				_hidden[l][n]._synapses[c]._trace += -traceDecay * _hidden[l][n]._synapses[c]._trace + grad._hiddenLayersGradient[l][n] * _hidden[l][n]._synapses[c]._pInput->_output;
				_hidden[l][n]._synapses[c]._weight += alpha * tdError * _hidden[l][n]._synapses[c]._trace;
			}
		}
	}
}

void FeedForwardNeuralNetwork::decayWeights(float decay) {
	for (size_t l = 0; l < getNumHiddenLayers(); l++)
	for (size_t n = 0; n < getNumNeuronsPerHiddenLayer(); n++) {
		_hidden[l][n]._bias += -decay * _hidden[l][n]._bias;

		for (size_t c = 0; c < _hidden[l][n]._synapses.size(); c++)
			_hidden[l][n]._synapses[c]._weight += -decay * _hidden[l][n]._synapses[c]._weight;
	}

	for (size_t n = 0; n < getNumOutputs(); n++) {
		_outputs[n]._bias += -decay * _outputs[n]._bias;

		for (size_t c = 0; c < _outputs[n]._synapses.size(); c++)
			_outputs[n]._synapses[c]._weight += -decay * _outputs[n]._synapses[c]._weight;
	}
}

void FeedForwardNeuralNetwork::decayWeightsExcludingOutputLayer(float decay) {
	for (size_t l = 0; l < getNumHiddenLayers(); l++)
	for (size_t n = 0; n < getNumNeuronsPerHiddenLayer(); n++) {
		_hidden[l][n]._bias += -decay * _hidden[l][n]._bias;

		for (size_t c = 0; c < _hidden[l][n]._synapses.size(); c++)
			_hidden[l][n]._synapses[c]._weight += -decay * _hidden[l][n]._synapses[c]._weight;
	}
}

void FeedForwardNeuralNetwork::getInputGradient(const Gradient &existingGrad, std::vector<float> &inputGrad) {
	inputGrad.resize(getNumInputs());

	if (_hidden.empty()) {
		for (size_t i = 0; i < inputGrad.size(); i++) {
			float sum = 0.0f;

			for (size_t c = 0; c < getNumOutputs(); c++)
				sum += existingGrad._outputGradient[c] * _outputs[c]._synapses[i]._weight;

			inputGrad[i] = sum;
		}
	} else {
		for (size_t i = 0; i < inputGrad.size(); i++) {
			float sum = 0.0f;

			for (size_t c = 0; c < getNumNeuronsPerHiddenLayer(); c++)
				sum += existingGrad._hiddenLayersGradient[0][c] * _hidden[0][c]._synapses[i]._weight;

			inputGrad[i] = sum;
		}
	}
}

void FeedForwardNeuralNetwork::getBrownianPerturbationSet(BrownianPerturbationSet &set) {
	set._outputPerturbations.resize(_outputs.size());

	if (!_hidden.empty()) {
		set._hiddenLayerPerturbations.resize(_hidden.size());

		for (size_t l = 0; l < getNumHiddenLayers(); l++)
			set._hiddenLayerPerturbations[l].resize(getNumNeuronsPerHiddenLayer());
	}
}

void FeedForwardNeuralNetwork::writeToStream(std::ostream &stream) {
	stream << getNumInputs() << " " << getNumOutputs() << " " << getNumHiddenLayers() << " " << getNumNeuronsPerHiddenLayer() << std::endl;
	stream << _activationMultiplier << " " << _outputTraceDecay << " " << _weightTraceDecay << std::endl;

	if (!_hidden.empty()) {
		for (size_t l = 0, numHiddenLayers = _hidden.size(); l < numHiddenLayers; l++)
		for (size_t n = 0; n < getNumNeuronsPerHiddenLayer(); n++) {
			stream << _hidden[l][n]._bias << " ";

			for (size_t c = 0; c < _hidden[l][n]._synapses.size(); c++)
				stream << _hidden[l][n]._synapses[c]._weight << " ";

			stream << std::endl;
		}
	}

	for (size_t n = 0; n < getNumOutputs(); n++) {
		stream << _outputs[n]._bias << " ";

		for (size_t c = 0; c < _outputs[n]._synapses.size(); c++)
			stream << _outputs[n]._synapses[c]._weight << " ";

		stream << std::endl;
	}
}

void FeedForwardNeuralNetwork::readFromStream(std::istream &stream) {
	size_t numInputs, numOutputs, numHiddenLayers, numNeuronsPerHiddenLayer;

	stream >> numInputs >> numOutputs >> numHiddenLayers >> numNeuronsPerHiddenLayer;

	stream >> _activationMultiplier >> _outputTraceDecay >> _weightTraceDecay;

	_inputs.resize(numInputs);
	_outputs.resize(numOutputs);

	if (numHiddenLayers == 0) {
		// Connect outputs directly to inputs
		for (size_t n = 0; n < numOutputs; n++) {
			stream >> _outputs[n]._bias;

			_outputs[n]._synapses.resize(numInputs);

			for (size_t i = 0; i < numInputs; i++) {
				_outputs[n]._synapses[i]._pInput = &_inputs[i];
				stream >> _outputs[n]._synapses[i]._weight;
			}
		}
	} else {
		_hidden.resize(numHiddenLayers);

		// First hidden layer
		_hidden[0].resize(numNeuronsPerHiddenLayer);

		for (size_t n = 0; n < numNeuronsPerHiddenLayer; n++) {
			stream >> _hidden[0][n]._bias;

			_hidden[0][n]._synapses.resize(numInputs);

			for (size_t i = 0; i < numInputs; i++) {
				_hidden[0][n]._synapses[i]._pInput = &_inputs[i];
				stream >> _hidden[0][n]._synapses[i]._weight;
			}
		}

		// All other hidden layers
		for (size_t l = 1; l < numHiddenLayers; l++) {
			size_t prevLayerIndex = l - 1;

			_hidden[l].resize(numNeuronsPerHiddenLayer);

			for (size_t n = 0; n < numNeuronsPerHiddenLayer; n++) {
				stream >> _hidden[l][n]._bias;

				_hidden[l][n]._synapses.resize(numNeuronsPerHiddenLayer);

				for (size_t i = 0; i < numNeuronsPerHiddenLayer; i++) {
					_hidden[l][n]._synapses[i]._pInput = &_hidden[prevLayerIndex][i];
					stream >> _hidden[l][n]._synapses[i]._weight;
				}
			}
		}

		for (size_t n = 0; n < numOutputs; n++) {
			stream >> _outputs[n]._bias;

			_outputs[n]._synapses.resize(numNeuronsPerHiddenLayer);

			for (size_t i = 0; i < numNeuronsPerHiddenLayer; i++) {
				_outputs[n]._synapses[i]._pInput = &_hidden.back()[i];
				stream >> _outputs[n]._synapses[i]._weight;
			}
		}
	}
}

void FeedForwardNeuralNetwork::getWeightVector(std::vector<float> &weights) {
	weights.clear();

	if (!_hidden.empty()) {
		for (size_t l = 0, numHiddenLayers = _hidden.size(); l < numHiddenLayers; l++)
		for (size_t n = 0; n < getNumNeuronsPerHiddenLayer(); n++) {
			weights.push_back(_hidden[l][n]._bias);

			for (size_t c = 0; c < _hidden[l][n]._synapses.size(); c++)
				weights.push_back(_hidden[l][n]._synapses[c]._weight);
		}
	}

	for (size_t n = 0; n < getNumOutputs(); n++) {
		weights.push_back(_outputs[n]._bias);

		for (size_t c = 0; c < _outputs[n]._synapses.size(); c++)
			weights.push_back(_outputs[n]._synapses[c]._weight);
	}
}

void FeedForwardNeuralNetwork::setWeightVector(const std::vector<float> &weights) {
	size_t index = 0;

	if (!_hidden.empty()) {
		for (size_t l = 0, numHiddenLayers = _hidden.size(); l < numHiddenLayers; l++)
		for (size_t n = 0; n < getNumNeuronsPerHiddenLayer(); n++) {
			_hidden[l][n]._bias = weights[index++];

			for (size_t c = 0; c < _hidden[l][n]._synapses.size(); c++)
				_hidden[l][n]._synapses[c]._weight = weights[index++];
		}
	}

	for (size_t n = 0; n < getNumOutputs(); n++) {
		_outputs[n]._bias = weights[index++];

		for (size_t c = 0; c < _outputs[n]._synapses.size(); c++)
			_outputs[n]._synapses[c]._weight = weights[index++];
	}
}

size_t FeedForwardNeuralNetwork::getWeightVectorSize() const {
	size_t size = 0;

	if (!_hidden.empty()) {
		for (size_t l = 0, numHiddenLayers = _hidden.size(); l < numHiddenLayers; l++)
		for (size_t n = 0; n < getNumNeuronsPerHiddenLayer(); n++) {
			size += 1 + _hidden[l][n]._synapses.size();
		}
	}

	for (size_t n = 0; n < getNumOutputs(); n++) {
		size += 1 + _outputs[n]._synapses.size();
	}

	return size;
}