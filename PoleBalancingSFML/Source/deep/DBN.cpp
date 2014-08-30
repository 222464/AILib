#include <deep/DBN.h>

#include <assert.h>

using namespace deep;

void DBN::createRandom(size_t numInputs, size_t numOutputs, const std::vector<size_t> &rbmNumHiddens, double minWeight, double maxWeight, std::mt19937 &generator) {
	_rbmLayers.resize(rbmNumHiddens.size());
	_rbmErrors.resize(rbmNumHiddens.size());

	size_t prevNumOutputs = numInputs;

	for (size_t i = 0; i < _rbmLayers.size(); i++) {
		_rbmLayers[i].createRandom(prevNumOutputs, rbmNumHiddens[i], minWeight, maxWeight, generator);
		_rbmErrors[i].clear();
		_rbmErrors[i].assign(rbmNumHiddens[i], 0.0);

		prevNumOutputs = rbmNumHiddens[i];
	}

	_outputNodes.resize(numOutputs);

	std::uniform_real_distribution<double> weightDist(minWeight, maxWeight);

	for (size_t i = 0; i < _outputNodes.size(); i++) {
		_outputNodes[i]._weights.resize(prevNumOutputs);
		_outputNodes[i]._prevDWeights.clear();
		_outputNodes[i]._prevDWeights.assign(prevNumOutputs, 0.0f);
		_outputNodes[i]._accumulatedDWeight.clear();
		_outputNodes[i]._accumulatedDWeight.assign(prevNumOutputs, 0.0f);

		_outputNodes[i]._bias = weightDist(generator);

		for (size_t j = 0; j < _outputNodes[i]._weights.size(); j++)
			_outputNodes[i]._weights[j] = weightDist(generator);
	}
}

void DBN::trainLayerUnsupervised(size_t layerIndex, const std::vector<double> &input, double alpha, std::mt19937 &generator) {
	assert(input.size() == _rbmLayers[layerIndex].getNumVisible());

	for (size_t i = 0; i < input.size(); i++)
		_rbmLayers[layerIndex].setVisible(i, input[i]);

	_rbmLayers[layerIndex].activate(generator);

	_rbmLayers[layerIndex].learn(alpha, generator);
}

void DBN::getLayerOutputMean(size_t layerIndex, const std::vector<double> &input, std::vector<double> &mean) {
	for (size_t i = 0; i < _rbmLayers[layerIndex].getNumVisible(); i++)
		_rbmLayers[layerIndex].setVisible(i, input[i]);

	_rbmLayers[layerIndex].activateLight();

	mean.resize(_rbmLayers[layerIndex].getNumHidden());

	for (size_t i = 0; i < _rbmLayers[layerIndex].getNumHidden(); i++)
		mean[i] = _rbmLayers[layerIndex].getHidden(i);
}

void DBN::getOutputMeanThroughLayers(size_t numLayers, const std::vector<double> &input, std::vector<double> &mean) {
	mean = input;

	for (size_t l = 0; l < numLayers; l++) {
		for (size_t i = 0; i < _rbmLayers[l].getNumVisible(); i++)
			_rbmLayers[l].setVisible(i, mean[i]);

		_rbmLayers[l].activateLight();

		mean.resize(_rbmLayers[l].getNumHidden());

		for (size_t i = 0; i < _rbmLayers[l].getNumHidden(); i++)
			mean[i] = _rbmLayers[l].getHidden(i);
	}
}

void DBN::prepareForGradientDescent() {
	for (size_t l = 0; l < _rbmLayers.size(); l++)
	for (size_t i = 0; i < _rbmLayers[l].getNumHidden(); i++)
	for (size_t j = 0; j < _rbmLayers[l]._hidden[i]._connections.size(); j++) {
		_rbmLayers[l]._hidden[i]._connections[j]._positive = 0.0f;
		_rbmLayers[l]._hidden[i]._connections[j]._negative = 0.0f;
	}
}

void DBN::execute(const std::vector<double> &input, std::vector<double> &output) {
	assert(input.size() == _rbmLayers[0].getNumVisible());

	_input = input;

	for (size_t i = 0; i < _rbmLayers[0].getNumVisible(); i++)
		_rbmLayers[0].setVisible(i, _input[i]);

	_rbmLayers[0].activateLight();

	for (size_t l = 1; l < _rbmLayers.size(); l++) {
		size_t lowerLayerIndex = l - 1;

		for (size_t i = 0; i < _rbmLayers[l].getNumVisible(); i++)
			_rbmLayers[l].setVisible(i, _rbmLayers[lowerLayerIndex].getHidden(i));

		_rbmLayers[l].activateLight();
	}

	// Output layer
	if (output.size() != _outputNodes.size())
		output.resize(_outputNodes.size());

	for (size_t i = 0; i < _outputNodes.size(); i++) {
		double sum = _outputNodes[i]._bias;

		for (size_t j = 0; j < _rbmLayers.back().getNumHidden(); j++)
			sum += _outputNodes[i]._weights[j] * _rbmLayers.back().getHidden(j);

		output[i] = _outputNodes[i]._output = sum;
	}
}

void DBN::getError(const std::vector<double> &target) {
	assert(target.size() == _outputNodes.size());

	// Output layer
	for (size_t i = 0; i < _outputNodes.size(); i++)
		_outputNodes[i]._error = target[i] - _outputNodes[i]._output;

	// Last RBM
	for (size_t i = 0; i < _rbmErrors.back().size(); i++) {
		double sum = 0.0;

		for (size_t j = 0; j < _outputNodes.size(); j++)
			sum += _outputNodes[j]._error * _outputNodes[j]._weights[i];
		
		_rbmErrors.back()[i] = sum * _rbmLayers.back().getHidden(i) * (1.0 - _rbmLayers.back().getHidden(i));
	}
	
	// All other RBMs
	for (int l = static_cast<int>(_rbmLayers.size()) - 2; l >= 0; l--) {
		int upperLayerIndex = l + 1;

		for (size_t i = 0; i < _rbmLayers[l].getNumHidden(); i++) {
			double sum = 0.0;

			for (size_t j = 0; j < _rbmErrors[upperLayerIndex].size(); j++)
				sum += _rbmErrors[upperLayerIndex][j] * _rbmLayers[upperLayerIndex]._hidden[j]._connections[i]._weight;

			_rbmErrors[l][i] = sum * _rbmLayers[l].getHidden(i) * (1.0 - _rbmLayers[l].getHidden(i));
		}
	}
}

void DBN::moveAlongGradient(double alpha, double momentum, double alphaLayerMuliplier) {
	// Output layer
	for (size_t i = 0; i < _outputNodes.size(); i++) {
		double dBias = alpha * _outputNodes[i]._error + _outputNodes[i]._prevDBias * momentum;
		_outputNodes[i]._bias += dBias;
		_outputNodes[i]._prevDBias = dBias;

		for (size_t j = 0; j < _outputNodes[i]._weights.size(); j++) {
			double dWeight = alpha * _outputNodes[i]._error * _rbmLayers.back().getHidden(j) + _outputNodes[i]._prevDWeights[j] * momentum;
			_outputNodes[i]._weights[j] += dWeight;
			_outputNodes[i]._prevDWeights[j] = dWeight;
		}
	}

	alpha *= alphaLayerMuliplier;

	// All RBMs but first
	for (size_t l = static_cast<int>(_rbmLayers.size()) - 1; l >= 1; l--) {
		size_t lowerLayerIndex = l - 1;

		for (size_t i = 0; i < _rbmLayers[l].getNumHidden(); i++) {
			double dBias = alpha * _rbmErrors[l][i] + _rbmLayers[l]._hidden[i]._connections.back()._positive * momentum;
			_rbmLayers[l]._hidden[i]._connections.back()._weight += dBias;
			_rbmLayers[l]._hidden[i]._connections.back()._positive = dBias;

			for (size_t j = 0; j < _rbmLayers[l]._hidden[i]._connections.size() - 1; j++) {
				float dWeight = alpha * _rbmErrors[l][i] * _rbmLayers[lowerLayerIndex].getHidden(j) + _rbmLayers[l]._hidden[i]._connections[j]._positive * momentum;
				_rbmLayers[l]._hidden[i]._connections[j]._weight += dWeight;
				_rbmLayers[l]._hidden[i]._connections[j]._positive = dWeight;
			}
		}

		alpha *= alphaLayerMuliplier;
	}

	// First RBM
	for (size_t i = 0; i < _rbmLayers[0].getNumHidden(); i++) {
		double dBias = alpha * _rbmErrors[0][i] + _rbmLayers[0]._hidden[i]._connections.back()._positive * momentum;
		_rbmLayers[0]._hidden[i]._connections.back()._weight += dBias;
		_rbmLayers[0]._hidden[i]._connections.back()._positive = dBias;

		for (size_t j = 0; j < _rbmLayers[0]._hidden[i]._connections.size() - 1; j++) {
			float dWeight = alpha * _rbmErrors[0][i] * _input[j] + _rbmLayers[0]._hidden[i]._connections[j]._positive * momentum;
			_rbmLayers[0]._hidden[i]._connections[j]._weight += dWeight;
			_rbmLayers[0]._hidden[i]._connections[j]._positive = dWeight;
		}
	}
}

void DBN::signError() {
	for (size_t i = 0; i < _outputNodes.size(); i++)
		_outputNodes[i]._error = _outputNodes[i]._error > 0.0 ? 1.0 : -1.0;

	for (size_t i = 0; i < _rbmErrors.size(); i++)
	for (size_t j = 0; j < _rbmErrors[i].size(); j++)
		_rbmErrors[i][j] = _rbmErrors[i][j] > 0.0 ? 1.0 : -1.0;
}

void DBN::accumulateGradient() {
	// Output layer
	for (size_t i = 0; i < _outputNodes.size(); i++) {
		_outputNodes[i]._accumulatedDBias += _outputNodes[i]._error;
	
		for (size_t j = 0; j < _outputNodes[i]._weights.size(); j++)
			_outputNodes[i]._accumulatedDWeight[j] += _outputNodes[i]._error * _rbmLayers.back().getHidden(j);
	}

	// All RBMs but first
	for (int l = static_cast<int>(_rbmLayers.size()) - 1; l >= 1; l--) {
		size_t lowerLayerIndex = l - 1;

		for (size_t i = 0; i < _rbmLayers[l].getNumHidden(); i++) {
			_rbmLayers[l]._hidden[i]._connections.back()._negative += _rbmErrors[l][i];

			for (size_t j = 0; j < _rbmLayers[l]._hidden[i]._connections.size() - 1; j++)
				_rbmLayers[l]._hidden[i]._connections[j]._negative += _rbmErrors[l][i] * _rbmLayers[lowerLayerIndex].getHidden(j);
		}
	}

	// First RBM
	for (size_t i = 0; i < _rbmLayers[0].getNumHidden(); i++) {
		_rbmLayers[0]._hidden[i]._connections.back()._negative += _rbmErrors[0][i];
	
		for (size_t j = 0; j < _rbmLayers[0]._hidden[i]._connections.size() - 1; j++)
			_rbmLayers[0]._hidden[i]._connections[j]._negative += _rbmErrors[0][i] * _input[j];
	}
}

void DBN::moveAlongAccumulatedGradient(double alpha) {
	// Output layer
	for (size_t i = 0; i < _outputNodes.size(); i++) {
		_outputNodes[i]._bias += alpha * _outputNodes[i]._accumulatedDBias;
		_outputNodes[i]._accumulatedDBias = 0.0;

		for (size_t j = 0; j < _outputNodes[i]._weights.size(); j++) {
			_outputNodes[i]._weights[j] += alpha * _outputNodes[i]._accumulatedDWeight[j];
			_outputNodes[i]._accumulatedDWeight[j] = 0.0;
		}
	}

	// All RBMs
	for (int l = static_cast<int>(_rbmLayers.size()) - 1; l >= 0; l--)
	for (size_t i = 0; i < _rbmLayers[l].getNumHidden(); i++) {
		_rbmLayers[l]._hidden[i]._connections.back()._weight += alpha * _rbmLayers[l]._hidden[i]._connections.back()._negative;
		_rbmLayers[l]._hidden[i]._connections.back()._negative = 0.0;

		for (size_t j = 0; j < _rbmLayers[l]._hidden[i]._connections.size() - 1; j++) {
			_rbmLayers[l]._hidden[i]._connections[j]._weight += alpha * _rbmLayers[l]._hidden[i]._connections[j]._negative;
			_rbmLayers[l]._hidden[i]._connections[j]._negative = 0.0;
		}
	}
}

void DBN::decayWeights(float decayMultiplier) {
	for (size_t l = 0; l < _rbmLayers.size(); l++)
	for (size_t i = 0; i < _rbmLayers[l].getNumHidden(); i++) {
		_rbmLayers[l]._hidden[i]._connections.back()._weight *= decayMultiplier;

		for (size_t j = 0; j < _rbmLayers[l]._hidden[i]._connections.size() - 1; j++)
			_rbmLayers[l]._hidden[i]._connections[j]._weight *= decayMultiplier;
	}

	for (size_t i = 0; i < _outputNodes.size(); i++) {
		_outputNodes[i]._bias *= decayMultiplier;

		for (size_t j = 0; j < _outputNodes[i]._weights.size(); j++)
			_outputNodes[i]._weights[j] *= decayMultiplier;
	}
}

void DBN::decayWeightsLastLayerOnly(float decayMultiplier) {
	for (size_t i = 0; i < _outputNodes.size(); i++) {
		_outputNodes[i]._bias *= decayMultiplier;

		for (size_t j = 0; j < _outputNodes[i]._weights.size(); j++)
			_outputNodes[i]._weights[j] *= decayMultiplier;
	}
}