#include "SRBMFA.h"

#include <algorithm>

using namespace deep;

void SRBMFA::createRandom(int numInputs, const std::vector<LayerDesc> &layerDescs, int numOutputUnits, float weightStdDev, std::mt19937 &generator) {
	std::normal_distribution<float> weightDist(0.0f, weightStdDev);

	_rbmLayers.resize(layerDescs.size());

	int visibleSize = numInputs;

	for (int l = 0; l < _rbmLayers.size(); l++) {
		_rbmLayers[l]._visibleUnits.resize(visibleSize);

		for (int vi = 0; vi < _rbmLayers[l]._visibleUnits.size(); vi++) {
			_rbmLayers[l]._visibleUnits[vi]._bias._rbmWeight = weightDist(generator);
			_rbmLayers[l]._visibleUnits[vi]._bias._ffnnWeight = weightDist(generator);
		}

		_rbmLayers[l]._hiddenUnits.resize(layerDescs[l]._numHiddenUnits);

		for (int hi = 0; hi < _rbmLayers[l]._hiddenUnits.size(); hi++) {
			_rbmLayers[l]._hiddenUnits[hi]._bias._rbmWeight = weightDist(generator);
			_rbmLayers[l]._hiddenUnits[hi]._bias._ffnnWeight = weightDist(generator);

			_rbmLayers[l]._hiddenUnits[hi]._connections.resize(_rbmLayers[l]._visibleUnits.size());

			for (int vi = 0; vi < _rbmLayers[l]._visibleUnits.size(); vi++) {
				_rbmLayers[l]._hiddenUnits[hi]._connections[vi]._rbmWeight = weightDist(generator);
				_rbmLayers[l]._hiddenUnits[hi]._connections[vi]._ffnnWeight = weightDist(generator);
			}
		}

		visibleSize = layerDescs[l]._numHiddenUnits;
	}

	_outputUnits.resize(numOutputUnits);

	for (int oi = 0; oi < _outputUnits.size(); oi++) {
		_outputUnits[oi]._bias._weight = weightDist(generator);

		_outputUnits[oi]._connections.resize(_rbmLayers.back()._hiddenUnits.size());

		for (int hi = 0; hi < _rbmLayers.back()._hiddenUnits.size(); hi++)
			_outputUnits[oi]._connections[hi]._weight = weightDist(generator);
	}
}

void SRBMFA::activate(float sparsity, int numLayersActivate) {
	int layersActivate = numLayersActivate < 0 ? _rbmLayers.size() : std::min<int>(numLayersActivate, _rbmLayers.size());

	for (int l = 0; l < layersActivate; l++) {
		int localActivity = std::round(_rbmLayers[l]._hiddenUnits.size() * sparsity);

		if (l != 0) {
			for (int vi = 0; vi < _rbmLayers[l]._visibleUnits.size(); vi++)
				_rbmLayers[l]._visibleUnits[vi]._input = _rbmLayers[l - 1]._hiddenUnits[vi]._ffnnOutput;
		}

		// Activate
		for (int hi = 0; hi < _rbmLayers[l]._hiddenUnits.size(); hi++) {
			float sum = _rbmLayers[l]._hiddenUnits[hi]._bias._rbmWeight;

			for (int vi = 0; vi < _rbmLayers[l]._visibleUnits.size(); vi++)
				sum += _rbmLayers[l]._visibleUnits[vi]._input * _rbmLayers[l]._hiddenUnits[hi]._connections[vi]._rbmWeight;

			_rbmLayers[l]._hiddenUnits[hi]._activation = sum;
		}

		// Inhibit
		for (int hi = 0; hi < _rbmLayers[l]._hiddenUnits.size(); hi++) {
			float numHigher = 0.0f;

			for (int hi2 = 0; hi2 < _rbmLayers[l]._hiddenUnits.size(); hi2++) {
				if (_rbmLayers[l]._hiddenUnits[hi2]._activation > _rbmLayers[l]._hiddenUnits[hi]._activation)
					numHigher += 1.0f;
			}

			_rbmLayers[l]._hiddenUnits[hi]._probability = numHigher < localActivity ? 1.0f : 0.0f;

			_rbmLayers[l]._hiddenUnits[hi]._output = _rbmLayers[l]._hiddenUnits[hi]._probability;
		}

		for (int hi = 0; hi < _rbmLayers[l]._hiddenUnits.size(); hi++) {
			float sum = _rbmLayers[l]._hiddenUnits[hi]._bias._ffnnWeight;

			for (int vi = 0; vi < _rbmLayers[l]._visibleUnits.size(); vi++)
				sum += _rbmLayers[l]._visibleUnits[vi]._input * _rbmLayers[l]._hiddenUnits[hi]._connections[vi]._ffnnWeight;

			_rbmLayers[l]._hiddenUnits[hi]._sig = sigmoid(sum);
			_rbmLayers[l]._hiddenUnits[hi]._ffnnOutput = _rbmLayers[l]._hiddenUnits[hi]._sig * _rbmLayers[l]._hiddenUnits[hi]._output;
		}
	}

	if (numLayersActivate == -1) {
		for (int oi = 0; oi < _outputUnits.size(); oi++) {
			float sum = _outputUnits[oi]._bias._weight;

			for (int hi = 0; hi < _rbmLayers.back()._hiddenUnits.size(); hi++)
				sum += _rbmLayers.back()._hiddenUnits[hi]._ffnnOutput * _outputUnits[oi]._connections[hi]._weight;

			_outputUnits[oi]._output = sum;
		}
	}
}

std::vector<float> SRBMFA::operator()(const std::vector<float> &input, float averageOutputDecay) {
	for (int vi = 0; vi < _rbmLayers.front()._visibleUnits.size(); vi++)
		setInput(vi, input[vi]);

	activate(averageOutputDecay);

	std::vector<float> output(_outputUnits.size());

	for (int oi = 0; oi < _outputUnits.size(); oi++)
		output[oi] = getOutputUnitOutput(oi);

	return output;
}

void SRBMFA::learnRBM(int layer, float rbmAlpha, float rbmBeta, float rbmMomentum, float columnRandomness, float sparsity, int gibbsSampleIterations, std::mt19937 &generator) {
	std::uniform_real_distribution<float> uniformDist(0.0f, 1.0f);
	
	int localActivity = std::round(_rbmLayers[layer]._hiddenUnits.size() * sparsity);

	// Activate
	for (int hi = 0; hi < _rbmLayers[layer]._hiddenUnits.size(); hi++) {
		float sum = _rbmLayers[layer]._hiddenUnits[hi]._bias._rbmWeight;

		for (int vi = 0; vi < _rbmLayers[layer]._visibleUnits.size(); vi++)
			sum += _rbmLayers[layer]._visibleUnits[vi]._input * _rbmLayers[layer]._hiddenUnits[hi]._connections[vi]._rbmWeight;

		_rbmLayers[layer]._hiddenUnits[hi]._activation = sum;
	}

	// Inhibit
	for (int hi = 0; hi < _rbmLayers[layer]._hiddenUnits.size(); hi++) {
		float numHigher = 0.0f;

		for (int hi2 = 0; hi2 < _rbmLayers[layer]._hiddenUnits.size(); hi2++) {
			if (_rbmLayers[layer]._hiddenUnits[hi2]._activation > _rbmLayers[layer]._hiddenUnits[hi]._activation)
				numHigher += 1.0f;
		}

		_rbmLayers[layer]._hiddenUnits[hi]._firstProbability = numHigher < localActivity ? 1.0f - columnRandomness : columnRandomness;

		_rbmLayers[layer]._hiddenUnits[hi]._output = uniformDist(generator) < _rbmLayers[layer]._hiddenUnits[hi]._firstProbability ? 1.0f : 0.0f;
	}

	// Reconstruct
	for (int vi = 0; vi < _rbmLayers[layer]._visibleUnits.size(); vi++) {
		float sum = 0.0f;// _rbmLayers[layer]._visibleUnits[vi]._bias._rbmWeight;

		for (int hi = 0; hi < _rbmLayers[layer]._hiddenUnits.size(); hi++)
			sum += _rbmLayers[layer]._hiddenUnits[hi]._output * _rbmLayers[layer]._hiddenUnits[hi]._connections[vi]._rbmWeight;

		_rbmLayers[layer]._visibleUnits[vi]._probability = sum;
	}

	// Perform Gibbs sampling
	for (int gs = 0; gs < gibbsSampleIterations - 1; gs++) {
		// Activate
		for (int hi = 0; hi < _rbmLayers[layer]._hiddenUnits.size(); hi++) {
			float sum = _rbmLayers[layer]._hiddenUnits[hi]._bias._rbmWeight;

			for (int vi = 0; vi < _rbmLayers[layer]._visibleUnits.size(); vi++)
				sum += _rbmLayers[layer]._visibleUnits[vi]._input * _rbmLayers[layer]._hiddenUnits[hi]._connections[vi]._rbmWeight;

			_rbmLayers[layer]._hiddenUnits[hi]._activation = sum;
		}

		// Inhibit
		for (int hi = 0; hi < _rbmLayers[layer]._hiddenUnits.size(); hi++) {
			float numHigher = 0.0f;

			for (int hi2 = 0; hi2 < _rbmLayers[layer]._hiddenUnits.size(); hi2++) {
				if (_rbmLayers[layer]._hiddenUnits[hi2]._activation > _rbmLayers[layer]._hiddenUnits[hi]._activation)
					numHigher += 1.0f;
			}

			_rbmLayers[layer]._hiddenUnits[hi]._probability = numHigher < localActivity ? 1.0f : 0.0f;

			_rbmLayers[layer]._hiddenUnits[hi]._output = _rbmLayers[layer]._hiddenUnits[hi]._probability;// uniformDist(generator) < _rbmLayers[layer]._hiddenUnits[hi]._probability ? 1.0f : 0.0f;
		}

		// Reconstruct
		for (int vi = 0; vi < _rbmLayers[layer]._visibleUnits.size(); vi++) {
			float sum = _rbmLayers[layer]._visibleUnits[vi]._bias._rbmWeight;

			for (int hi = 0; hi < _rbmLayers[layer]._hiddenUnits.size(); hi++)
				sum += _rbmLayers[layer]._hiddenUnits[hi]._output * _rbmLayers[layer]._hiddenUnits[hi]._connections[vi]._rbmWeight;

			_rbmLayers[layer]._visibleUnits[vi]._probability = sum;
		}
	}

	// Activate
	for (int hi = 0; hi < _rbmLayers[layer]._hiddenUnits.size(); hi++) {
		float sum = _rbmLayers[layer]._hiddenUnits[hi]._bias._rbmWeight;

		for (int vi = 0; vi < _rbmLayers[layer]._visibleUnits.size(); vi++)
			sum += _rbmLayers[layer]._visibleUnits[vi]._input * _rbmLayers[layer]._hiddenUnits[hi]._connections[vi]._rbmWeight;

		_rbmLayers[layer]._hiddenUnits[hi]._activation = sum;
	}

	// Inhibit
	for (int hi = 0; hi < _rbmLayers[layer]._hiddenUnits.size(); hi++) {
		float numHigher = 0.0f;

		for (int hi2 = 0; hi2 < _rbmLayers[layer]._hiddenUnits.size(); hi2++) {
			if (_rbmLayers[layer]._hiddenUnits[hi2]._activation > _rbmLayers[layer]._hiddenUnits[hi]._activation)
				numHigher += 1.0f;
		}

		_rbmLayers[layer]._hiddenUnits[hi]._probability = numHigher < localActivity ? 1.0f : 0.0f;

		_rbmLayers[layer]._hiddenUnits[hi]._output = _rbmLayers[layer]._hiddenUnits[hi]._probability;
	}

	// Contrastive divergence
	for (int vi = 0; vi < _rbmLayers[layer]._visibleUnits.size(); vi++) {
		float biasDelta = rbmMomentum * _rbmLayers[layer]._visibleUnits[vi]._bias._rbmPrevDWeight + rbmAlpha * (_rbmLayers[layer]._visibleUnits[vi]._input - _rbmLayers[layer]._visibleUnits[vi]._probability);

		_rbmLayers[layer]._visibleUnits[vi]._bias._rbmWeight += biasDelta;
		_rbmLayers[layer]._visibleUnits[vi]._bias._rbmPrevDWeight = biasDelta;
	}

	for (int hi = 0; hi < _rbmLayers[layer]._hiddenUnits.size(); hi++) {
		float biasDelta = rbmMomentum * _rbmLayers[layer]._hiddenUnits[hi]._bias._rbmPrevDWeight + rbmAlpha * (_rbmLayers[layer]._hiddenUnits[hi]._firstProbability - _rbmLayers[layer]._hiddenUnits[hi]._probability);

		_rbmLayers[layer]._hiddenUnits[hi]._bias._rbmWeight += biasDelta;
		_rbmLayers[layer]._hiddenUnits[hi]._bias._rbmPrevDWeight = biasDelta;

		for (int vi = 0; vi < _rbmLayers[layer]._visibleUnits.size(); vi++) {
			float delta = rbmMomentum * _rbmLayers[layer]._hiddenUnits[hi]._connections[vi]._rbmPrevDWeight + rbmAlpha * (_rbmLayers[layer]._visibleUnits[vi]._input - _rbmLayers[layer]._visibleUnits[vi]._probability);

			_rbmLayers[layer]._hiddenUnits[hi]._connections[vi]._rbmWeight += delta;
			_rbmLayers[layer]._hiddenUnits[hi]._connections[vi]._rbmPrevDWeight = delta;
		}
	}
}

void SRBMFA::learnFFNN(float ffnnAlpha, float ffnnMomentum) {
	// Calculate errors
	for (int oi = 0; oi < _outputUnits.size(); oi++)
		_outputUnits[oi]._error = _outputUnits[oi]._target - _outputUnits[oi]._output;

	// Last hidden layer
	for (int hi = 0; hi < _rbmLayers.back()._hiddenUnits.size(); hi++) {
		float error = 0.0f;

		for (int oi = 0; oi < _outputUnits.size(); oi++)
			error += _outputUnits[oi]._connections[hi]._weight * _outputUnits[oi]._error;

		error *= _rbmLayers.back()._hiddenUnits[hi]._sig * (1.0f - _rbmLayers.back()._hiddenUnits[hi]._sig) * _rbmLayers.back()._hiddenUnits[hi]._output;

		_rbmLayers.back()._hiddenUnits[hi]._error = error;
	}

	// All other hidden layers
	for (int l = _rbmLayers.size() - 2; l >= 0; l--) {
		for (int hi = 0; hi < _rbmLayers[l]._hiddenUnits.size(); hi++) {
			float error = 0.0f;

			for (int hi2 = 0; hi2 < _rbmLayers[l + 1]._hiddenUnits.size(); hi2++)
				error += _rbmLayers[l + 1]._hiddenUnits[hi2]._connections[hi]._ffnnWeight * _rbmLayers[l + 1]._hiddenUnits[hi2]._error;

			error *= _rbmLayers[l]._hiddenUnits[hi]._sig * (1.0f - _rbmLayers[l]._hiddenUnits[hi]._sig) * _rbmLayers[l]._hiddenUnits[hi]._output;

			_rbmLayers[l]._hiddenUnits[hi]._error = error;
		}
	}

	// Weight updates
	for (int oi = 0; oi < _outputUnits.size(); oi++) {
		float biasDelta = ffnnMomentum * _outputUnits[oi]._bias._prevDWeight + ffnnAlpha * _outputUnits[oi]._error;

		_outputUnits[oi]._bias._weight += biasDelta;
		_outputUnits[oi]._bias._prevDWeight = biasDelta;

		for (int hi = 0; hi < _rbmLayers.back()._hiddenUnits.size(); hi++) {
			float delta = ffnnMomentum * _outputUnits[oi]._connections[hi]._prevDWeight + ffnnAlpha * _outputUnits[oi]._error * _rbmLayers.back()._hiddenUnits[hi]._ffnnOutput;

			_outputUnits[oi]._connections[hi]._weight += delta;
			_outputUnits[oi]._connections[hi]._prevDWeight = delta;
		}
	}

	for (int l = 0; l < _rbmLayers.size(); l++) {
		for (int hi = 0; hi < _rbmLayers.back()._hiddenUnits.size(); hi++) {
			float biasDelta = ffnnMomentum * _rbmLayers[l]._hiddenUnits[hi]._bias._ffnnPrevDWeight + ffnnAlpha * _rbmLayers[l]._hiddenUnits[hi]._error;

			_rbmLayers[l]._hiddenUnits[hi]._bias._ffnnWeight += biasDelta;
			_rbmLayers[l]._hiddenUnits[hi]._bias._ffnnPrevDWeight = biasDelta;

			for (int vi = 0; vi < _rbmLayers[l]._visibleUnits.size(); vi++) {
				float delta = ffnnMomentum * _rbmLayers[l]._hiddenUnits[hi]._connections[vi]._ffnnPrevDWeight + ffnnAlpha * _rbmLayers[l]._hiddenUnits[hi]._error * _rbmLayers[l]._visibleUnits[vi]._input;

				_rbmLayers[l]._hiddenUnits[hi]._connections[vi]._ffnnWeight += delta;
				_rbmLayers[l]._hiddenUnits[hi]._connections[vi]._ffnnPrevDWeight = delta;
			}
		}
	}
}

std::ostream &deep::operator<<(std::ostream &os, const SRBMFA &fa) {
	for (int l = 0; l < fa._rbmLayers.size(); l++) {
		for (int hi = 0; hi < fa._rbmLayers[l]._hiddenUnits.size(); hi++)
			if (fa._rbmLayers[l]._hiddenUnits[hi]._output > 0.5f)
				std::cout << "X";
			else
				std::cout << "_";

		os << std::endl;
	}

	return os;
}