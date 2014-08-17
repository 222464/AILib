#include <dnf/Field.h>

#include <tuple>

using namespace dnf;

Field::Field()
: _hI(0.0f)
{}

void Field::createRandom(int numInputs, int size, int weightRadius, float weightDeviation, float minWeight, float maxWeight, std::mt19937 &generator) {
	_numInputs = numInputs;
	_size = size;
	_weightRadius = weightRadius;
	_weightDeviation = weightDeviation;

	_inputData.resize(_numInputs);

	for (int i = 0; i < _numInputs; i++) {
		_inputData[i]._averageRate = 0.5f;
		_inputData[i]._bdnf = 1.0f;
	}

	_nodes.resize(_size * _size);

	int weightDimSize = _weightRadius * 2 + 1;
	int numWeights = weightDimSize * weightDimSize;

	std::uniform_real_distribution<float> distWeight(minWeight, maxWeight);

	for (size_t n = 0; n < _nodes.size(); n++) {
		_nodes[n]._activationE = 0.0f;
		_nodes[n]._outputE = 0.5f;
		_nodes[n]._averageRateE = 0.5f;
		_nodes[n]._bdnfE = 1.0f;

		_nodes[n]._activationI = 0.0f;
		_nodes[n]._outputI = 0.5f;
		//_nodes[n]._averageRateI = 0.5f;
		//_nodes[n]._bdnfI = 1.0f;

		_nodes[n]._hE = distWeight(generator);

		_nodes[n]._wee.resize(numWeights);
		_nodes[n]._wei.resize(numWeights);
		_nodes[n]._wie.resize(numWeights);

		_nodes[n]._wext.resize(_numInputs);

		for (int w = 0; w < numWeights; w++) {
			_nodes[n]._wee[w]._weight = distWeight(generator);
			_nodes[n]._wei[w]._weight = distWeight(generator);
			_nodes[n]._wie[w]._weight = distWeight(generator);

			_nodes[n]._wee[w]._eligibilityTrace = 0.0f;
			_nodes[n]._wei[w]._eligibilityTrace = 0.0f;
			_nodes[n]._wie[w]._eligibilityTrace = 0.0f;
		}

		for (int w = 0; w < _numInputs; w++) {
			_nodes[n]._wext[w]._weight = distWeight(generator);

			_nodes[n]._wext[w]._eligibilityTrace = 0.0f;
		}
	}

	// Precompute g
	_gLookup.resize(weightDimSize * weightDimSize);

	float gCoeff = 1.0f / (_weightDeviation * std::sqrt(2.0f * std::_Pi));
	float dCoeff = -1.0f / (2.0f * _weightDeviation);

	for (int i = 0; i < _gLookup.size(); i++) {
		int dx = i % weightDimSize - _weightRadius;
		int dy = i / weightDimSize - _weightRadius;

		float distSquared = dx * dx + dy * dy;

		_gLookup[i] = gCoeff * std::exp(dCoeff * distSquared);
	}
}

void Field::step(const std::vector<float> &input, float dt, float threshold, float gain, float reward, float eligibilityDecay,
	float averageDecay, float homeoAlphaH, float homeoAlphaT, float targetActivation)
{
	std::vector<Node> newNodes = _nodes;

	int weightDimSize = _weightRadius * 2 + 1;

	for (int n = 0; n < newNodes.size(); n++) {
		int x = n % _size;
		int y = n / _size;

		float eeSum = 0.0f;
		float eiSum = 0.0f;
		float ieSum = 0.0f;
		float extSum = 0.0f;

		for (int dx = -_weightRadius; dx <= _weightRadius; dx++)
		for (int dy = -_weightRadius; dy <= _weightRadius; dy++) {
			int nx = (x + dx) % _size;
			int ny = (y + dy) % _size;

			if (nx < 0)
				nx += _size;
			if (ny < 0)
				ny += _size;

			int nCoord = nx + ny * _size;
			int wCoord = dx + _weightRadius + (dy + _weightRadius) * weightDimSize;

			float ou = _nodes[nCoord]._outputE;
			float ov = _nodes[nCoord]._outputI;

			float g = _gLookup[wCoord];

			newNodes[n]._wee[wCoord]._weight = (_nodes[n]._wee[wCoord]._weight + _nodes[n]._wee[wCoord]._eligibilityTrace * reward) / (_nodes[n]._bdnfE * _nodes[nCoord]._bdnfE);
			newNodes[n]._wei[wCoord]._weight = (_nodes[n]._wei[wCoord]._weight + _nodes[n]._wei[wCoord]._eligibilityTrace * reward) * _nodes[n]._bdnfE;
			newNodes[n]._wie[wCoord]._weight = (_nodes[n]._wie[wCoord]._weight + _nodes[n]._wie[wCoord]._eligibilityTrace * reward) * _nodes[nCoord]._bdnfE;

			eeSum += g * newNodes[n]._wee[wCoord]._weight * ou;
			eiSum += newNodes[n]._wei[wCoord]._weight * ov;
			
			ieSum += g * newNodes[n]._wie[wCoord]._weight * ou;
		}

		for (int i = 0; i < _numInputs; i++) {
			newNodes[n]._wext[i]._weight = (_nodes[n]._wext[i]._weight + _nodes[n]._wext[i]._eligibilityTrace * reward) / (newNodes[n]._bdnfE * _inputData[i]._bdnf);

			extSum += newNodes[n]._wext[i]._weight * input[i];
		}

		// Modify resting potential
		newNodes[n]._hE = _nodes[n]._hE + homeoAlphaT * (targetActivation - _nodes[n]._averageRateE) / targetActivation;

		// Calculate excitatory activation
		newNodes[n]._activationE += dt * (-_nodes[n]._activationE + eeSum - eiSum + extSum + newNodes[n]._hE);
		newNodes[n]._outputE = sigmoid((_nodes[n]._activationE - threshold) * gain);
		newNodes[n]._averageRateE = (1.0f - averageDecay) * _nodes[n]._averageRateE + averageDecay * newNodes[n]._outputE;
		newNodes[n]._bdnfE = bdnf(newNodes[n]._averageRateE, targetActivation, homeoAlphaH);

		// Calculate inhibitory activation
		newNodes[n]._activationI += dt * (-_nodes[n]._activationI + ieSum + _hI);
		newNodes[n]._outputI = sigmoid((_nodes[n]._activationI - threshold) * gain);
		//newNodes[n]._averageRateI = (1.0f - averageDecay) * _nodes[n]._averageRateI + averageDecay * newNodes[n]._outputI;
		//newNodes[n]._bdnfI = bdnf(newNodes[n]._averageRateI, targetActivation, homeoAlpha);

		for (int i = 0; i < _numInputs; i++) {
			_inputData[i]._averageRate = (1.0f - averageDecay) * _inputData[i]._averageRate + averageDecay * input[i];
			_inputData[i]._bdnf = bdnf(_inputData[i]._averageRate, targetActivation, homeoAlphaH);
		}

		// Update eligibility traces
		for (int dx = -_weightRadius; dx <= _weightRadius; dx++)
		for (int dy = -_weightRadius; dy <= _weightRadius; dy++) {
			int nx = (x + dx) % _size;
			int ny = (y + dy) % _size;

			if (nx < 0)
				nx += _size;
			if (ny < 0)
				ny += _size;

			int nCoord = nx + ny * _size;
			int wCoord = dx + _weightRadius + (dy + _weightRadius) * weightDimSize;

			newNodes[n]._wee[wCoord]._eligibilityTrace = (1.0f - eligibilityDecay) * _nodes[n]._wee[wCoord]._eligibilityTrace + (newNodes[nCoord]._outputE * newNodes[n]._outputE - newNodes[n]._wee[wCoord]._weight * newNodes[nCoord]._outputE * newNodes[nCoord]._outputE);
			newNodes[n]._wei[wCoord]._eligibilityTrace = (1.0f - eligibilityDecay) * _nodes[n]._wei[wCoord]._eligibilityTrace + (newNodes[nCoord]._outputI * newNodes[n]._outputE - newNodes[n]._wei[wCoord]._weight * newNodes[nCoord]._outputI * newNodes[nCoord]._outputI);
			newNodes[n]._wie[wCoord]._eligibilityTrace = (1.0f - eligibilityDecay) * _nodes[n]._wie[wCoord]._eligibilityTrace + (newNodes[nCoord]._outputE * newNodes[n]._outputI - newNodes[n]._wie[wCoord]._weight * newNodes[nCoord]._outputE * newNodes[nCoord]._outputE);
		}

		for (int i = 0; i < _numInputs; i++)
			newNodes[n]._wext[i]._eligibilityTrace = (1.0f - eligibilityDecay) * _nodes[n]._wext[i]._eligibilityTrace + (input[i] * newNodes[n]._outputE - newNodes[n]._wext[i]._weight * input[i] * input[i]);
	}

	_nodes = newNodes;
}