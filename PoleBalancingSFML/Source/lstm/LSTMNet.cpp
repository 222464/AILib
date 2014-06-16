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

#include <lstm/LSTMNet.h>

#include <algorithm>

#include <iostream>

using namespace lstm;

void LSTMNet::createRandom(size_t numInputs, size_t numOutputs, size_t hiddenSize, size_t numMemoryCells, float minWeight, float maxWeight, std::mt19937 &generator) {
	_currentInputs.assign(numInputs, 0.0f);

	_outputNodes.resize(numOutputs);
	_hiddenGroups.resize(numOutputs);
	_memoryCells.resize(numMemoryCells);

	std::uniform_real_distribution<float> weightDist(minWeight, maxWeight);

	const size_t numOutputInputs = hiddenSize + numMemoryCells;
	const size_t numHiddenInputs = numInputs + numMemoryCells;
	const size_t numGateInputs = numMemoryCells * 3 + numInputs + numMemoryCells + hiddenSize * numOutputs;
	const size_t numMemoryCellInputs = numInputs + numMemoryCells + hiddenSize * numOutputs;

	for (size_t i = 0; i < numOutputs; i++) {
		_outputNodes[i]._bias._weight = weightDist(generator);
		_outputNodes[i]._bias._trace = 0.0f;
		_outputNodes[i]._prevOutput = _outputNodes[i]._output = 0.0f;

		_outputNodes[i]._synapses.resize(numOutputInputs);

		for (size_t j = 0; j < _outputNodes[i]._synapses.size(); j++) {
			_outputNodes[i]._synapses[j]._weight = weightDist(generator);
			_outputNodes[i]._synapses[j]._trace = 0.0f;
		}

		_hiddenGroups[i].resize(hiddenSize);

		for (size_t j = 0; j < hiddenSize; j++) {
			_hiddenGroups[i][j]._bias._weight = weightDist(generator);
			_hiddenGroups[i][j]._bias._trace = 0.0f;
			_hiddenGroups[i][j]._prevOutput = _hiddenGroups[i][j]._output = 0.0f;

			_hiddenGroups[i][j]._synapses.resize(numHiddenInputs);

			for (size_t k = 0; k < _hiddenGroups[i][j]._synapses.size(); k++) {
				_hiddenGroups[i][j]._synapses[k]._weight = weightDist(generator);
				_hiddenGroups[i][j]._synapses[k]._trace = 0.0f;
			}
		}
	}

	for (size_t i = 0; i < numMemoryCells; i++) {
		_memoryCells[i]._outputGate._bias._weight = weightDist(generator);
		_memoryCells[i]._outputGate._bias._trace = 0.0f;
		_memoryCells[i]._outputGate._prevOutput = _memoryCells[i]._outputGate._output = 0.0f;

		_memoryCells[i]._inputGate._bias._weight = weightDist(generator);
		_memoryCells[i]._inputGate._bias._trace = 0.0f;
		_memoryCells[i]._inputGate._bias._derivative = 0.0f;
		_memoryCells[i]._inputGate._prevOutput = _memoryCells[i]._inputGate._output = 0.0f;

		_memoryCells[i]._forgetGate._bias._weight = weightDist(generator);
		_memoryCells[i]._forgetGate._bias._trace = 0.0f;
		_memoryCells[i]._forgetGate._bias._derivative = 0.0f;
		_memoryCells[i]._forgetGate._prevOutput = _memoryCells[i]._forgetGate._output = 0.0f;

		_memoryCells[i]._outputGate._synapses.resize(numGateInputs);
		_memoryCells[i]._inputGate._synapses.resize(numGateInputs);
		_memoryCells[i]._forgetGate._synapses.resize(numGateInputs);

		for (size_t j = 0; j < numGateInputs; j++) {
			_memoryCells[i]._outputGate._synapses[j]._weight = weightDist(generator);
			_memoryCells[i]._outputGate._synapses[j]._trace = 0.0f;

			_memoryCells[i]._inputGate._synapses[j]._weight = weightDist(generator);
			_memoryCells[i]._inputGate._synapses[j]._trace = 0.0f;
			_memoryCells[i]._inputGate._synapses[j]._derivative = 0.0f;

			_memoryCells[i]._forgetGate._synapses[j]._weight = weightDist(generator);
			_memoryCells[i]._forgetGate._synapses[j]._trace = 0.0f;
			_memoryCells[i]._forgetGate._synapses[j]._derivative = 0.0f;
		}

		_memoryCells[i]._net = 0.0f;
		_memoryCells[i]._bias._weight = weightDist(generator);
		_memoryCells[i]._bias._trace = 0.0f;
		_memoryCells[i]._bias._derivative = 0.0f;
		_memoryCells[i]._prevOutput = _memoryCells[i]._output = 0.0f;
		_memoryCells[i]._state = 0.0f;

		_memoryCells[i]._synapses.resize(numGateInputs);

		for (size_t j = 0; j < numGateInputs; j++) {
			_memoryCells[i]._synapses[j]._weight = weightDist(generator);
			_memoryCells[i]._synapses[j]._trace = 0.0f;
			_memoryCells[i]._synapses[j]._derivative = 0.0f;
		}
	}
}

void LSTMNet::step() {
	// Update hidden units
	for (size_t i = 0; i < _hiddenGroups.size(); i++)
	for (size_t j = 0; j < _hiddenGroups[i].size(); j++) {
		float sum = _hiddenGroups[i][j]._bias._weight;

		size_t si = 0;

		for (size_t k = 0; k < _currentInputs.size(); k++)
			sum += _currentInputs[k] * _hiddenGroups[i][j]._synapses[si++]._weight;

		for (size_t k = 0; k < _memoryCells.size(); k++)
			sum += _memoryCells[k]._prevOutput * _hiddenGroups[i][j]._synapses[si++]._weight;

		_hiddenGroups[i][j]._output = sigmoid(sum);
	}

	// Update memory cells
	for (size_t i = 0; i < _memoryCells.size(); i++) {
		float sum;
		size_t si;

		// ----------------------------- Update Gates -----------------------------

		// Output
		sum = _memoryCells[i]._outputGate._bias._weight;

		si = 0;

		for (size_t j = 0; j < _memoryCells.size(); j++) {
			sum += _memoryCells[j]._outputGate._prevOutput * _memoryCells[i]._outputGate._synapses[si++]._weight;
			sum += _memoryCells[j]._inputGate._prevOutput * _memoryCells[i]._outputGate._synapses[si++]._weight;
			sum += _memoryCells[j]._forgetGate._prevOutput * _memoryCells[i]._outputGate._synapses[si++]._weight;

			sum += _memoryCells[j]._prevOutput * _memoryCells[i]._outputGate._synapses[si++]._weight;
		}

		for (size_t j = 0; j < _currentInputs.size(); j++)
			sum += _currentInputs[j] * _memoryCells[i]._outputGate._synapses[si++]._weight;

		for (size_t j = 0; j < _hiddenGroups.size(); j++)
		for (size_t k = 0; k < _hiddenGroups[j].size(); k++)
			sum += _hiddenGroups[j][k]._prevOutput * _memoryCells[i]._outputGate._synapses[si++]._weight;

		_memoryCells[i]._outputGate._output = sigmoid(sum);

		// Input
		sum = _memoryCells[i]._inputGate._bias._weight;

		si = 0;

		for (size_t j = 0; j < _memoryCells.size(); j++) {
			sum += _memoryCells[j]._outputGate._prevOutput * _memoryCells[i]._inputGate._synapses[si++]._weight;
			sum += _memoryCells[j]._inputGate._prevOutput * _memoryCells[i]._inputGate._synapses[si++]._weight;
			sum += _memoryCells[j]._forgetGate._prevOutput * _memoryCells[i]._inputGate._synapses[si++]._weight;

			sum += _memoryCells[j]._prevOutput * _memoryCells[i]._inputGate._synapses[si++]._weight;
		}

		for (size_t j = 0; j < _currentInputs.size(); j++)
			sum += _currentInputs[j] * _memoryCells[i]._inputGate._synapses[si++]._weight;

		for (size_t j = 0; j < _hiddenGroups.size(); j++)
		for (size_t k = 0; k < _hiddenGroups[j].size(); k++)
			sum += _hiddenGroups[j][k]._prevOutput * _memoryCells[i]._inputGate._synapses[si++]._weight;

		_memoryCells[i]._inputGate._output = sigmoid(sum);

		// Forget
		sum = _memoryCells[i]._forgetGate._bias._weight;

		si = 0;

		for (size_t j = 0; j < _memoryCells.size(); j++) {
			sum += _memoryCells[j]._outputGate._prevOutput * _memoryCells[i]._forgetGate._synapses[si++]._weight;
			sum += _memoryCells[j]._inputGate._prevOutput * _memoryCells[i]._forgetGate._synapses[si++]._weight;
			sum += _memoryCells[j]._forgetGate._prevOutput * _memoryCells[i]._forgetGate._synapses[si++]._weight;

			sum += _memoryCells[j]._prevOutput * _memoryCells[i]._forgetGate._synapses[si++]._weight;
		}

		for (size_t j = 0; j < _currentInputs.size(); j++)
			sum += _currentInputs[j] * _memoryCells[i]._forgetGate._synapses[si++]._weight;

		for (size_t j = 0; j < _hiddenGroups.size(); j++)
		for (size_t k = 0; k < _hiddenGroups[j].size(); k++)
			sum += _hiddenGroups[j][k]._prevOutput * _memoryCells[i]._forgetGate._synapses[si++]._weight;

		_memoryCells[i]._forgetGate._output = sigmoid(sum);

		// ----------------------------- Update CEC -----------------------------

		sum = _memoryCells[i]._bias._weight;

		si = 0;

		for (size_t j = 0; j < _memoryCells.size(); j++) {
			sum += _memoryCells[j]._outputGate._prevOutput * _memoryCells[i]._synapses[si++]._weight;
			sum += _memoryCells[j]._inputGate._prevOutput * _memoryCells[i]._synapses[si++]._weight;
			sum += _memoryCells[j]._forgetGate._prevOutput * _memoryCells[i]._synapses[si++]._weight;

			sum += _memoryCells[j]._prevOutput * _memoryCells[i]._synapses[si++]._weight;
		}

		for (size_t j = 0; j < _currentInputs.size(); j++)
			sum += _currentInputs[j] * _memoryCells[i]._synapses[si++]._weight;

		for (size_t j = 0; j < _hiddenGroups.size(); j++)
		for (size_t k = 0; k < _hiddenGroups[j].size(); k++)
			sum += _hiddenGroups[j][k]._prevOutput * _memoryCells[i]._synapses[si++]._weight;

		float memoryCellInput = sigmoid(sum) * 4.0f - 2.0f;

		_memoryCells[i]._net = memoryCellInput;

		_memoryCells[i]._state = _memoryCells[i]._forgetGate._output * _memoryCells[i]._state + _memoryCells[i]._inputGate._output * memoryCellInput;

		_memoryCells[i]._output = (sigmoid(_memoryCells[i]._state) * 2.0f - 1.0f) * _memoryCells[i]._outputGate._output;
	}

	// Update output units
	for (size_t i = 0; i < _outputNodes.size(); i++) {
		float sum = _outputNodes[i]._bias._weight;

		size_t si = 0;

		for (size_t j = 0; j < _hiddenGroups[i].size(); j++)
			sum += _hiddenGroups[i][j]._prevOutput * _outputNodes[i]._synapses[si++]._weight;

		for (size_t j = 0; j < _memoryCells.size(); j++)
			sum += _memoryCells[j]._prevOutput * _outputNodes[i]._synapses[si++]._weight;

		_outputNodes[i]._output = sum;
	}
}

void LSTMNet::updateQ(const std::vector<float> &targets, float error, float gammaLambda) {
	// Update hidden units
	for (size_t i = 0; i < _hiddenGroups.size(); i++)
	for (size_t j = 0; j < _hiddenGroups[i].size(); j++) {
		float sum = _hiddenGroups[i][j]._bias._weight;

		size_t si = 0;

		for (size_t k = 0; k < _currentInputs.size(); k++)
			sum += _currentInputs[k] * _hiddenGroups[i][j]._synapses[si++]._weight;

		for (size_t k = 0; k < _memoryCells.size(); k++)
			sum += _memoryCells[k]._prevOutput * _hiddenGroups[i][j]._synapses[si++]._weight;

		_hiddenGroups[i][j]._output = sigmoid(sum);
	}

	// Update memory cells
	for (size_t i = 0; i < _memoryCells.size(); i++) {
		float sum;
		size_t si;

		// ----------------------------- Update Gates -----------------------------

		// Output
		sum = _memoryCells[i]._outputGate._bias._weight;

		si = 0;

		for (size_t j = 0; j < _memoryCells.size(); j++) {
			sum += _memoryCells[j]._outputGate._prevOutput * _memoryCells[i]._outputGate._synapses[si++]._weight;
			sum += _memoryCells[j]._inputGate._prevOutput * _memoryCells[i]._outputGate._synapses[si++]._weight;
			sum += _memoryCells[j]._forgetGate._prevOutput * _memoryCells[i]._outputGate._synapses[si++]._weight;

			sum += _memoryCells[j]._prevOutput * _memoryCells[i]._outputGate._synapses[si++]._weight;
		}

		for (size_t j = 0; j < _currentInputs.size(); j++)
			sum += _currentInputs[j] * _memoryCells[i]._outputGate._synapses[si++]._weight;

		for (size_t j = 0; j < _hiddenGroups.size(); j++)
		for (size_t k = 0; k < _hiddenGroups[j].size(); k++)
			sum += _hiddenGroups[j][k]._prevOutput * _memoryCells[i]._outputGate._synapses[si++]._weight;

		_memoryCells[i]._outputGate._output = sigmoid(sum);

		// Input
		sum = _memoryCells[i]._inputGate._bias._weight;

		si = 0;

		for (size_t j = 0; j < _memoryCells.size(); j++) {
			sum += _memoryCells[j]._outputGate._prevOutput * _memoryCells[i]._inputGate._synapses[si++]._weight;
			sum += _memoryCells[j]._inputGate._prevOutput * _memoryCells[i]._inputGate._synapses[si++]._weight;
			sum += _memoryCells[j]._forgetGate._prevOutput * _memoryCells[i]._inputGate._synapses[si++]._weight;

			sum += _memoryCells[j]._prevOutput * _memoryCells[i]._inputGate._synapses[si++]._weight;
		}

		for (size_t j = 0; j < _currentInputs.size(); j++)
			sum += _currentInputs[j] * _memoryCells[i]._inputGate._synapses[si++]._weight;

		for (size_t j = 0; j < _hiddenGroups.size(); j++)
		for (size_t k = 0; k < _hiddenGroups[j].size(); k++)
			sum += _hiddenGroups[j][k]._prevOutput * _memoryCells[i]._inputGate._synapses[si++]._weight;

		_memoryCells[i]._inputGate._output = sigmoid(sum);

		// Forget
		sum = _memoryCells[i]._forgetGate._bias._weight;

		si = 0;

		for (size_t j = 0; j < _memoryCells.size(); j++) {
			sum += _memoryCells[j]._outputGate._prevOutput * _memoryCells[i]._forgetGate._synapses[si++]._weight;
			sum += _memoryCells[j]._inputGate._prevOutput * _memoryCells[i]._forgetGate._synapses[si++]._weight;
			sum += _memoryCells[j]._forgetGate._prevOutput * _memoryCells[i]._forgetGate._synapses[si++]._weight;

			sum += _memoryCells[j]._prevOutput * _memoryCells[i]._forgetGate._synapses[si++]._weight;
		}

		for (size_t j = 0; j < _currentInputs.size(); j++)
			sum += _currentInputs[j] * _memoryCells[i]._forgetGate._synapses[si++]._weight;

		for (size_t j = 0; j < _hiddenGroups.size(); j++)
		for (size_t k = 0; k < _hiddenGroups[j].size(); k++)
			sum += _hiddenGroups[j][k]._prevOutput * _memoryCells[i]._forgetGate._synapses[si++]._weight;

		_memoryCells[i]._forgetGate._output = sigmoid(sum);

		// ----------------------------- Update CEC -----------------------------

		sum = _memoryCells[i]._bias._weight;

		si = 0;

		for (size_t j = 0; j < _memoryCells.size(); j++) {
			sum += _memoryCells[j]._outputGate._prevOutput * _memoryCells[i]._synapses[si++]._weight;
			sum += _memoryCells[j]._inputGate._prevOutput * _memoryCells[i]._synapses[si++]._weight;
			sum += _memoryCells[j]._forgetGate._prevOutput * _memoryCells[i]._synapses[si++]._weight;

			sum += _memoryCells[j]._prevOutput * _memoryCells[i]._synapses[si++]._weight;
		}

		for (size_t j = 0; j < _currentInputs.size(); j++)
			sum += _currentInputs[j] * _memoryCells[i]._synapses[si++]._weight;

		for (size_t j = 0; j < _hiddenGroups.size(); j++)
		for (size_t k = 0; k < _hiddenGroups[j].size(); k++)
			sum += _hiddenGroups[j][k]._prevOutput * _memoryCells[i]._synapses[si++]._weight;

		float memoryCellInput = sigmoid(sum) * 4.0f - 2.0f;

		_memoryCells[i]._net = memoryCellInput;

		_memoryCells[i]._state = _memoryCells[i]._forgetGate._output * _memoryCells[i]._state + _memoryCells[i]._inputGate._output * memoryCellInput;

		_memoryCells[i]._output = (sigmoid(_memoryCells[i]._state) * 2.0f - 1.0f) * _memoryCells[i]._outputGate._output;
	}

	// Update output units
	for (size_t i = 0; i < _outputNodes.size(); i++) {
		float sum = _outputNodes[i]._bias._weight;

		size_t si = 0;

		for (size_t j = 0; j < _hiddenGroups[i].size(); j++)
			sum += _hiddenGroups[i][j]._prevOutput * _outputNodes[i]._synapses[si++]._weight;

		for (size_t j = 0; j < _memoryCells.size(); j++)
			sum += _memoryCells[j]._prevOutput * _outputNodes[i]._synapses[si++]._weight;

		_outputNodes[i]._output = sum;
	}

	// ------------------------------------------ Offset Outputs ------------------------------------------

	std::vector<float> offsets(_outputNodes.size());

	for (size_t i = 0; i < _outputNodes.size(); i++) {
		float original = _outputNodes[i]._output;

		_outputNodes[i]._output = targets[i];

		offsets[i] = _outputNodes[i]._output - original;
	}

	// ------------------------------------------ Update weights ------------------------------------------

	size_t si;

	// Update output for selected action output unit
	for (size_t i = 0; i < _outputNodes.size(); i++) {
		float z = offsets[i];

		_outputNodes[i]._bias._weight += error * _outputNodes[i]._bias._trace;
		_outputNodes[i]._bias._trace = gammaLambda * _outputNodes[i]._bias._trace + z;

		si = 0;

		for (size_t j = 0; j < _hiddenGroups[i].size(); j++) {
			_outputNodes[i]._synapses[si]._weight += error * _outputNodes[i]._synapses[si]._trace;
			_outputNodes[i]._synapses[si]._trace = gammaLambda * _outputNodes[i]._synapses[si]._trace + z * _hiddenGroups[i][j]._output;

			si++;
		}

		for (size_t j = 0; j < _memoryCells.size(); j++) {
			_outputNodes[i]._synapses[si]._weight += error * _outputNodes[i]._synapses[si]._trace;
			_outputNodes[i]._synapses[si]._trace = gammaLambda * _outputNodes[i]._synapses[si]._trace + z * _memoryCells[j]._output;

			si++;
		}
	}

	// Update hidden units
	for (size_t i = 0; i < _hiddenGroups.size(); i++)
	for (size_t j = 0; j < _hiddenGroups[i].size(); j++) {
		float cTerm = offsets[i] * _outputNodes[i]._synapses[j]._weight * _hiddenGroups[i][j]._output * (1.0f - _hiddenGroups[i][j]._output);

		_hiddenGroups[i][j]._bias._weight += error * _hiddenGroups[i][j]._bias._trace;
		_hiddenGroups[i][j]._bias._trace = gammaLambda * _hiddenGroups[i][j]._bias._trace + cTerm;

		si = 0;

		for (size_t k = 0; k < _currentInputs.size(); k++) {
			_hiddenGroups[i][j]._synapses[si]._weight += error * _hiddenGroups[i][j]._synapses[si]._trace;
			_hiddenGroups[i][j]._synapses[si]._trace = gammaLambda * _hiddenGroups[i][j]._synapses[si]._trace + cTerm * _currentInputs[k];

			si++;
		}

		for (size_t k = 0; k < _memoryCells.size(); k++) {
			_hiddenGroups[i][j]._synapses[si]._weight += error * _hiddenGroups[i][j]._synapses[si]._trace;
			_hiddenGroups[i][j]._synapses[si]._trace = gammaLambda * _hiddenGroups[i][j]._synapses[si]._trace + cTerm * _memoryCells[k]._output;

			si++;
		}
	}

	// Update memory cells
	for (size_t i = 0; i < _memoryCells.size(); i++) {
		// ----------------------------- Update Gates -----------------------------

		const float sigmoidNet = sigmoid(_memoryCells[i]._state);
		const float hNet = sigmoidNet * 2.0f - 1.0f;
		const float hPrimeNet = 2.0f * sigmoidNet * (1.0f - sigmoidNet);

		float wKc = 0.0f;

		for (size_t j = 0; j < _outputNodes.size(); j++)
			wKc += _outputNodes[j]._synapses[_hiddenGroups[j].size() + i]._weight;

		float sum = 0.0f;

		for (size_t j = 0; j < _outputNodes.size(); j++)
		for (size_t k = 0; k < _hiddenGroups[j].size(); k++)
			sum += offsets[j] * _outputNodes[j]._synapses[k]._weight * _hiddenGroups[j][k]._synapses[_currentInputs.size() + i]._weight * _hiddenGroups[j][k]._output * (1.0f - _hiddenGroups[j][k]._output);

		float coeff;

		// Output
		coeff = hNet * (wKc + sum) * _memoryCells[i]._outputGate._output * (1.0f - _memoryCells[i]._outputGate._output);

		_memoryCells[i]._outputGate._bias._weight += error * _memoryCells[i]._outputGate._bias._trace;
		_memoryCells[i]._outputGate._bias._trace = gammaLambda * _memoryCells[i]._outputGate._bias._trace + coeff;

		si = 0;

		for (size_t j = 0; j < _memoryCells.size(); j++) {
			_memoryCells[i]._outputGate._synapses[si]._weight += error * _memoryCells[i]._outputGate._synapses[si]._trace;
			_memoryCells[i]._outputGate._synapses[si]._trace = gammaLambda * _memoryCells[i]._outputGate._synapses[si]._trace + coeff * _memoryCells[j]._output;

			si++;
		}

		for (size_t j = 0; j < _currentInputs.size(); j++) {
			_memoryCells[i]._outputGate._synapses[si]._weight += error * _memoryCells[i]._outputGate._synapses[si]._trace;
			_memoryCells[i]._outputGate._synapses[si]._trace = gammaLambda * _memoryCells[i]._outputGate._synapses[si]._trace + coeff * _currentInputs[j];

			si++;
		}

		for (size_t j = 0; j < _hiddenGroups.size(); j++)
		for (size_t k = 0; k < _hiddenGroups[j].size(); k++) {
			_memoryCells[i]._outputGate._synapses[si]._weight += error * _memoryCells[i]._outputGate._synapses[si]._trace;
			_memoryCells[i]._outputGate._synapses[si]._trace = gammaLambda * _memoryCells[i]._outputGate._synapses[si]._trace + coeff * _hiddenGroups[j][k]._output;

			si++;
		}

		// CEC
		coeff = (wKc + sum) * _memoryCells[i]._outputGate._output * hPrimeNet;

		const float sigmoidNetCEC = sigmoid(_memoryCells[i]._net);
		const float scaledSigmoidNetCEC = 4.0f * sigmoidNetCEC;
		const float gPrimeTimesInputGate = scaledSigmoidNetCEC * (1.0f - sigmoidNetCEC) * _memoryCells[i]._inputGate._output;

		_memoryCells[i]._bias._weight += error * _memoryCells[i]._synapses[si]._trace;
		_memoryCells[i]._bias._trace = gammaLambda * _memoryCells[i]._synapses[si]._trace + coeff * _memoryCells[i]._synapses[si]._derivative;
		_memoryCells[i]._bias._derivative = _memoryCells[i]._synapses[si]._derivative * _memoryCells[i]._forgetGate._output + gPrimeTimesInputGate;

		si = 0;

		for (size_t j = 0; j < _memoryCells.size(); j++) {
			_memoryCells[i]._synapses[si]._weight += error * _memoryCells[i]._synapses[si]._trace;
			_memoryCells[i]._synapses[si]._trace = gammaLambda * _memoryCells[i]._synapses[si]._trace + coeff * _memoryCells[i]._synapses[si]._derivative;
			_memoryCells[i]._synapses[si]._derivative = _memoryCells[i]._synapses[si]._derivative * _memoryCells[i]._forgetGate._output + gPrimeTimesInputGate * _memoryCells[j]._output;

			si++;
		}

		for (size_t j = 0; j < _currentInputs.size(); j++) {
			_memoryCells[i]._synapses[si]._weight += error * _memoryCells[i]._synapses[si]._trace;
			_memoryCells[i]._synapses[si]._trace = gammaLambda * _memoryCells[i]._synapses[si]._trace + coeff * _memoryCells[i]._synapses[si]._derivative;
			_memoryCells[i]._synapses[si]._derivative = _memoryCells[i]._synapses[si]._derivative * _memoryCells[i]._forgetGate._output + gPrimeTimesInputGate * _currentInputs[j];

			si++;
		}

		for (size_t j = 0; j < _hiddenGroups.size(); j++)
		for (size_t k = 0; k < _hiddenGroups[j].size(); k++) {
			_memoryCells[i]._synapses[si]._weight += error * _memoryCells[i]._synapses[si]._trace;
			_memoryCells[i]._synapses[si]._trace = gammaLambda * _memoryCells[i]._synapses[si]._trace + coeff * _memoryCells[i]._synapses[si]._derivative;
			_memoryCells[i]._synapses[si]._derivative = _memoryCells[i]._synapses[si]._derivative * _memoryCells[i]._forgetGate._output + gPrimeTimesInputGate * _hiddenGroups[j][k]._output;

			si++;
		}

		// Input
		// coeff is unchanged from previous, since it is the same equation. Therefore the following line has been commented out
		//coeff = (wKc + sum) * _memoryCells[i]._outputGate._output * hPrimeNet;

		const float gTimesInputPrime = scaledSigmoidNetCEC * _memoryCells[i]._inputGate._output * (1.0f - _memoryCells[i]._inputGate._output);

		_memoryCells[i]._inputGate._bias._weight += error * _memoryCells[i]._inputGate._synapses[si]._trace;
		_memoryCells[i]._inputGate._bias._trace = gammaLambda * _memoryCells[i]._inputGate._synapses[si]._trace + coeff * _memoryCells[i]._synapses[si]._derivative;
		_memoryCells[i]._inputGate._bias._derivative = _memoryCells[i]._synapses[si]._derivative * _memoryCells[i]._forgetGate._output + gTimesInputPrime;

		si = 0;

		for (size_t j = 0; j < _memoryCells.size(); j++) {
			_memoryCells[i]._inputGate._synapses[si]._weight += error * _memoryCells[i]._inputGate._synapses[si]._trace;
			_memoryCells[i]._inputGate._synapses[si]._trace = gammaLambda * _memoryCells[i]._inputGate._synapses[si]._trace + coeff * _memoryCells[i]._synapses[si]._derivative;
			_memoryCells[i]._inputGate._synapses[si]._derivative = _memoryCells[i]._synapses[si]._derivative * _memoryCells[i]._forgetGate._output + gTimesInputPrime * _memoryCells[j]._output;

			si++;
		}

		for (size_t j = 0; j < _currentInputs.size(); j++) {
			_memoryCells[i]._inputGate._synapses[si]._weight += error * _memoryCells[i]._inputGate._synapses[si]._trace;
			_memoryCells[i]._inputGate._synapses[si]._trace = gammaLambda * _memoryCells[i]._inputGate._synapses[si]._trace + coeff * _memoryCells[i]._synapses[si]._derivative;
			_memoryCells[i]._inputGate._synapses[si]._derivative = _memoryCells[i]._synapses[si]._derivative * _memoryCells[i]._forgetGate._output + gTimesInputPrime * _currentInputs[j];

			si++;
		}

		for (size_t j = 0; j < _hiddenGroups.size(); j++)
		for (size_t k = 0; k < _hiddenGroups[j].size(); k++) {
			_memoryCells[i]._inputGate._synapses[si]._weight += error * _memoryCells[i]._inputGate._synapses[si]._trace;
			_memoryCells[i]._inputGate._synapses[si]._trace = gammaLambda * _memoryCells[i]._inputGate._synapses[si]._trace + coeff * _memoryCells[i]._synapses[si]._derivative;
			_memoryCells[i]._inputGate._synapses[si]._derivative = _memoryCells[i]._synapses[si]._derivative * _memoryCells[i]._forgetGate._output + gTimesInputPrime * _hiddenGroups[j][k]._output;

			si++;
		}

		// Forget
		// coeff is unchanged from previous, since it is the same equation. Therefore the following line has been commented out
		//coeff = (wKc + sum) * _memoryCells[i]._outputGate._output * hPrimeNet;

		const float stateTimesForgetPrime = _memoryCells[i]._state * _memoryCells[i]._forgetGate._output * (1.0f - _memoryCells[i]._forgetGate._output);

		_memoryCells[i]._forgetGate._bias._weight += error * _memoryCells[i]._inputGate._synapses[si]._trace;
		_memoryCells[i]._forgetGate._bias._trace = gammaLambda * _memoryCells[i]._inputGate._synapses[si]._trace + coeff * _memoryCells[i]._synapses[si]._derivative;
		_memoryCells[i]._forgetGate._bias._derivative = _memoryCells[i]._synapses[si]._derivative * _memoryCells[i]._forgetGate._output + gTimesInputPrime;

		si = 0;

		for (size_t j = 0; j < _memoryCells.size(); j++) {
			_memoryCells[i]._forgetGate._synapses[si]._weight += error * _memoryCells[i]._forgetGate._synapses[si]._trace;
			_memoryCells[i]._forgetGate._synapses[si]._trace = gammaLambda * _memoryCells[i]._forgetGate._synapses[si]._trace + coeff * _memoryCells[i]._synapses[si]._derivative;
			_memoryCells[i]._forgetGate._synapses[si]._derivative = _memoryCells[i]._synapses[si]._derivative * _memoryCells[i]._forgetGate._output + stateTimesForgetPrime * _memoryCells[j]._output;

			si++;
		}

		for (size_t j = 0; j < _currentInputs.size(); j++) {
			_memoryCells[i]._forgetGate._synapses[si]._weight += error * _memoryCells[i]._forgetGate._synapses[si]._trace;
			_memoryCells[i]._forgetGate._synapses[si]._trace = gammaLambda * _memoryCells[i]._forgetGate._synapses[si]._trace + coeff * _memoryCells[i]._synapses[si]._derivative;
			_memoryCells[i]._forgetGate._synapses[si]._derivative = _memoryCells[i]._synapses[si]._derivative * _memoryCells[i]._forgetGate._output + stateTimesForgetPrime * _currentInputs[j];

			si++;
		}

		for (size_t j = 0; j < _hiddenGroups.size(); j++)
		for (size_t k = 0; k < _hiddenGroups[j].size(); k++) {
			_memoryCells[i]._forgetGate._synapses[si]._weight += error * _memoryCells[i]._forgetGate._synapses[si]._trace;
			_memoryCells[i]._forgetGate._synapses[si]._trace = gammaLambda * _memoryCells[i]._forgetGate._synapses[si]._trace + coeff * _memoryCells[i]._synapses[si]._derivative;
			_memoryCells[i]._forgetGate._synapses[si]._derivative = _memoryCells[i]._synapses[si]._derivative * _memoryCells[i]._forgetGate._output + stateTimesForgetPrime * _hiddenGroups[j][k]._output;

			si++;
		}
	}

	// --------------------------------- Set Previous Outputs to Current ---------------------------------

	for (size_t i = 0; i < _hiddenGroups.size(); i++)
	for (size_t j = 0; j < _hiddenGroups[i].size(); j++)
		_hiddenGroups[i][j]._prevOutput = _hiddenGroups[i][j]._output;

	for (size_t i = 0; i < _memoryCells.size(); i++) {
		_memoryCells[i]._outputGate._prevOutput = _memoryCells[i]._outputGate._output;
		_memoryCells[i]._forgetGate._prevOutput = _memoryCells[i]._forgetGate._output;
		_memoryCells[i]._inputGate._prevOutput = _memoryCells[i]._inputGate._output;

		_memoryCells[i]._prevOutput = _memoryCells[i]._output;
	}

	for (size_t i = 0; i < _outputNodes.size(); i++)
		_outputNodes[i]._prevOutput = _outputNodes[i]._output;
}

void LSTMNet::stepReinforce(float error, float gammaLambda, float offsetStdDev, std::mt19937 &generator) {
	// Update hidden units
	for (size_t i = 0; i < _hiddenGroups.size(); i++)
	for (size_t j = 0; j < _hiddenGroups[i].size(); j++) {
		float sum = _hiddenGroups[i][j]._bias._weight;

		size_t si = 0;

		for (size_t k = 0; k < _currentInputs.size(); k++)
			sum += _currentInputs[k] * _hiddenGroups[i][j]._synapses[si++]._weight;

		for (size_t k = 0; k < _memoryCells.size(); k++)
			sum += _memoryCells[k]._prevOutput * _hiddenGroups[i][j]._synapses[si++]._weight;

		_hiddenGroups[i][j]._output = sigmoid(sum);
	}

	// Update memory cells
	for (size_t i = 0; i < _memoryCells.size(); i++) {
		float sum;
		size_t si;

		// ----------------------------- Update Gates -----------------------------

		// Output
		sum = _memoryCells[i]._outputGate._bias._weight;

		si = 0;

		for (size_t j = 0; j < _memoryCells.size(); j++) {
			sum += _memoryCells[j]._outputGate._prevOutput * _memoryCells[i]._outputGate._synapses[si++]._weight;
			sum += _memoryCells[j]._inputGate._prevOutput * _memoryCells[i]._outputGate._synapses[si++]._weight;
			sum += _memoryCells[j]._forgetGate._prevOutput * _memoryCells[i]._outputGate._synapses[si++]._weight;

			sum += _memoryCells[j]._prevOutput * _memoryCells[i]._outputGate._synapses[si++]._weight;
		}

		for (size_t j = 0; j < _currentInputs.size(); j++)
			sum += _currentInputs[j] * _memoryCells[i]._outputGate._synapses[si++]._weight;

		for (size_t j = 0; j < _hiddenGroups.size(); j++)
		for (size_t k = 0; k < _hiddenGroups[j].size(); k++)
			sum += _hiddenGroups[j][k]._prevOutput * _memoryCells[i]._outputGate._synapses[si++]._weight;

		_memoryCells[i]._outputGate._output = sigmoid(sum);

		// Input
		sum = _memoryCells[i]._inputGate._bias._weight;

		si = 0;

		for (size_t j = 0; j < _memoryCells.size(); j++) {
			sum += _memoryCells[j]._outputGate._prevOutput * _memoryCells[i]._inputGate._synapses[si++]._weight;
			sum += _memoryCells[j]._inputGate._prevOutput * _memoryCells[i]._inputGate._synapses[si++]._weight;
			sum += _memoryCells[j]._forgetGate._prevOutput * _memoryCells[i]._inputGate._synapses[si++]._weight;

			sum += _memoryCells[j]._prevOutput * _memoryCells[i]._inputGate._synapses[si++]._weight;
		}

		for (size_t j = 0; j < _currentInputs.size(); j++)
			sum += _currentInputs[j] * _memoryCells[i]._inputGate._synapses[si++]._weight;

		for (size_t j = 0; j < _hiddenGroups.size(); j++)
		for (size_t k = 0; k < _hiddenGroups[j].size(); k++)
			sum += _hiddenGroups[j][k]._prevOutput * _memoryCells[i]._inputGate._synapses[si++]._weight;

		_memoryCells[i]._inputGate._output = sigmoid(sum);

		// Forget
		sum = _memoryCells[i]._forgetGate._bias._weight;

		si = 0;

		for (size_t j = 0; j < _memoryCells.size(); j++) {
			sum += _memoryCells[j]._outputGate._prevOutput * _memoryCells[i]._forgetGate._synapses[si++]._weight;
			sum += _memoryCells[j]._inputGate._prevOutput * _memoryCells[i]._forgetGate._synapses[si++]._weight;
			sum += _memoryCells[j]._forgetGate._prevOutput * _memoryCells[i]._forgetGate._synapses[si++]._weight;

			sum += _memoryCells[j]._prevOutput * _memoryCells[i]._forgetGate._synapses[si++]._weight;
		}

		for (size_t j = 0; j < _currentInputs.size(); j++)
			sum += _currentInputs[j] * _memoryCells[i]._forgetGate._synapses[si++]._weight;

		for (size_t j = 0; j < _hiddenGroups.size(); j++)
		for (size_t k = 0; k < _hiddenGroups[j].size(); k++)
			sum += _hiddenGroups[j][k]._prevOutput * _memoryCells[i]._forgetGate._synapses[si++]._weight;

		_memoryCells[i]._forgetGate._output = sigmoid(sum);

		// ----------------------------- Update CEC -----------------------------

		sum = _memoryCells[i]._bias._weight;

		si = 0;

		for (size_t j = 0; j < _memoryCells.size(); j++) {
			sum += _memoryCells[j]._outputGate._prevOutput * _memoryCells[i]._synapses[si++]._weight;
			sum += _memoryCells[j]._inputGate._prevOutput * _memoryCells[i]._synapses[si++]._weight;
			sum += _memoryCells[j]._forgetGate._prevOutput * _memoryCells[i]._synapses[si++]._weight;

			sum += _memoryCells[j]._prevOutput * _memoryCells[i]._synapses[si++]._weight;
		}

		for (size_t j = 0; j < _currentInputs.size(); j++)
			sum += _currentInputs[j] * _memoryCells[i]._synapses[si++]._weight;

		for (size_t j = 0; j < _hiddenGroups.size(); j++)
		for (size_t k = 0; k < _hiddenGroups[j].size(); k++)
			sum += _hiddenGroups[j][k]._prevOutput * _memoryCells[i]._synapses[si++]._weight;

		float memoryCellInput = sigmoid(sum) * 4.0f - 2.0f;

		_memoryCells[i]._net = memoryCellInput;

		_memoryCells[i]._state = _memoryCells[i]._forgetGate._output * _memoryCells[i]._state + _memoryCells[i]._inputGate._output * memoryCellInput;

		_memoryCells[i]._output = (sigmoid(_memoryCells[i]._state) * 2.0f - 1.0f) * _memoryCells[i]._outputGate._output;
	}

	// Update output units
	for (size_t i = 0; i < _outputNodes.size(); i++) {
		float sum = _outputNodes[i]._bias._weight;

		size_t si = 0;

		for (size_t j = 0; j < _hiddenGroups[i].size(); j++)
			sum += _hiddenGroups[i][j]._prevOutput * _outputNodes[i]._synapses[si++]._weight;

		for (size_t j = 0; j < _memoryCells.size(); j++)
			sum += _memoryCells[j]._prevOutput * _outputNodes[i]._synapses[si++]._weight;

		_outputNodes[i]._output = sum;
	}

	// ------------------------------------------ Offset Outputs ------------------------------------------

	std::vector<float> offsets(_outputNodes.size());

	std::normal_distribution<float> distOutputOffset(0.0f, offsetStdDev);

	for (size_t i = 0; i < _outputNodes.size(); i++) {
		float original = _outputNodes[i]._output;
		float delta = distOutputOffset(generator);

		_outputNodes[i]._output = std::min(1.0f, std::max(-1.0f, _outputNodes[i]._output + delta));

		offsets[i] = _outputNodes[i]._output - original;
	}

	// ------------------------------------------ Update weights ------------------------------------------

	size_t si;

	// Update output for selected action output unit
	for (size_t i = 0; i < _outputNodes.size(); i++) {
		float z = offsets[i];

		_outputNodes[i]._bias._weight += error * _outputNodes[i]._bias._trace;
		_outputNodes[i]._bias._trace = gammaLambda * _outputNodes[i]._bias._trace + z;

		si = 0;

		for (size_t j = 0; j < _hiddenGroups[i].size(); j++) {
			_outputNodes[i]._synapses[si]._weight += error * _outputNodes[i]._synapses[si]._trace;
			_outputNodes[i]._synapses[si]._trace = gammaLambda * _outputNodes[i]._synapses[si]._trace + z * _hiddenGroups[i][j]._output;

			si++;
		}

		for (size_t j = 0; j < _memoryCells.size(); j++) {
			_outputNodes[i]._synapses[si]._weight += error * _outputNodes[i]._synapses[si]._trace;
			_outputNodes[i]._synapses[si]._trace = gammaLambda * _outputNodes[i]._synapses[si]._trace + z * _memoryCells[j]._output;

			si++;
		}
	}

	// Update hidden units
	for (size_t i = 0; i < _hiddenGroups.size(); i++)
	for (size_t j = 0; j < _hiddenGroups[i].size(); j++) {
		float cTerm = offsets[i] * _outputNodes[i]._synapses[j]._weight * _hiddenGroups[i][j]._output * (1.0f - _hiddenGroups[i][j]._output);

		_hiddenGroups[i][j]._bias._weight += error * _hiddenGroups[i][j]._bias._trace;
		_hiddenGroups[i][j]._bias._trace = gammaLambda * _hiddenGroups[i][j]._bias._trace + cTerm;

		si = 0;

		for (size_t k = 0; k < _currentInputs.size(); k++) {
			_hiddenGroups[i][j]._synapses[si]._weight += error * _hiddenGroups[i][j]._synapses[si]._trace;
			_hiddenGroups[i][j]._synapses[si]._trace = gammaLambda * _hiddenGroups[i][j]._synapses[si]._trace + cTerm * _currentInputs[k];

			si++;
		}

		for (size_t k = 0; k < _memoryCells.size(); k++) {
			_hiddenGroups[i][j]._synapses[si]._weight += error * _hiddenGroups[i][j]._synapses[si]._trace;
			_hiddenGroups[i][j]._synapses[si]._trace = gammaLambda * _hiddenGroups[i][j]._synapses[si]._trace + cTerm * _memoryCells[k]._output;

			si++;
		}
	}

	// Update memory cells
	for (size_t i = 0; i < _memoryCells.size(); i++) {
		// ----------------------------- Update Gates -----------------------------

		const float sigmoidNet = sigmoid(_memoryCells[i]._state);
		const float hNet = sigmoidNet * 2.0f - 1.0f;
		const float hPrimeNet = 2.0f * sigmoidNet * (1.0f - sigmoidNet);

		float wKc = 0.0f;

		for (size_t j = 0; j < _outputNodes.size(); j++)
			wKc += _outputNodes[j]._synapses[_hiddenGroups[j].size() + i]._weight;

		float sum = 0.0f;

		for (size_t j = 0; j < _outputNodes.size(); j++)
		for (size_t k = 0; k < _hiddenGroups[j].size(); k++)
			sum += offsets[j] * _outputNodes[j]._synapses[k]._weight * _hiddenGroups[j][k]._synapses[_currentInputs.size() + i]._weight * _hiddenGroups[j][k]._output * (1.0f - _hiddenGroups[j][k]._output);

		float coeff;

		// Output
		coeff = hNet * (wKc + sum) * _memoryCells[i]._outputGate._output * (1.0f - _memoryCells[i]._outputGate._output);

		_memoryCells[i]._outputGate._bias._weight += error * _memoryCells[i]._outputGate._bias._trace;
		_memoryCells[i]._outputGate._bias._trace = gammaLambda * _memoryCells[i]._outputGate._bias._trace + coeff;

		si = 0;

		for (size_t j = 0; j < _memoryCells.size(); j++) {
			_memoryCells[i]._outputGate._synapses[si]._weight += error * _memoryCells[i]._outputGate._synapses[si]._trace;
			_memoryCells[i]._outputGate._synapses[si]._trace = gammaLambda * _memoryCells[i]._outputGate._synapses[si]._trace + coeff * _memoryCells[j]._output;

			si++;
		}

		for (size_t j = 0; j < _currentInputs.size(); j++) {
			_memoryCells[i]._outputGate._synapses[si]._weight += error * _memoryCells[i]._outputGate._synapses[si]._trace;
			_memoryCells[i]._outputGate._synapses[si]._trace = gammaLambda * _memoryCells[i]._outputGate._synapses[si]._trace + coeff * _currentInputs[j];

			si++;
		}

		for (size_t j = 0; j < _hiddenGroups.size(); j++)
		for (size_t k = 0; k < _hiddenGroups[j].size(); k++) {
			_memoryCells[i]._outputGate._synapses[si]._weight += error * _memoryCells[i]._outputGate._synapses[si]._trace;
			_memoryCells[i]._outputGate._synapses[si]._trace = gammaLambda * _memoryCells[i]._outputGate._synapses[si]._trace + coeff * _hiddenGroups[j][k]._output;

			si++;
		}

		// CEC
		coeff = (wKc + sum) * _memoryCells[i]._outputGate._output * hPrimeNet;

		const float sigmoidNetCEC = sigmoid(_memoryCells[i]._net);
		const float scaledSigmoidNetCEC = 4.0f * sigmoidNetCEC;
		const float gPrimeTimesInputGate = scaledSigmoidNetCEC * (1.0f - sigmoidNetCEC) * _memoryCells[i]._inputGate._output;

		_memoryCells[i]._bias._weight += error * _memoryCells[i]._synapses[si]._trace;
		_memoryCells[i]._bias._trace = gammaLambda * _memoryCells[i]._synapses[si]._trace + coeff * _memoryCells[i]._synapses[si]._derivative;
		_memoryCells[i]._bias._derivative = _memoryCells[i]._synapses[si]._derivative * _memoryCells[i]._forgetGate._output + gPrimeTimesInputGate;

		si = 0;

		for (size_t j = 0; j < _memoryCells.size(); j++) {
			_memoryCells[i]._synapses[si]._weight += error * _memoryCells[i]._synapses[si]._trace;
			_memoryCells[i]._synapses[si]._trace = gammaLambda * _memoryCells[i]._synapses[si]._trace + coeff * _memoryCells[i]._synapses[si]._derivative;
			_memoryCells[i]._synapses[si]._derivative = _memoryCells[i]._synapses[si]._derivative * _memoryCells[i]._forgetGate._output + gPrimeTimesInputGate * _memoryCells[j]._output;

			si++;
		}

		for (size_t j = 0; j < _currentInputs.size(); j++) {
			_memoryCells[i]._synapses[si]._weight += error * _memoryCells[i]._synapses[si]._trace;
			_memoryCells[i]._synapses[si]._trace = gammaLambda * _memoryCells[i]._synapses[si]._trace + coeff * _memoryCells[i]._synapses[si]._derivative;
			_memoryCells[i]._synapses[si]._derivative = _memoryCells[i]._synapses[si]._derivative * _memoryCells[i]._forgetGate._output + gPrimeTimesInputGate * _currentInputs[j];

			si++;
		}

		for (size_t j = 0; j < _hiddenGroups.size(); j++)
		for (size_t k = 0; k < _hiddenGroups[j].size(); k++) {
			_memoryCells[i]._synapses[si]._weight += error * _memoryCells[i]._synapses[si]._trace;
			_memoryCells[i]._synapses[si]._trace = gammaLambda * _memoryCells[i]._synapses[si]._trace + coeff * _memoryCells[i]._synapses[si]._derivative;
			_memoryCells[i]._synapses[si]._derivative = _memoryCells[i]._synapses[si]._derivative * _memoryCells[i]._forgetGate._output + gPrimeTimesInputGate * _hiddenGroups[j][k]._output;

			si++;
		}

		// Input
		// coeff is unchanged from previous, since it is the same equation. Therefore the following line has been commented out
		//coeff = (wKc + sum) * _memoryCells[i]._outputGate._output * hPrimeNet;

		const float gTimesInputPrime = scaledSigmoidNetCEC * _memoryCells[i]._inputGate._output * (1.0f - _memoryCells[i]._inputGate._output);

		_memoryCells[i]._inputGate._bias._weight += error * _memoryCells[i]._inputGate._synapses[si]._trace;
		_memoryCells[i]._inputGate._bias._trace = gammaLambda * _memoryCells[i]._inputGate._synapses[si]._trace + coeff * _memoryCells[i]._synapses[si]._derivative;
		_memoryCells[i]._inputGate._bias._derivative = _memoryCells[i]._synapses[si]._derivative * _memoryCells[i]._forgetGate._output + gTimesInputPrime;

		si = 0;

		for (size_t j = 0; j < _memoryCells.size(); j++) {
			_memoryCells[i]._inputGate._synapses[si]._weight += error * _memoryCells[i]._inputGate._synapses[si]._trace;
			_memoryCells[i]._inputGate._synapses[si]._trace = gammaLambda * _memoryCells[i]._inputGate._synapses[si]._trace + coeff * _memoryCells[i]._synapses[si]._derivative;
			_memoryCells[i]._inputGate._synapses[si]._derivative = _memoryCells[i]._synapses[si]._derivative * _memoryCells[i]._forgetGate._output + gTimesInputPrime * _memoryCells[j]._output;

			si++;
		}

		for (size_t j = 0; j < _currentInputs.size(); j++) {
			_memoryCells[i]._inputGate._synapses[si]._weight += error * _memoryCells[i]._inputGate._synapses[si]._trace;
			_memoryCells[i]._inputGate._synapses[si]._trace = gammaLambda * _memoryCells[i]._inputGate._synapses[si]._trace + coeff * _memoryCells[i]._synapses[si]._derivative;
			_memoryCells[i]._inputGate._synapses[si]._derivative = _memoryCells[i]._synapses[si]._derivative * _memoryCells[i]._forgetGate._output + gTimesInputPrime * _currentInputs[j];

			si++;
		}

		for (size_t j = 0; j < _hiddenGroups.size(); j++)
		for (size_t k = 0; k < _hiddenGroups[j].size(); k++) {
			_memoryCells[i]._inputGate._synapses[si]._weight += error * _memoryCells[i]._inputGate._synapses[si]._trace;
			_memoryCells[i]._inputGate._synapses[si]._trace = gammaLambda * _memoryCells[i]._inputGate._synapses[si]._trace + coeff * _memoryCells[i]._synapses[si]._derivative;
			_memoryCells[i]._inputGate._synapses[si]._derivative = _memoryCells[i]._synapses[si]._derivative * _memoryCells[i]._forgetGate._output + gTimesInputPrime * _hiddenGroups[j][k]._output;

			si++;
		}

		// Forget
		// coeff is unchanged from previous, since it is the same equation. Therefore the following line has been commented out
		//coeff = (wKc + sum) * _memoryCells[i]._outputGate._output * hPrimeNet;

		const float stateTimesForgetPrime = _memoryCells[i]._state * _memoryCells[i]._forgetGate._output * (1.0f - _memoryCells[i]._forgetGate._output);

		_memoryCells[i]._forgetGate._bias._weight += error * _memoryCells[i]._inputGate._synapses[si]._trace;
		_memoryCells[i]._forgetGate._bias._trace = gammaLambda * _memoryCells[i]._inputGate._synapses[si]._trace + coeff * _memoryCells[i]._synapses[si]._derivative;
		_memoryCells[i]._forgetGate._bias._derivative = _memoryCells[i]._synapses[si]._derivative * _memoryCells[i]._forgetGate._output + gTimesInputPrime;

		si = 0;

		for (size_t j = 0; j < _memoryCells.size(); j++) {	
			_memoryCells[i]._forgetGate._synapses[si]._weight += error * _memoryCells[i]._forgetGate._synapses[si]._trace;
			_memoryCells[i]._forgetGate._synapses[si]._trace = gammaLambda * _memoryCells[i]._forgetGate._synapses[si]._trace + coeff * _memoryCells[i]._synapses[si]._derivative;
			_memoryCells[i]._forgetGate._synapses[si]._derivative = _memoryCells[i]._synapses[si]._derivative * _memoryCells[i]._forgetGate._output + stateTimesForgetPrime * _memoryCells[j]._output;

			si++;
		}

		for (size_t j = 0; j < _currentInputs.size(); j++) {
			_memoryCells[i]._forgetGate._synapses[si]._weight += error * _memoryCells[i]._forgetGate._synapses[si]._trace;
			_memoryCells[i]._forgetGate._synapses[si]._trace = gammaLambda * _memoryCells[i]._forgetGate._synapses[si]._trace + coeff * _memoryCells[i]._synapses[si]._derivative;
			_memoryCells[i]._forgetGate._synapses[si]._derivative = _memoryCells[i]._synapses[si]._derivative * _memoryCells[i]._forgetGate._output + stateTimesForgetPrime * _currentInputs[j];

			si++;
		}

		for (size_t j = 0; j < _hiddenGroups.size(); j++)
		for (size_t k = 0; k < _hiddenGroups[j].size(); k++) {
			_memoryCells[i]._forgetGate._synapses[si]._weight += error * _memoryCells[i]._forgetGate._synapses[si]._trace;
			_memoryCells[i]._forgetGate._synapses[si]._trace = gammaLambda * _memoryCells[i]._forgetGate._synapses[si]._trace + coeff * _memoryCells[i]._synapses[si]._derivative;
			_memoryCells[i]._forgetGate._synapses[si]._derivative = _memoryCells[i]._synapses[si]._derivative * _memoryCells[i]._forgetGate._output + stateTimesForgetPrime * _hiddenGroups[j][k]._output;

			si++;
		}
	}

	// --------------------------------- Set Previous Outputs to Current ---------------------------------

	for (size_t i = 0; i < _hiddenGroups.size(); i++)
	for (size_t j = 0; j < _hiddenGroups[i].size(); j++)
		_hiddenGroups[i][j]._prevOutput = _hiddenGroups[i][j]._output;

	for (size_t i = 0; i < _memoryCells.size(); i++) {
		_memoryCells[i]._outputGate._prevOutput = _memoryCells[i]._outputGate._output;
		_memoryCells[i]._forgetGate._prevOutput = _memoryCells[i]._forgetGate._output;
		_memoryCells[i]._inputGate._prevOutput = _memoryCells[i]._inputGate._output;

		_memoryCells[i]._prevOutput = _memoryCells[i]._output;
	}

	for (size_t i = 0; i < _outputNodes.size(); i++)
		_outputNodes[i]._prevOutput = _outputNodes[i]._output;
}