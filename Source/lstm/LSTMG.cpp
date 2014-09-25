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

#include <lstm/LSTMG.h>

#include <algorithm>

#include <iostream>

#include <assert.h>

using namespace lstm;

void LSTMG::createRandomLayered(int numInputs, int numOutputs,
	int numMemoryLayers, int memoryLayerSize, int numHiddenLayers, int hiddenLayerSize,
	float minWeight, float maxWeight, std::mt19937 &randomGenerator)
{
	std::uniform_real_distribution<float> weightDist(minWeight, maxWeight);

	// First numInputs units are input units
	_inputIndices.resize(numInputs);

	for (int i = 0; i < numInputs; i++)
		_inputIndices[i] = i;

	_units.resize(numInputs + numMemoryLayers * memoryLayerSize * 4 + numHiddenLayers * hiddenLayerSize + numOutputs);

	int inputIndex = 0;

	// -------------------------------- Input Layer ---------------------------------

	inputIndex = numInputs;

	// ----------------------------- First memory layer -----------------------------

	std::vector<int> outputGatersStarts;
	std::vector<int> memoryUnitsStarts;

	int inputGatersStart;
	int forgetGatersStart;
	int memoryUnitsStart;
	int outputGatersStart;

	if (numMemoryLayers > 0) {
		inputGatersStart = inputIndex;

		// Create input gaters
		for (int ui = 0; ui < memoryLayerSize; ui++) {
			Unit &inputGate = _units[inputIndex];

			inputGate._bias = inputGate._prevBias = weightDist(randomGenerator);

			// Input connections
			for (int i = 0; i < numInputs; i++) {
				Connection c;

				c._weight = c._prevWeight = weightDist(randomGenerator);
				c._inputIndex = i;

				inputGate._ingoingConnections.push_back(c);
			}

			inputIndex++;
		}

		forgetGatersStart = inputIndex;

		// Create forget gaters
		for (int ui = 0; ui < memoryLayerSize; ui++) {
			Unit &forgetGate = _units[inputIndex];

			forgetGate._bias = forgetGate._prevBias = weightDist(randomGenerator);

			// Input connections
			for (int i = 0; i < numInputs; i++) {
				Connection c;

				c._weight = c._prevWeight = weightDist(randomGenerator);
				c._inputIndex = i;

				forgetGate._ingoingConnections.push_back(c);
			}

			inputIndex++;
		}

		memoryUnitsStart = inputIndex;

		// Create memory units
		for (int ui = 0; ui < memoryLayerSize; ui++) {
			Unit &memoryUnit = _units[inputIndex];

			memoryUnit._bias = memoryUnit._prevBias = weightDist(randomGenerator);

			// Input connections
			for (int i = 0; i < numInputs; i++) {
				Connection c;

				c._gaterIndex = inputGatersStart + i;
				_units[c._gaterIndex]._gatingConnections.push_back(inputIndex);

				c._weight = c._prevWeight = weightDist(randomGenerator);
				c._inputIndex = i;

				memoryUnit._ingoingConnections.push_back(c);
			}

			// Recurrent connection
			Connection c;

			c._gaterIndex = forgetGatersStart + inputIndex - memoryUnitsStart;
			_units[c._gaterIndex]._gatingConnections.push_back(inputIndex);

			c._weight = c._prevWeight = weightDist(randomGenerator);
			c._inputIndex = inputIndex;

			memoryUnit._ingoingConnections.push_back(c);

			memoryUnit._recurrentConnectionIndex = memoryUnit._ingoingConnections.size() - 1;

			inputIndex++;
		}

		outputGatersStart = inputIndex;

		// Create output gaters
		for (int ui = 0; ui < memoryLayerSize; ui++) {
			Unit &outputGater = _units[inputIndex];

			_orderedGaterIndices.push_back(inputIndex);

			outputGater._bias = outputGater._prevBias = weightDist(randomGenerator);

			// Input connections
			for (int i = 0; i < numInputs; i++) {
				Connection c;

				c._weight = c._prevWeight = weightDist(randomGenerator);
				c._inputIndex = i;

				outputGater._ingoingConnections.push_back(c);
			}

			inputIndex++;
		}

		// Add connections to input gaters (not fully connected)
		for (int i = 0; i < memoryLayerSize; i++) {
			Connection c;

			c._weight = c._prevWeight = weightDist(randomGenerator);
			c._inputIndex = i + memoryUnitsStart;

			_units[i + inputGatersStart]._ingoingConnections.push_back(c);
		}

		// Add connections to forget gaters (not fully connected)
		for (int i = 0; i < memoryLayerSize; i++) {
			Connection c;

			c._weight = c._prevWeight = weightDist(randomGenerator);
			c._inputIndex = i + memoryUnitsStart;

			_units[i + forgetGatersStart]._ingoingConnections.push_back(c);
		}

		// Add connections to output gaters (not fully connected)
		for (int i = 0; i < memoryLayerSize; i++) {
			Connection c;

			c._weight = c._prevWeight = weightDist(randomGenerator);
			c._inputIndex = i + memoryUnitsStart;

			_units[i + outputGatersStart]._ingoingConnections.push_back(c);
		}

		outputGatersStarts.push_back(outputGatersStart);
		memoryUnitsStarts.push_back(memoryUnitsStart);
	}

	// --------------------------- All other memory layers --------------------------

	for (int ml = 1; ml < numMemoryLayers; ml++) {
		int prevInputGatersStart = inputGatersStart;
		int prevForgetGatersStart = forgetGatersStart;
		int prevMemoryUnitsStart = memoryUnitsStart;
		int prevOutputGatersStart = outputGatersStart;

		inputGatersStart = inputIndex;

		// Create input gaters
		for (int ui = 0; ui < memoryLayerSize; ui++) {
			Unit &inputGate = _units[inputIndex];

			inputGate._bias = inputGate._prevBias = weightDist(randomGenerator);

			// Input connections
			for (int i = 0; i < numInputs; i++) {
				Connection c;

				c._weight = c._prevWeight = weightDist(randomGenerator);
				c._inputIndex = i;

				inputGate._ingoingConnections.push_back(c);
			}

			// Previous memory cell connections
			for (int i = 0; i < memoryLayerSize; i++) {
				Connection c;

				c._weight = c._prevWeight = weightDist(randomGenerator);
				c._inputIndex = prevMemoryUnitsStart + i;

				inputGate._ingoingConnections.push_back(c);
			}

			inputIndex++;
		}

		forgetGatersStart = inputIndex;

		// Create forget gaters
		for (int ui = 0; ui < memoryLayerSize; ui++) {
			Unit &forgetGate = _units[inputIndex];

			forgetGate._bias = forgetGate._prevBias = weightDist(randomGenerator);

			// Input connections
			for (int i = 0; i < numInputs; i++) {
				Connection c;

				c._weight = c._prevWeight = weightDist(randomGenerator);
				c._inputIndex = i;

				forgetGate._ingoingConnections.push_back(c);
			}

			// Previous memory cell connections
			for (int i = 0; i < memoryLayerSize; i++) {
				Connection c;

				c._weight = c._prevWeight = weightDist(randomGenerator);
				c._inputIndex = prevMemoryUnitsStart + i;

				forgetGate._ingoingConnections.push_back(c);
			}

			inputIndex++;
		}

		memoryUnitsStart = inputIndex;

		// Create memory units
		for (int ui = 0; ui < memoryLayerSize; ui++) {
			Unit &memoryUnit = _units[inputIndex];

			memoryUnit._bias = memoryUnit._prevBias = weightDist(randomGenerator);

			// Input connections
			for (int i = 0; i < numInputs; i++) {
				Connection c;

				c._gaterIndex = inputGatersStart + i;
				_units[c._gaterIndex]._gatingConnections.push_back(inputIndex);

				c._weight = c._prevWeight = weightDist(randomGenerator);
				c._inputIndex = i;

				memoryUnit._ingoingConnections.push_back(c);
			}

			// Recurrent connection
			{
				Connection c;

				c._gaterIndex = forgetGatersStart + inputIndex - memoryUnitsStart;
				_units[c._gaterIndex]._gatingConnections.push_back(inputIndex);

				c._weight = c._prevWeight = weightDist(randomGenerator);
				c._inputIndex = inputIndex;

				memoryUnit._ingoingConnections.push_back(c);

				memoryUnit._recurrentConnectionIndex = memoryUnit._ingoingConnections.size() - 1;
			}

			// Previous memory cell connections
			for (int i = 0; i < memoryLayerSize; i++) {
				Connection c;

				c._gaterIndex = inputGatersStart + i;
				_units[c._gaterIndex]._gatingConnections.push_back(inputIndex);

				c._weight = c._prevWeight = weightDist(randomGenerator);
				c._inputIndex = prevMemoryUnitsStart + i;

				memoryUnit._ingoingConnections.push_back(c);
			}

			inputIndex++;
		}

		outputGatersStart = inputIndex;

		// Create output gaters
		for (int ui = 0; ui < memoryLayerSize; ui++) {
			Unit &outputGater = _units[inputIndex];

			outputGater._bias = outputGater._prevBias = weightDist(randomGenerator);

			// Input connections
			for (int i = 0; i < numInputs; i++) {
				Connection c;

				c._weight = c._prevWeight = weightDist(randomGenerator);
				c._inputIndex = i;

				outputGater._ingoingConnections.push_back(c);
			}

			// Previous memory cell connections
			for (int i = 0; i < memoryLayerSize; i++) {
				Connection c;

				c._weight = c._prevWeight = weightDist(randomGenerator);
				c._inputIndex = prevMemoryUnitsStart + i;

				outputGater._ingoingConnections.push_back(c);
			}

			inputIndex++;
		}

		// Add connections to input gaters (not fully connected)
		for (int i = 0; i < memoryLayerSize; i++) {
			Connection c;

			c._weight = c._prevWeight = weightDist(randomGenerator);
			c._inputIndex = i + memoryUnitsStart;

			_units[i + inputGatersStart]._ingoingConnections.push_back(c);
		}

		// Add connections to forget gaters (not fully connected)
		for (int i = 0; i < memoryLayerSize; i++) {
			Connection c;

			c._weight = c._prevWeight = weightDist(randomGenerator);
			c._inputIndex = i + memoryUnitsStart;

			_units[i + forgetGatersStart]._ingoingConnections.push_back(c);
		}

		// Add connections to output gaters (not fully connected)
		for (int i = 0; i < memoryLayerSize; i++) {
			Connection c;

			c._weight = c._prevWeight = weightDist(randomGenerator);
			c._inputIndex = i + memoryUnitsStart;

			_units[i + outputGatersStart]._ingoingConnections.push_back(c);
		}

		outputGatersStarts.push_back(outputGatersStart);
		memoryUnitsStarts.push_back(memoryUnitsStart);
	}

	if (numHiddenLayers > 0) {

		// --------------------------- First hidden layer ---------------------------

		int hiddenUnitsStart = inputIndex;

		for (int ui = 0; ui < hiddenLayerSize; ui++) {
			Unit &hiddenUnit = _units[inputIndex];

			hiddenUnit._bias = hiddenUnit._prevBias = weightDist(randomGenerator);

			// Input connections
			for (int i = 0; i < numInputs; i++) {
				Connection c;

				c._weight = c._prevWeight = weightDist(randomGenerator);
				c._inputIndex = i;

				hiddenUnit._ingoingConnections.push_back(c);
			}

			// Memory cell (gated) connections
			for (int ogl = 0; ogl < outputGatersStarts.size(); ogl++) {
				for (int i = 0; i < memoryLayerSize; i++) {
					Connection c;

					c._gaterIndex = outputGatersStarts[ogl] + i;
					_units[c._gaterIndex]._gatingConnections.push_back(inputIndex);

					c._weight = c._prevWeight = weightDist(randomGenerator);
					c._inputIndex = memoryUnitsStarts[ogl] + i;

					hiddenUnit._ingoingConnections.push_back(c);
				}
			}

			inputIndex++;
		}

		// ------------------------- All other hidden layers -------------------------

		for (int hl = 1; hl < numHiddenLayers; hl++) {
			int prevHiddenUnitsStart = hiddenUnitsStart;

			hiddenUnitsStart = inputIndex;

			for (int ui = 0; ui < hiddenLayerSize; ui++) {
				Unit &hiddenUnit = _units[inputIndex];

				hiddenUnit._bias = hiddenUnit._prevBias = weightDist(randomGenerator);

				// Previous hidden connections
				for (int i = 0; i < hiddenLayerSize; i++) {
					Connection c;

					c._weight = c._prevWeight = weightDist(randomGenerator);
					c._inputIndex = prevHiddenUnitsStart + i;

					hiddenUnit._ingoingConnections.push_back(c);
				}

				inputIndex++;
			}
		}

		// --------------------------- Output layer ---------------------------

		{
			int prevHiddenUnitsStart = hiddenUnitsStart;

			hiddenUnitsStart = inputIndex;

			for (int ui = 0; ui < numOutputs; ui++) {
				_outputIndices.push_back(inputIndex);

				Unit &outputUnit = _units[inputIndex];

				outputUnit._bias = outputUnit._prevBias = weightDist(randomGenerator);

				// Previous hidden connections
				for (int i = 0; i < hiddenLayerSize; i++) {
					Connection c;

					c._weight = c._prevWeight = weightDist(randomGenerator);
					c._inputIndex = prevHiddenUnitsStart + i;

					outputUnit._ingoingConnections.push_back(c);
				}

				inputIndex++;
			}
		}
	}
	else {

		// --------------------------- Output layer ---------------------------

		for (int ui = 0; ui < numOutputs; ui++) {
			_outputIndices.push_back(inputIndex);

			Unit &outputUnit = _units[inputIndex];

			outputUnit._bias = outputUnit._prevBias = weightDist(randomGenerator);

			// Input connections
			for (int i = 0; i < numInputs; i++) {
				Connection c;

				c._weight = c._prevWeight = weightDist(randomGenerator);
				c._inputIndex = i;

				outputUnit._ingoingConnections.push_back(c);
			}

			// Memory cell (gated) connections
			for (int ogl = 0; ogl < outputGatersStarts.size(); ogl++) {
				for (int i = 0; i < memoryLayerSize; i++) {
					Connection c;

					c._gaterIndex = outputGatersStarts[ogl] + i;
					_units[c._gaterIndex]._gatingConnections.push_back(inputIndex);

					c._weight = c._prevWeight = weightDist(randomGenerator);
					c._inputIndex = memoryUnitsStarts[ogl] + i;

					outputUnit._ingoingConnections.push_back(c);
				}
			}

			inputIndex++;
		}
	}

	// Build gaters maps as well as ingoing and outgoing connection arrays
	for (int j = 0; j < _units.size(); j++)
	for (int ci = 0; ci < _units[j]._ingoingConnections.size(); ci++) {
		Connection &c = _units[j]._ingoingConnections[ci];

		_units[c._inputIndex]._outgoingConnectionIndices.push_back(ConnectionIndex(j, ci));
	}

	// Build sorted list of gater indices
	std::sort(_orderedGaterIndices.begin(), _orderedGaterIndices.end());

	clear();
}

bool LSTMG::connectionExists(int j, int i) {
	for (int ii = 0; ii < _units[j]._ingoingConnections.size(); ii++)
	if (_units[j]._ingoingConnections[ii]._inputIndex == i)
		return true;

	return false;
}

void LSTMG::clear() {
	for (int j = 0; j < _units.size(); j++) {
		_units[j]._state = _units[j]._activation = 0.0f;

		for (int ci = 0; ci < _units[j]._ingoingConnections.size(); ci++) {
			_units[j]._ingoingConnections[ci]._trace = 0.0f;

			_units[j]._ingoingConnections[ci]._extendedTraces.clear();

			if (_units[j]._ingoingConnections[ci]._inputIndex != j && !_units[j]._gatingConnections.empty()) {
				for (int oi = 0; oi < _units[j]._outgoingConnectionIndices.size(); oi++)
					_units[j]._ingoingConnections[ci]._extendedTraces.push_back(ExtendedTrace(0.0f, _units[j]._outgoingConnectionIndices[oi]._nodeIndex));
			}
		}
	}
}

float LSTMG::gain(int j, int ci) {
	if (ci == -1) // Unused
		return 0.0f;

	int gaterIndex = _units[j]._ingoingConnections[ci]._gaterIndex;

	if (gaterIndex != -1)
		return _units[gaterIndex]._activation;

	if (j == _units[j]._ingoingConnections[ci]._inputIndex)
		return 0.0f;

	assert(connectionExists(j, _units[j]._ingoingConnections[ci]._inputIndex));

	return 1.0f;
}

float LSTMG::theTerm(int j, int k) {
	float term = 0.0f;

	if (_units[k]._ingoingConnections.empty())
		return 0.0f;

	int gaterIndex;

	if (_units[k]._recurrentConnectionIndex == -1)
		return 0.0f;
	else
		gaterIndex = _units[k]._ingoingConnections[_units[k]._recurrentConnectionIndex]._gaterIndex;

	if (gaterIndex == j)
		term = _units[k]._prevState;

	for (int ii = 0; ii < _units[k]._ingoingConnections.size(); ii++)
		term += _units[k]._ingoingConnections[ii]._weight * _units[_units[k]._ingoingConnections[ii]._gaterIndex]._prevActivation;

	return term;
}

void LSTMG::step(bool linearOutput) {
	for (int j = 0; j < _units.size(); j++) {
		_units[j]._prevGain = 0.0f;
		_units[j]._prevActivation = 0.0f;
	}

	// Copy previous states
	for (int j = 0; j < _units.size(); j++)
		_units[j]._prevState = _units[j]._state;

	for (int j = _inputIndices.size(); j < _units.size(); j++) {
		float sg = _units[j]._prevGain = gain(j, -1);

		_units[j]._state *= sg;

		for (int ci = 0; ci < _units[j]._ingoingConnections.size(); ci++) {
			int i = _units[j]._ingoingConnections[ci]._inputIndex;

			float g = _units[j]._ingoingConnections[ci]._prevGain = gain(j, ci);

			float a = _units[j]._ingoingConnections[ci]._prevActivation = _units[i]._activation;

			Connection &c = _units[j]._ingoingConnections[ci];

			_units[j]._state += g * c._weight * _units[i]._activation;

			c._trace *= sg;
			c._trace += g * a;
		}

		// If not output unit
		if (!linearOutput || j < _units.size() - _outputIndices.size()) {
			// Add bias to state? Or keep out of state and use only for activation?
			_units[j]._state += _units[j]._bias;

			_units[j]._activation = sigmoid(_units[j]._state);
			//_units[j]._activation = sigmoid(_units[j]._state + _units[j]._bias);
		}
		else {
			_units[j]._state += _units[j]._bias;

			_units[j]._activation = _units[j]._state + _units[j]._bias;
		}
	}

	for (size_t j = 0; j < _units.size(); j++)
	for (size_t ci = 0; ci < _units[j]._ingoingConnections.size(); ci++)
	for (size_t ti = 0; ti < _units[j]._ingoingConnections[ci]._extendedTraces.size(); ti++) {
		float terms = sigmoidDerivative(_units[j]._prevState) * _units[j]._ingoingConnections[ci]._trace * theTerm(j, ti);
		_units[j]._ingoingConnections[ci]._extendedTraces[ti]._trace = _units[ti]._prevGain * _units[j]._ingoingConnections[ci]._extendedTraces[ti]._trace + terms;
	}
}

void LSTMG::getDeltas(const std::vector<float> &targets, float eligibilityDecay, bool linearOutput) {
	std::vector<float> errorResp(_units.size(), 0.0f);
	std::vector<float> errorProj(_units.size(), 0.0f);

	// Output layer errors
	for (int i = 0; i < _outputIndices.size(); i++)
		errorResp[_outputIndices[i]] = targets[i] - _units[_outputIndices[i]]._activation;

	// Loop over all units in reversed order of activation
	for (int j = _units.size() - _outputIndices.size() - 1; j >= _inputIndices.size(); j--) {
		errorResp[j] = errorProj[j] = 0.0f;

		for (int ci = 0; ci < _units[j]._outgoingConnectionIndices.size(); ci++) {
			int k = _units[j]._outgoingConnectionIndices[ci]._nodeIndex;

			errorProj[j] += errorResp[k] * _units[k]._ingoingConnections[_units[j]._outgoingConnectionIndices[ci]._connectionIndex]._prevGain * _units[k]._ingoingConnections[_units[j]._outgoingConnectionIndices[ci]._connectionIndex]._weight;
		}

		float jDeriv = sigmoidDerivative(_units[j]._state);

		errorProj[j] *= jDeriv;

		int lastK = 0;

		for (int gi = 0; gi < _orderedGaterIndices.size(); gi++) {
			if (j == _orderedGaterIndices[gi]) {
				// For each connection
				const std::vector<int> &gaterConnections = _units[j]._gatingConnections;

				for (int gci = 0; gci < gaterConnections.size(); gci++) {
					int k = gaterConnections[gci];

					if (lastK < k && j < k) {
						lastK = k;

						errorResp[j] += errorResp[k] * theTerm(j, k);
					}
				}
			}
		}

		errorResp[j] = errorProj[j] + jDeriv * errorResp[j];
	}

	for (int j = 0; j < _units.size(); j++)
	for (int ci = 0; ci < _units[j]._ingoingConnections.size(); ci++) {
		int i = _units[j]._ingoingConnections[ci]._inputIndex;

		_units[j]._ingoingConnections[ci]._eligibility *= eligibilityDecay;

		// If not output
		if (j < _units.size() - _outputIndices.size()) {
			_units[j]._ingoingConnections[ci]._eligibility += errorProj[j] * _units[j]._ingoingConnections[ci]._trace;

			for (int ti = 0; ti < _units[j]._ingoingConnections[ci]._extendedTraces.size(); ti++)
				_units[j]._ingoingConnections[ci]._eligibility += errorResp[_units[j]._ingoingConnections[ci]._extendedTraces[ti]._index] * _units[j]._ingoingConnections[ci]._extendedTraces[ti]._trace;
		}
		else
			_units[j]._ingoingConnections[ci]._eligibility += errorResp[j] * _units[j]._ingoingConnections[ci]._trace;
	}

	// Update biases
	for (int j = 0; j < _units.size(); j++) {
		_units[j]._biasEligibility *= eligibilityDecay;
		_units[j]._biasEligibility += errorResp[j];
	}
}

void LSTMG::moveAlongDeltas(float error, float momentum) {
	for (int j = 0; j < _units.size(); j++)
	for (int ci = 0; ci < _units[j]._ingoingConnections.size(); ci++) {
		float d = error * _units[j]._ingoingConnections[ci]._eligibility + momentum * _units[j]._ingoingConnections[ci]._prevWeight;
		_units[j]._ingoingConnections[ci]._weight += d;
		_units[j]._ingoingConnections[ci]._prevWeight = d;
	}

	// Update biases
	for (int j = 0; j < _units.size(); j++) {
		float d = error * _units[j]._biasEligibility + momentum * _units[j]._prevBias;
		_units[j]._bias += d;
		_units[j]._prevBias = d;
	}
}

void LSTMG::moveAlongDeltasAndHebbian(float error, float hebbianAlpha, float momentum) {
	for (int j = 0; j < _units.size(); j++) {
		float postTerm = _units[j]._activation;

		for (int ci = 0; ci < _units[j]._ingoingConnections.size(); ci++) {
			float preTerm = _units[_units[j]._ingoingConnections[ci]._inputIndex]._activation;
			float d = error * _units[j]._ingoingConnections[ci]._eligibility + momentum * _units[j]._ingoingConnections[ci]._prevWeight + hebbianAlpha * postTerm * (preTerm - _units[j]._ingoingConnections[ci]._weight * postTerm);
			_units[j]._ingoingConnections[ci]._weight += d;
			_units[j]._ingoingConnections[ci]._prevWeight = d;
		}
	}

	// Update biases
	for (int j = 0; j < _units.size(); j++) {
		float postTerm = _units[j]._activation;

		float d = error * _units[j]._biasEligibility + momentum * _units[j]._prevBias + hebbianAlpha * postTerm * (1.0f - _units[j]._bias * postTerm);
		_units[j]._bias += d;
		_units[j]._prevBias = d;
	}
}