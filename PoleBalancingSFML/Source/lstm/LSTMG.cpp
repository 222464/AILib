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

	// -------------------------------- Input Layer ---------------------------------

	for (int ui = 0; ui < numInputs; ui++) {
		Unit inputUnit;

		inputUnit._activation = 0.0f;
		inputUnit._state = 0.0f;
		inputUnit._prevState = 0.0f;
		inputUnit._bias = 0.0f;
		inputUnit._biasEligibility = 0.0f;
		inputUnit._prevBias = 0.0f;

		_units.push_back(inputUnit);
	}

	// ----------------------------- First memory layer -----------------------------

	std::vector<int> outputGatersStarts;
	std::vector<int> memoryUnitsStarts;

	int inputGatersStart;
	int forgetGatersStart;
	int memoryUnitsStart;
	int outputGatersStart;

	if (numMemoryLayers > 0) {
		inputGatersStart = _units.size();

		// Create input gaters
		for (int ui = 0; ui < memoryLayerSize; ui++) {
			Unit inputGate;

			_orderedGaterIndices.push_back(_units.size());

			inputGate._activation = 0.0f;
			inputGate._state = 0.0f;
			inputGate._prevState = 0.0f;
			inputGate._bias = weightDist(randomGenerator);
			inputGate._biasEligibility = 0.0f;
			inputGate._prevBias = 0.0f;

			// Input connections
			for (int i = 0; i < numInputs; i++) {
				Connection c;
				c._gaterIndex = -1;
				c._weight = weightDist(randomGenerator);
				c._trace = 0.0f;
				c._eligibility = 0.0f;
				c._prevWeight = 0.0f;

				_connections[std::make_tuple(static_cast<int>(_units.size()), i)] = c;
			}

			_units.push_back(inputGate);
		}

		forgetGatersStart = _units.size();

		// Create forget gaters
		for (int ui = 0; ui < memoryLayerSize; ui++) {
			Unit forgetGate;

			_orderedGaterIndices.push_back(_units.size());

			forgetGate._activation = 0.0f;
			forgetGate._state = 0.0f;
			forgetGate._prevState = 0.0f;
			forgetGate._bias = weightDist(randomGenerator);
			forgetGate._biasEligibility = 0.0f;
			forgetGate._prevBias = 0.0f;

			// Input connections
			for (int i = 0; i < numInputs; i++) {
				Connection c;
				c._gaterIndex = -1;
				c._weight = weightDist(randomGenerator);
				c._trace = 0.0f;
				c._eligibility = 0.0f;
				c._prevWeight = 0.0f;

				_connections[std::make_tuple(static_cast<int>(_units.size()), i)] = c;
			}

			_units.push_back(forgetGate);
		}

		memoryUnitsStart = _units.size();

		// Create memory units
		for (int ui = 0; ui < memoryLayerSize; ui++) {
			Unit memoryUnit;

			memoryUnit._activation = 0.0f;
			memoryUnit._state = 0.0f;
			memoryUnit._prevState = 0.0f;
			memoryUnit._bias = weightDist(randomGenerator);
			memoryUnit._biasEligibility = 0.0f;
			memoryUnit._prevBias = 0.0f;

			// Input connections
			for (int i = 0; i < numInputs; i++) {
				Connection c;
				c._gaterIndex = inputGatersStart + i;
				c._weight = weightDist(randomGenerator);
				c._trace = 0.0f;
				c._eligibility = 0.0f;
				c._prevWeight = 0.0f;

				_connections[std::make_tuple(static_cast<int>(_units.size()), i)] = c;
			}

			// Recurrent connection
			Connection c;
			c._gaterIndex = forgetGatersStart + _units.size() - memoryUnitsStart;
			c._weight = weightDist(randomGenerator);
			c._trace = 0.0f;
			c._eligibility = 0.0f;
			c._prevWeight = 0.0f;

			_connections[std::make_tuple(static_cast<int>(_units.size()), static_cast<int>(_units.size()))] = c;

			_units.push_back(memoryUnit);
		}

		outputGatersStart = _units.size();

		// Create output gaters
		for (int ui = 0; ui < memoryLayerSize; ui++) {
			Unit outputGater;

			_orderedGaterIndices.push_back(_units.size());

			outputGater._activation = 0.0f;
			outputGater._state = 0.0f;
			outputGater._prevState = 0.0f;
			outputGater._bias = weightDist(randomGenerator);
			outputGater._biasEligibility = 0.0f;
			outputGater._prevBias = 0.0f;

			// Input connections
			for (int i = 0; i < numInputs; i++) {
				Connection c;
				c._gaterIndex = -1;
				c._weight = weightDist(randomGenerator);
				c._trace = 0.0f;
				c._eligibility = 0.0f;
				c._prevWeight = 0.0f;

				_connections[std::make_tuple(static_cast<int>(_units.size()), i)] = c;
			}

			_units.push_back(outputGater);
		}

		// Add connections to input gaters (not fully connected)
		for (int i = 0; i < memoryLayerSize; i++) {
			Connection c;
			c._gaterIndex = -1;
			c._weight = weightDist(randomGenerator);
			c._trace = 0.0f;
			c._eligibility = 0.0f;
			c._prevWeight = 0.0f;

			_connections[std::make_tuple(i + inputGatersStart, i + memoryUnitsStart)] = c;
		}

		// Add connections to forget gaters (not fully connected)
		for (int i = 0; i < memoryLayerSize; i++) {
			Connection c;
			c._gaterIndex = -1;
			c._weight = weightDist(randomGenerator);
			c._trace = 0.0f;
			c._eligibility = 0.0f;
			c._prevWeight = 0.0f;

			_connections[std::make_tuple(i + forgetGatersStart, i + memoryUnitsStart)] = c;
		}

		// Add connections to output gaters (not fully connected)
		for (int i = 0; i < memoryLayerSize; i++) {
			Connection c;
			c._gaterIndex = -1;
			c._weight = weightDist(randomGenerator);
			c._trace = 0.0f;
			c._eligibility = 0.0f;
			c._prevWeight = 0.0f;

			_connections[std::make_tuple(i + outputGatersStart, i + memoryUnitsStart)] = c;
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

		inputGatersStart = _units.size();

		// Create input gaters
		for (int ui = 0; ui < memoryLayerSize; ui++) {
			Unit inputGate;

			_orderedGaterIndices.push_back(_units.size());

			inputGate._activation = 0.0f;
			inputGate._state = 0.0f;
			inputGate._prevState = 0.0f;
			inputGate._bias = weightDist(randomGenerator);
			inputGate._biasEligibility = 0.0f;
			inputGate._prevBias = 0.0f;

			// Input connections
			for (int i = 0; i < numInputs; i++) {
				Connection c;
				c._gaterIndex = -1;
				c._weight = weightDist(randomGenerator);
				c._trace = 0.0f;
				c._eligibility = 0.0f;
				c._prevWeight = 0.0f;

				_connections[std::make_tuple(static_cast<int>(_units.size()), i)] = c;
			}

			// Previous memory cell connections
			for (int i = 0; i < memoryLayerSize; i++) {
				Connection c;
				c._gaterIndex = -1;
				c._weight = weightDist(randomGenerator);
				c._trace = 0.0f;
				c._eligibility = 0.0f;
				c._prevWeight = 0.0f;

				_connections[std::make_tuple(static_cast<int>(_units.size()), prevMemoryUnitsStart + i)] = c;
			}

			_units.push_back(inputGate);
		}

		forgetGatersStart = _units.size();

		// Create forget gaters
		for (int ui = 0; ui < memoryLayerSize; ui++) {
			Unit forgetGate;

			_orderedGaterIndices.push_back(_units.size());

			forgetGate._activation = 0.0f;
			forgetGate._state = 0.0f;
			forgetGate._prevState = 0.0f;
			forgetGate._bias = weightDist(randomGenerator);
			forgetGate._biasEligibility = 0.0f;
			forgetGate._prevBias = 0.0f;

			// Input connections
			for (int i = 0; i < numInputs; i++) {
				Connection c;
				c._gaterIndex = -1;
				c._weight = weightDist(randomGenerator);
				c._trace = 0.0f;
				c._eligibility = 0.0f;
				c._prevWeight = 0.0f;

				_connections[std::make_tuple(static_cast<int>(_units.size()), i)] = c;
			}

			// Previous memory cell connections
			for (int i = 0; i < memoryLayerSize; i++) {
				Connection c;
				c._gaterIndex = -1;
				c._weight = weightDist(randomGenerator);
				c._trace = 0.0f;
				c._eligibility = 0.0f;
				c._prevWeight = 0.0f;

				_connections[std::make_tuple(static_cast<int>(_units.size()), prevMemoryUnitsStart + i)] = c;
			}

			_units.push_back(forgetGate);
		}

		memoryUnitsStart = _units.size();

		// Create memory units
		for (int ui = 0; ui < memoryLayerSize; ui++) {
			Unit memoryUnit;

			memoryUnit._activation = 0.0f;
			memoryUnit._state = 0.0f;
			memoryUnit._prevState = 0.0f;
			memoryUnit._bias = weightDist(randomGenerator);
			memoryUnit._biasEligibility = 0.0f;
			memoryUnit._prevBias = 0.0f;

			// Input connections
			for (int i = 0; i < numInputs; i++) {
				Connection c;
				c._gaterIndex = inputGatersStart + i;
				c._weight = weightDist(randomGenerator);
				c._trace = 0.0f;
				c._eligibility = 0.0f;
				c._prevWeight = 0.0f;

				_connections[std::make_tuple(static_cast<int>(_units.size()), i)] = c;
			}

			// Recurrent connection
			{
				Connection c;
				c._gaterIndex = forgetGatersStart + _units.size() - memoryUnitsStart;
				c._weight = weightDist(randomGenerator);
				c._trace = 0.0f;
				c._eligibility = 0.0f;
				c._prevWeight = 0.0f;

				_connections[std::make_tuple(static_cast<int>(_units.size()), static_cast<int>(_units.size()))] = c;
			}

			// Previous memory cell connections
			for (int i = 0; i < memoryLayerSize; i++) {
				Connection c;
				c._gaterIndex = inputGatersStart + i;
				c._weight = weightDist(randomGenerator);
				c._trace = 0.0f;
				c._eligibility = 0.0f;
				c._prevWeight = 0.0f;

				_connections[std::make_tuple(static_cast<int>(_units.size()), prevMemoryUnitsStart + i)] = c;
			}

			_units.push_back(memoryUnit);
		}

		outputGatersStart = _units.size();

		// Create output gaters
		for (int ui = 0; ui < memoryLayerSize; ui++) {
			Unit outputGater;

			_orderedGaterIndices.push_back(_units.size());

			outputGater._activation = 0.0f;
			outputGater._state = 0.0f;
			outputGater._prevState = 0.0f;
			outputGater._bias = weightDist(randomGenerator);
			outputGater._biasEligibility = 0.0f;
			outputGater._prevBias = 0.0f;

			// Input connections
			for (int i = 0; i < numInputs; i++) {
				Connection c;
				c._gaterIndex = -1;
				c._weight = weightDist(randomGenerator);
				c._trace = 0.0f;
				c._eligibility = 0.0f;
				c._prevWeight = 0.0f;

				_connections[std::make_tuple(static_cast<int>(_units.size()), i)] = c;
			}

			// Previous memory cell connections
			for (int i = 0; i < memoryLayerSize; i++) {
				Connection c;
				c._gaterIndex = -1;
				c._weight = weightDist(randomGenerator);
				c._trace = 0.0f;
				c._eligibility = 0.0f;
				c._prevWeight = 0.0f;

				_connections[std::make_tuple(static_cast<int>(_units.size()), prevMemoryUnitsStart + i)] = c;
			}

			_units.push_back(outputGater);
		}

		// Add connections to input gaters (not fully connected)
		for (int i = 0; i < memoryLayerSize; i++) {
			Connection c;
			c._gaterIndex = -1;
			c._weight = weightDist(randomGenerator);
			c._trace = 0.0f;
			c._eligibility = 0.0f;
			c._prevWeight = 0.0f;

			_connections[std::make_tuple(i + inputGatersStart, i + memoryUnitsStart)] = c;
		}

		// Add connections to forget gaters (not fully connected)
		for (int i = 0; i < memoryLayerSize; i++) {
			Connection c;
			c._gaterIndex = -1;
			c._weight = weightDist(randomGenerator);
			c._trace = 0.0f;
			c._eligibility = 0.0f;
			c._prevWeight = 0.0f;

			_connections[std::make_tuple(i + forgetGatersStart, i + memoryUnitsStart)] = c;
		}

		// Add connections to output gaters (not fully connected)
		for (int i = 0; i < memoryLayerSize; i++) {
			Connection c;
			c._gaterIndex = -1;
			c._weight = weightDist(randomGenerator);
			c._trace = 0.0f;
			c._eligibility = 0.0f;
			c._prevWeight = 0.0f;

			_connections[std::make_tuple(i + outputGatersStart, i + memoryUnitsStart)] = c;
		}

		outputGatersStarts.push_back(outputGatersStart);
		memoryUnitsStarts.push_back(memoryUnitsStart);
	}

	if (numHiddenLayers > 0) {

		// --------------------------- First hidden layer ---------------------------

		int hiddenUnitsStart = _units.size();

		for (int ui = 0; ui < hiddenLayerSize; ui++) {
			Unit hiddenUnit;

			hiddenUnit._activation = 0.0f;
			hiddenUnit._state = 0.0f;
			hiddenUnit._prevState = 0.0f;
			hiddenUnit._bias = weightDist(randomGenerator);
			hiddenUnit._biasEligibility = 0.0f;
			hiddenUnit._prevBias = 0.0f;

			// Input connections
			for (int i = 0; i < numInputs; i++) {
				Connection c;
				c._gaterIndex = -1;
				c._weight = weightDist(randomGenerator);
				c._trace = 0.0f;
				c._eligibility = 0.0f;
				c._prevWeight = 0.0f;

				_connections[std::make_tuple(static_cast<int>(_units.size()), i)] = c;
			}

			// Memory cell (gated) connections
			for (int ogl = 0; ogl < outputGatersStarts.size(); ogl++) {
				for (int i = 0; i < memoryLayerSize; i++) {
					Connection c;
					c._gaterIndex = outputGatersStarts[ogl] + i;
					c._weight = weightDist(randomGenerator);
					c._trace = 0.0f;
					c._eligibility = 0.0f;
					c._prevWeight = 0.0f;

					_connections[std::make_tuple(static_cast<int>(_units.size()), memoryUnitsStarts[ogl] + i)] = c;
				}
			}

			_units.push_back(hiddenUnit);
		}

		// ------------------------- All other hidden layers -------------------------

		for (int hl = 1; hl < numHiddenLayers; hl++) {
			int prevHiddenUnitsStart = hiddenUnitsStart;

			hiddenUnitsStart = _units.size();

			for (int ui = 0; ui < hiddenLayerSize; ui++) {
				Unit hiddenUnit;

				hiddenUnit._activation = 0.0f;
				hiddenUnit._state = 0.0f;
				hiddenUnit._prevState = 0.0f;
				hiddenUnit._bias = weightDist(randomGenerator);
				hiddenUnit._biasEligibility = 0.0f;
				hiddenUnit._prevBias = 0.0f;

				// Previous hidden connections
				for (int i = 0; i < hiddenLayerSize; i++) {
					Connection c;
					c._gaterIndex = -1;
					c._weight = weightDist(randomGenerator);
					c._trace = 0.0f;
					c._eligibility = 0.0f;
					c._prevWeight = 0.0f;

					_connections[std::make_tuple(static_cast<int>(_units.size()), prevHiddenUnitsStart + i)] = c;
				}

				_units.push_back(hiddenUnit);
			}
		}

		// --------------------------- Output layer ---------------------------

		{
			int prevHiddenUnitsStart = hiddenUnitsStart;

			hiddenUnitsStart = _units.size();

			for (int ui = 0; ui < numOutputs; ui++) {
				_outputIndices.push_back(_units.size());

				Unit outputUnit;

				outputUnit._activation = 0.0f;
				outputUnit._state = 0.0f;
				outputUnit._prevState = 0.0f;
				outputUnit._bias = weightDist(randomGenerator);
				outputUnit._biasEligibility = 0.0f;
				outputUnit._prevBias = 0.0f;

				// Previous hidden connections
				for (int i = 0; i < hiddenLayerSize; i++) {
					Connection c;
					c._gaterIndex = -1;
					c._weight = weightDist(randomGenerator);
					c._trace = 0.0f;
					c._eligibility = 0.0f;
					c._prevWeight = 0.0f;

					_connections[std::make_tuple(static_cast<int>(_units.size()), prevHiddenUnitsStart + i)] = c;
				}

				_units.push_back(outputUnit);
			}
		}
	}
	else {

		// --------------------------- Output layer ---------------------------

		for (int ui = 0; ui < numOutputs; ui++) {
			_outputIndices.push_back(_units.size());

			Unit outputUnit;

			outputUnit._activation = 0.0f;
			outputUnit._state = 0.0f;
			outputUnit._prevState = 0.0f;
			outputUnit._bias = weightDist(randomGenerator);
			outputUnit._biasEligibility = 0.0f;
			outputUnit._prevBias;

			// Input connections
			for (int i = 0; i < numInputs; i++) {
				Connection c;
				c._gaterIndex = -1;
				c._weight = weightDist(randomGenerator);
				c._trace = 0.0f;
				c._eligibility = 0.0f;
				c._prevWeight = 0.0f;

				_connections[std::make_tuple(static_cast<int>(_units.size()), i)] = c;
			}

			// Memory cell (gated) connections
			for (int ogl = 0; ogl < outputGatersStarts.size(); ogl++) {
				for (int i = 0; i < memoryLayerSize; i++) {
					Connection c;
					c._gaterIndex = outputGatersStarts[ogl] + i;
					c._weight = weightDist(randomGenerator);
					c._trace = 0.0f;
					c._eligibility = 0.0f;
					c._prevWeight = 0.0f;

					_connections[std::make_tuple(static_cast<int>(_units.size()), memoryUnitsStarts[ogl] + i)] = c;
				}
			}

			_units.push_back(outputUnit);
		}
	}

	// Build gaters maps as well as ingoing and outgoing connection arrays
	for (std::unordered_map<std::tuple<int, int>, Connection>::iterator it0 = _connections.begin(); it0 != _connections.end(); it0++) {
		int j = std::get<0>(it0->first);
		int i = std::get<1>(it0->first);

		if (it0->second._gaterIndex != -1) {
			_gaterIndices[it0->first] = it0->second._gaterIndex;
			_reverseGaterIndices[it0->second._gaterIndex].push_back(it0->first);
		}

		_units[j]._ingoingConnectionIndices.push_back(i);
		_units[i]._outgoingConnectionIndices.push_back(j);
	}

	// Build sorted list of gater indices
	std::sort(_orderedGaterIndices.begin(), _orderedGaterIndices.end());

	clear();
}

void LSTMG::clear() {
	for (std::unordered_map<std::tuple<int, int>, Connection>::iterator it0 = _connections.begin(); it0 != _connections.end(); it0++) {
		int j = std::get<0>(it0->first);
		int i = std::get<1>(it0->first);

		if (i != j) {
			_units[j]._state = _units[j]._activation = it0->second._trace = 0.0f;

			for (std::unordered_map<std::tuple<int, int>, int>::iterator it1 = _gaterIndices.begin(); it1 != _gaterIndices.end(); it1++) {
				int k = std::get<0>(it1->first);
				int a = std::get<1>(it1->first);

				if (j < k && j == it1->second)
					_extendedTraces[std::make_tuple(j, i, k)] = 0.0f;
			}
		}
	}
}

float LSTMG::gain(int j, int i) {
	std::unordered_map<std::tuple<int, int>, int>::iterator it0 = _gaterIndices.find(std::make_tuple(j, i));

	if (it0 != _gaterIndices.end())
		return _units[it0->second]._activation;

	if (_connections.find(std::make_tuple(j, i)) != _connections.end())
		return 1.0f;

	return 0.0f;
}

float LSTMG::theTerm(int j, int k) {
	float term = 0.0f;

	std::unordered_map<std::tuple<int, int>, int>::iterator it0 = _gaterIndices.find(std::make_tuple(k, k));

	if (it0 != _gaterIndices.end() && it0->second == j)
		term = _units[k]._prevState;

	for (std::unordered_map<std::tuple<int, int>, int>::iterator it1 = _gaterIndices.begin(); it1 != _gaterIndices.end(); it1++) {
		int l = std::get<0>(it1->first);
		int a = std::get<1>(it1->first);

		if (l == k && a != k && j == _gaterIndices[std::make_tuple(k, a)])
			term += _connections[std::make_tuple(k, a)]._weight * _prevActivations[std::make_tuple(k, a)];
	}

	return term;
}

void LSTMG::step(bool linearOutput) {
	_prevGains.clear();
	_prevActivations.clear();

	// Copy previous states
	for (int j = 0; j < _units.size(); j++)
		_units[j]._prevState = _units[j]._state;

	for (int j = _inputIndices.size(); j < _units.size(); j++) {
		float sg = _prevGains[std::make_tuple(j, -1)] = gain(j, j); // -1 means unused

		_units[j]._state *= sg;

		for (int ci = 0; ci < _units[j]._ingoingConnectionIndices.size(); ci++) {
			int i = _units[j]._ingoingConnectionIndices[ci];

			float g = _prevGains[std::make_tuple(j, i)] = gain(j, i);

			float a = _prevActivations[std::make_tuple(j, i)] = _units[i]._activation;

			Connection &c = _connections[std::make_tuple(j, i)];

			_units[j]._state += gain(j, i) * c._weight * _units[i]._activation;

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

	for (std::unordered_map<std::tuple<int, int, int>, float>::iterator it0 = _extendedTraces.begin(); it0 != _extendedTraces.end(); it0++) {
		int j = std::get<0>(it0->first);
		int i = std::get<1>(it0->first);
		int k = std::get<2>(it0->first);

		float terms = sigmoidDerivative(_units[j]._prevState) * _connections[std::make_tuple(j, i)]._trace * theTerm(j, k);
		it0->second = _prevGains[std::make_tuple(k, -1)] * it0->second + terms;
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
			int k = _units[j]._outgoingConnectionIndices[ci];

			errorProj[j] += errorResp[k] * _prevGains[std::make_tuple(k, j)] * _connections[std::make_tuple(k, j)]._weight;
		}

		float jDeriv = sigmoidDerivative(_units[j]._state);

		errorProj[j] *= jDeriv;

		int lastK = 0;

		for (int gi = 0; gi < _orderedGaterIndices.size(); gi++) {
			if (j == _orderedGaterIndices[gi]) {
				// For each connection
				const std::vector<std::tuple<int, int>> &gaterConnections = _reverseGaterIndices[_orderedGaterIndices[gi]];

				for (int gci = 0; gci < gaterConnections.size(); gci++) {
					int k = std::get<0>(gaterConnections[gci]);
					int a = std::get<1>(gaterConnections[gci]);

					if (lastK < k && j < k) {
						lastK = k;

						errorResp[j] += errorResp[k] * theTerm(j, k);
					}
				}
			}
		}

		errorResp[j] = errorProj[j] + jDeriv * errorResp[j];
	}

	for (std::unordered_map<std::tuple<int, int>, Connection>::iterator it0 = _connections.begin(); it0 != _connections.end(); it0++) {
		int j = std::get<0>(it0->first);
		int i = std::get<1>(it0->first);

		it0->second._eligibility *= eligibilityDecay;

		// If not output
		if (j < _units.size() - _outputIndices.size()) {
			it0->second._eligibility += errorProj[j] * it0->second._trace;

			for (std::unordered_map<std::tuple<int, int, int>, float>::iterator it1 = _extendedTraces.begin(); it1 != _extendedTraces.end(); it1++) {
				int l = std::get<0>(it1->first);
				int m = std::get<1>(it1->first);
				int k = std::get<2>(it1->first);

				if (l == j && m == i)
					it0->second._eligibility += errorResp[k] * it1->second;
			}
		}
		else
			it0->second._eligibility += errorResp[j] * it0->second._trace;
	}

	// Update biases
	for (int j = 0; j < _units.size(); j++) {
		_units[j]._biasEligibility *= eligibilityDecay;
		_units[j]._biasEligibility += errorResp[j];
	}
}

void LSTMG::getDeltas(const std::vector<float> &targets, std::vector<float> &inputError, float eligibilityDecay, bool linearOutput) {
	std::vector<float> errorResp(_units.size(), 0.0f);
	std::vector<float> errorProj(_units.size(), 0.0f);

	// Output layer errors
	for (int i = 0; i < _outputIndices.size(); i++)
		errorResp[_outputIndices[i]] = targets[i] - _units[_outputIndices[i]]._activation;

	// Loop over all units in reversed order of activation
	for (int j = _units.size() - _outputIndices.size() - 1; j >= 0; j--) {
		errorResp[j] = errorProj[j] = 0.0f;

		for (int ci = 0; ci < _units[j]._outgoingConnectionIndices.size(); ci++) {
			int k = _units[j]._outgoingConnectionIndices[ci];

			errorProj[j] += errorResp[k] * _prevGains[std::make_tuple(k, j)] * _connections[std::make_tuple(k, j)]._weight;
		}

		float jDeriv = sigmoidDerivative(_units[j]._state);

		errorProj[j] *= jDeriv;

		int lastK = 0;

		for (int gi = 0; gi < _orderedGaterIndices.size(); gi++) {
			if (j == _orderedGaterIndices[gi]) {
				// For each connection
				const std::vector<std::tuple<int, int>> &gaterConnections = _reverseGaterIndices[_orderedGaterIndices[gi]];

				for (int gci = 0; gci < gaterConnections.size(); gci++) {
					int k = std::get<0>(gaterConnections[gci]);
					int a = std::get<1>(gaterConnections[gci]);

					if (lastK < k && j < k) {
						lastK = k;

						errorResp[j] += errorResp[k] * theTerm(j, k);
					}
				}
			}
		}

		errorResp[j] = errorProj[j] + jDeriv * errorResp[j];
	}

	for (std::unordered_map<std::tuple<int, int>, Connection>::iterator it0 = _connections.begin(); it0 != _connections.end(); it0++) {
		int j = std::get<0>(it0->first);
		int i = std::get<1>(it0->first);

		it0->second._eligibility *= eligibilityDecay;

		// If not output
		if (j < _units.size() - _outputIndices.size()) {
			it0->second._eligibility += errorProj[j] * it0->second._trace;

			for (std::unordered_map<std::tuple<int, int, int>, float>::iterator it1 = _extendedTraces.begin(); it1 != _extendedTraces.end(); it1++) {
				int l = std::get<0>(it1->first);
				int m = std::get<1>(it1->first);
				int k = std::get<2>(it1->first);

				if (l == j && m == i)
					it0->second._eligibility += errorResp[k] * it1->second;
			}
		}
		else
			it0->second._eligibility += errorResp[j] * it0->second._trace;
	}

	// Update biases
	for (int j = 0; j < _units.size(); j++) {
		_units[j]._biasEligibility *= eligibilityDecay;
		_units[j]._biasEligibility += errorResp[j];
	}

	if (inputError.size() != _inputIndices.size())
		inputError.resize(_inputIndices.size());

	for (int j = 0; j < _inputIndices.size(); j++)
		inputError[j] = errorResp[_inputIndices[j]];
}

void LSTMG::moveAlongDeltas(float error, float momentum) {
	for (std::unordered_map<std::tuple<int, int>, Connection>::iterator it0 = _connections.begin(); it0 != _connections.end(); it0++) {
		int j = std::get<0>(it0->first);
		int i = std::get<1>(it0->first);

		float d = error * it0->second._eligibility + momentum * it0->second._prevWeight;
		it0->second._weight += d;
		it0->second._prevWeight = d;
	}

	// Update biases
	for (int j = 0; j < _units.size(); j++) {
		float d = error * _units[j]._biasEligibility + momentum * _units[j]._prevBias;
		_units[j]._bias += d;
		_units[j]._prevBias = d;
	}
}