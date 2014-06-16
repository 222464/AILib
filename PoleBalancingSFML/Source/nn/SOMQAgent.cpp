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

#include <nn/SOMQAgent.h>

#include <iostream>

using namespace nn;

SOMQAgent::SOMQAgent()
: _alpha(0.1f), _gamma(0.98f),
_tupleChainSize(20)
{}

void SOMQAgent::createRandom(size_t numInputs, size_t numOutputs, size_t dimensions, size_t dimensionSize, const nn::BrownianPerturbation &perturbation, float minWeight, float maxWeight, unsigned long seed) {
	_numInputs = numInputs;
	_numOutputs = numOutputs;

	/*_stateActionSOM.createRandom(_numInputs, dimensions, dimensionSize, _numStates, minWeight, maxWeight, seed + 1);

	_qMatrix.assign(_numStates * _numActions, 0.0f);

	_generator.seed(seed);

	_input.assign(_numInputs, 0.0f);
	_output.assign(_numOutputs, 0.0f);

	_prevInput.assign(_numInputs, 0.0f);
	_prevOutput.assign(_numOutputs, 0.0f);

	_outputOffsets.assign(_numOutputs, perturbation);*/
}

void SOMQAgent::step(float fitness, float dt) {
	/*// Identify unit in state map closest to input
	SOM::SOMCoords closestState = _stateSOM.getBestMatchingUnit(_input);

	size_t state = closestState._coords[0];

	// Find action with best Q for this state
	float maxQ = getQ(state, 0);
	size_t action = 0;

	for (size_t a = 1; a < _numActions; a++)
	if (getQ(state, a) > maxQ) {
		maxQ = getQ(state, a);
		action = a;
	}

	SOM::SOMCoords prevClosestState = _stateSOM.getBestMatchingUnit(_prevInput);
	SOM::SOMCoords prevClosestAction = _actionSOM.getBestMatchingUnit(_prevOutput);

	// If previously selected action was good
	if (fitness + _gamma * maxQ > getQ(prevClosestState._coords[0], prevClosestAction._coords[0]))
		// Update action map towards the selected action
		_actionSOM.updateNeighborhood(prevClosestAction, _prevOutput);

	int stateRadius = static_cast<int>(std::ceil(static_cast<float>(_stateSOM.getDimensionSize()) * _stateSOM._neighborhoodRadius));
	float stateRadiusSquaredf = static_cast<float>(stateRadius * stateRadius);

	int actionRadius = static_cast<int>(std::ceil(static_cast<float>(_actionSOM.getDimensionSize()) * _actionSOM._neighborhoodRadius));
	float actionRadiusSquaredf = static_cast<float>(actionRadius * actionRadius);

	// Update Q values
	for (size_t s = 0; s < _numStates; s++)
	for (size_t a = 0; a < _numActions; a++) {
		float deltaState = static_cast<float>(s) - static_cast<float>(prevClosestState._coords[0]);
		float stateInfluence = std::expf(-(deltaState * deltaState) / (2.0f * stateRadiusSquaredf));

		float deltaAction = static_cast<float>(a) - static_cast<float>(prevClosestAction._coords[0]);
		float actionInfluence = std::expf(-(deltaAction * deltaAction) / (2.0f * actionRadiusSquaredf));

		setQ(s, a, getQ(s, a) + _alpha * stateInfluence * actionInfluence * (fitness + _gamma * maxQ - getQ(s, a)));
	}

	// Update previous tuples
	float gamma = _gamma;

	for (std::list<StateActionTuple>::iterator it = _tupleChain.begin(); it != _tupleChain.end(); it++, gamma *= _gamma) {
		SOM::SOMCoords closestState = _stateSOM.getBestMatchingUnit(it->_input);
		SOM::SOMCoords closestAction = _actionSOM.getBestMatchingUnit(it->_output);

		for (size_t s = 0; s < _numStates; s++)
		for (size_t a = 0; a < _numActions; a++) {
			float deltaState = static_cast<float>(s) - static_cast<float>(closestState._coords[0]);
			float stateInfluence = std::expf(-(deltaState * deltaState) / (2.0f * stateRadiusSquaredf));

			float deltaAction = static_cast<float>(a) - static_cast<float>(closestAction._coords[0]);
			float actionInfluence = std::expf(-(deltaAction * deltaAction) / (2.0f * actionRadiusSquaredf));

			setQ(closestState._coords[0], closestAction._coords[0], getQ(closestState._coords[0], closestAction._coords[0]) + _alpha * stateInfluence * actionInfluence * gamma * fitness);
		}
	}

	// Update input map
	_stateSOM.updateNeighborhood(closestState, _input);

	_prevOutput = _output;

	// Get new action
	SOM::SOMCoords actionCoords;
	actionCoords._coords.assign(1, action);

	// Get action vector, perturb it for exploration
	for (size_t i = 0; i < _numOutputs; i++) {
		_outputOffsets[i].update(_generator, dt);
		_output[i] = _actionSOM.getNode(actionCoords)._weights[i] + _outputOffsets[i]._position;
	}

	// Add tuple
	StateActionTuple tuple;
	tuple._input = _input;
	tuple._output = _output;

	_tupleChain.push_front(tuple);

	if (_tupleChain.size() > _tupleChainSize)
		_tupleChain.pop_back();

	_prevInput = _input;*/
}
