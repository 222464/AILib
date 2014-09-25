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

#include <nn/RLLSTMAgent.h>

using namespace nn;

RLLSTMAgent::RLLSTMAgent()
: _selectedAction(0), _prevError(0.0f),
_alpha(2.0f), _gamma(0.8f), _k(1.2f),
_numExpBackpropPasses(8), _explorationMultiplier(10.0f),
_expAlpha(0.01f), _expMomentum(0.05f), _expGamma(0.8f)
{}

void RLLSTMAgent::createRandom(size_t numInputs, size_t numActions,
	size_t recNumHiddenLayers, size_t recNumNeuronsPerHiddenLayer,
	size_t expNumHiddenLayers, size_t expNumNeuronsPerHiddenLayer,
	size_t numMemoryCells, float minWeight, float maxWeight, unsigned long seed)
{
	_numInputs = numInputs;
	_numActions = numActions;

	std::mt19937 generator;
	generator.seed(seed);

	_rnn.createRandom(_numInputs + numMemoryCells, _numActions + numMemoryCells * 4, recNumHiddenLayers, recNumNeuronsPerHiddenLayer, minWeight, maxWeight, generator);

	_expnn.createRandom(_numInputs, 1, expNumHiddenLayers, expNumNeuronsPerHiddenLayer, minWeight, maxWeight, generator);

	_memoryCells.resize(numMemoryCells);

	_prevInputs.resize(_numInputs, 0.0f);

	_generator.seed(seed);
}

void RLLSTMAgent::step(float fitness) {
	size_t prevSeletedAction = _selectedAction;

	float prevAdvantage = _rnn.getOutput(prevSeletedAction);

	float prevValue = _rnn.getOutput(0);

	for (size_t i = 1; i < _numActions; i++)
	if (_rnn.getOutput(i) > prevValue)
		prevValue = _rnn.getOutput(i);

	// Update to find new action
	for (size_t i = 0; i < _memoryCells.size(); i++)
		_rnn.setInput(_numInputs + i, _memoryCells[i]._output);

	_rnn.activateLinearOutputLayer();

	// Update memory cells
	size_t outputIndex = _numActions;

	for (size_t i = 0; i < _memoryCells.size(); i++) {
		_memoryCells[i]._input = _rnn.getOutput(outputIndex++);
		_memoryCells[i]._gateInput = _rnn.getOutput(outputIndex++);
		_memoryCells[i]._gateOutput = _rnn.getOutput(outputIndex++);
		_memoryCells[i]._gateForget = _rnn.getOutput(outputIndex++);

		_memoryCells[i].activate(_rnn._activationMultiplier, _rnn._outputTraceDecay);
	}

	// Find exploration factor
	_expnn.activateLinearOutputLayer();

	float predictedError = _expnn.getOutput(0);

	float explorationStd = predictedError * _explorationMultiplier;

	// Find new value
	_selectedAction = 0;

	float value = _rnn.getOutput(0);

	std::normal_distribution<float> distribution(0.0f, explorationStd);

	float perturbedValue = _rnn.getOutput(0) + distribution(_generator);
	
	for (size_t i = 1; i < _numActions; i++) {
		float perturbedOutput = _rnn.getOutput(i) + distribution(_generator);

		if (perturbedOutput > perturbedValue) {
			perturbedValue = perturbedOutput;
			_selectedAction = i;
		}

		if (_rnn.getOutput(i) > value)
			value = _rnn.getOutput(i);
	}

	// Find TD error
	float error = prevValue + (fitness + _gamma * value - prevValue) / _k - prevAdvantage;

	_rnn.reinforce(error * _alpha);

	// Update exploration error predictor
	for (size_t i = 0; i < _numInputs; i++)
		_expnn.setInput(i, _prevInputs[i]);

	float expTarget = _prevError + _expGamma * error;

	for (size_t p = 0; p < _numExpBackpropPasses; p++) {
		_expnn.activateLinearOutputLayer();

		FeedForwardNeuralNetwork::Gradient gradient;
		_expnn.getGradientLinearOutputLayer(std::vector<float>(1, expTarget), gradient);
		_expnn.moveAlongGradientMomentum(gradient, _expAlpha, _expMomentum);
	}

	for (size_t i = 0; i < _numInputs; i++)
		_prevInputs[i] = _rnn.getInput(i);

	_prevError = error;
}