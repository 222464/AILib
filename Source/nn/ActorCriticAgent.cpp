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

#include <nn/ActorCriticAgent.h>

#include <algorithm>
#include <iostream>

using namespace nn;

ActorCriticAgent::ActorCriticAgent()
: _actorAlpha(0.2f), _criticAlpha(0.02f), _criticDecay(0.002f),
_lambda(1.0f), _actorLambda(0.05f), _gamma(0.95f), _errorScalar(1.0f), _minError(0.0001f),
_numBackpropPasses(3), _actorMomentum(0.01f), _criticMomentum(0.01f),
_criticErrorTolerance(0.02f),
_negativeErrorMultiplier(1.0f),
_numPseudoRehearsalSamplesCritic(12),
_pseudoRehearsalSampleStdDev(0.75f),
_pseudoRehearsalSampleMean(0.0f),
_maxEligibilityTraceChainSize(60)
{}

void ActorCriticAgent::createRandom(size_t numInputs, size_t numOutputs,
	size_t numActorHiddenLayers, size_t numActorNeuronsPerHiddenLayer,
	size_t numCriticHiddenLayers, size_t numCriticNeuronsPerHiddenLayer,
	float minWeight, float maxWeight, unsigned long seed)
{
	std::mt19937 generator;
	generator.seed(seed);

	_actor.createRandom(numInputs, numOutputs, numActorHiddenLayers, numActorNeuronsPerHiddenLayer, minWeight, maxWeight, generator);
	_critic.createRandom(numInputs, 1, numCriticHiddenLayers, numCriticNeuronsPerHiddenLayer, minWeight, maxWeight, generator);

	_prevInputs.assign(numInputs, 0.0f);
}

void ActorCriticAgent::step(float reward) {
	std::vector<float> currentInputs(getNumInputs());

	for (size_t i = 0; i < getNumInputs(); i++) {
		currentInputs[i] = _actor.getInput(i);
		_critic.setInput(i, _actor.getInput(i));
	}

	_critic.activateLinearOutputLayer();

	float value = _critic.getOutput(0);

	for (size_t i = 0; i < getNumInputs(); i++)
		_critic.setInput(i, _prevInputs[i]);

	_critic.activateLinearOutputLayer();

	float prevValue = _critic.getOutput(0);

	// Gather rehearsal samples
	std::vector<IOSet> criticRehearsalSamples(_numPseudoRehearsalSamplesCritic);

	std::normal_distribution<float> pseudoRehearsalInputDistribution(_pseudoRehearsalSampleMean, _pseudoRehearsalSampleStdDev);

	for (size_t i = 0; i < _numPseudoRehearsalSamplesCritic; i++) {
		// Generate critic sample
		criticRehearsalSamples[i]._inputs.resize(_critic.getNumInputs());

		for (size_t j = 0; j < _actor.getNumInputs(); j++) {
			criticRehearsalSamples[i]._inputs[j] = pseudoRehearsalInputDistribution(_generator);
			_critic.setInput(j, criticRehearsalSamples[i]._inputs[j]);
		}

		_critic.activateLinearOutputLayer();

		criticRehearsalSamples[i]._outputs.resize(_critic.getNumOutputs());

		for (size_t j = 0; j < _critic.getNumOutputs(); j++)
			criticRehearsalSamples[i]._outputs[j] = _critic.getOutput(j);
	}

	float newPrevValue = reward + _gamma * value;
	float tdError = newPrevValue - prevValue;
	float criticTarget = prevValue + _lambda * tdError;

	// Add discounted reward to eligibility traces. Propagate reward downward
	float prevChainValue = criticTarget;

	for (std::list<EligibilityTrace>::iterator it = _eligibilityTraceChain.begin(); it != _eligibilityTraceChain.end(); it++) {
		it->_value = (1.0f - _lambda) * it->_value + _lambda * (it->_reward + _gamma * prevChainValue);
		prevChainValue = it->_value;
	}

	for (size_t i = 0; i < getNumInputs(); i++)
		_critic.setInput(i, _prevInputs[i]);

	_critic.activateLinearOutputLayer();

	_critic.updateValueFunction(tdError, _criticAlpha, 0.01f);

	_critic.decayWeights(_criticDecay);

	// Add sample
	EligibilityTrace trace;
	trace._inputs = _prevInputs;
	trace._value = criticTarget;
	trace._reward = reward;

	_eligibilityTraceChain.push_front(trace);

	while (_eligibilityTraceChain.size() > _maxEligibilityTraceChainSize)
		_eligibilityTraceChain.pop_back();

	for (size_t i = 0; i < getNumInputs(); i++)
		_actor.setInput(i, _prevInputs[i]);

	_actor.reinforceArp(std::min(1.0f, std::max(-1.0f, tdError * _errorScalar)) * 0.5f + 0.5f, _actorAlpha, _actorLambda);

	for (size_t i = 0; i < getNumInputs(); i++)
		_actor.setInput(i, currentInputs[i]);

	_actor.activateArp(_generator);

	for (size_t i = 0; i < getNumInputs(); i++)
		_critic.setInput(i, currentInputs[i]);

	_critic.activateLinearOutputLayer();

	_prevInputs = currentInputs;
}

void ActorCriticAgent::writeToStream(std::ostream &stream) {
	stream << _actorAlpha << " " << _criticAlpha << " " << _lambda << " " << _gamma << " " << _minError << std::endl;
	stream << _numBackpropPasses << " " << _criticMomentum << " " << _criticErrorTolerance << " " << _negativeErrorMultiplier << std::endl;
	stream << _numPseudoRehearsalSamplesCritic << " " << _pseudoRehearsalSampleStdDev << " " << _pseudoRehearsalSampleMean << std::endl;
	stream << _maxEligibilityTraceChainSize << std::endl;

	_actor.writeToStream(stream);
	_critic.writeToStream(stream);
}

void ActorCriticAgent::readFromStream(std::istream &stream) {
	stream >> _actorAlpha;
	stream >> _criticAlpha;
	stream >> _lambda;
	stream >> _gamma;
	stream >> _minError;

	stream >> _numBackpropPasses;
	stream >> _criticMomentum;
	stream >> _criticErrorTolerance;
	stream >> _negativeErrorMultiplier;

	stream >> _numPseudoRehearsalSamplesCritic;
	stream >> _pseudoRehearsalSampleStdDev;
	stream >> _pseudoRehearsalSampleMean;

	stream >> _maxEligibilityTraceChainSize;

	_actor.readFromStream(stream);
	_critic.readFromStream(stream);

	_prevInputs.assign(_actor.getNumInputs(), 0.0f);
}