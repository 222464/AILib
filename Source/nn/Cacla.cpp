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

#include <nn/Cacla.h>
#include <algorithm>
#include <iostream>
#include <SFML/Window.hpp>

using namespace nn;

Cacla::Cacla()
: _variance(1.0f), _outputOffsetScalar(1.0f), _outputOffsetAbsErrorScalar(0.0f), _reverseOffsetErrorScalar(40.0f),
_actorAlpha(0.1f), _criticAlpha(0.1f), _actorDecay(0.0f), _criticDecay(0.0f),
_lambda(0.1f), _gamma(0.95f),
_numCriticBackpropPasses(50), _numActorBackpropPasses(03),
_actorMomentum(0.05f), _criticMomentum(0.05f),
_varianceDecay(0.07f),
_actorErrorTolerance(0.0001f),
_criticErrorTolerance(0.0001f),
_numPseudoRehearsalSamplesActor(50),
_numPseudoRehearsalSamplesCritic(0),
_pseudoRehearsalSampleStdDev(0.75f),
_pseudoRehearsalSampleMean(0.0f),
_maxEligibilityTraceChainSize(200),
_prevReward(0.0f),
_actorRehearseNewSampleChance(0.333f)
{}

void Cacla::createRandom(size_t numInputs, size_t numOutputs,
	size_t numActorHiddenLayers, size_t numActorNeuronsPerHiddenLayer,
	size_t numCriticHiddenLayers, size_t numCriticNeuronsPerHiddenLayer,
	const BrownianPerturbation &perturbation,
	float minWeight, float maxWeight,
	unsigned long seed)
{
	std::mt19937 generator;
	generator.seed(seed);

	_actor.createRandom(numInputs, numOutputs, numActorHiddenLayers, numActorNeuronsPerHiddenLayer, minWeight, maxWeight, generator);
	_critic.createRandom(numInputs, 1, numCriticHiddenLayers, numCriticNeuronsPerHiddenLayer, minWeight, maxWeight, generator);

	_prevInputs.assign(numInputs, 0.0f);
	_prevPrevInputs.assign(numInputs, 0.0f);
	_outputOffsets.assign(numOutputs, perturbation);
	_outputs.assign(numOutputs, 0.0f);
	_prevOutputs.assign(numOutputs, 0.0f);

	_generator.seed(seed);
}

void Cacla::step(float reward, float dt) {
	std::vector<float> currentInputs(getNumInputs());

	for (size_t i = 0; i < getNumInputs(); i++)
		currentInputs[i] = _actor.getInput(i);

	_critic.activateLinearOutputLayer();

	float value = _critic.getOutput(0);

	for (size_t i = 0; i < getNumInputs(); i++)
		_critic.setInput(i, _prevInputs[i]);

	_critic.activateLinearOutputLayer();

	float prevValue = _critic.getOutput(0);

	// Gather rehearsal samples
	std::vector<IOSet> actorRehearsalSamples(_numPseudoRehearsalSamplesActor);
	std::vector<IOSet> criticRehearsalSamples(_numPseudoRehearsalSamplesCritic);

	std::normal_distribution<float> pseudoRehearsalInputDistribution(_pseudoRehearsalSampleMean, _pseudoRehearsalSampleStdDev);

	for (size_t i = 0; i < _numPseudoRehearsalSamplesActor; i++) {
		// Generate actor sample
		actorRehearsalSamples[i]._inputs.resize(_actor.getNumInputs());

		for (size_t j = 0; j < _actor.getNumInputs(); j++) {
			actorRehearsalSamples[i]._inputs[j] = pseudoRehearsalInputDistribution(_generator);
			_actor.setInput(j, actorRehearsalSamples[i]._inputs[j]);
		}

		_actor.activateLinearOutputLayer();

		actorRehearsalSamples[i]._outputs.resize(_actor.getNumOutputs());

		for (size_t j = 0; j < _actor.getNumOutputs(); j++)
			actorRehearsalSamples[i]._outputs[j] = _actor.getOutput(j);
	}

	for (size_t i = 0; i < _numPseudoRehearsalSamplesCritic; i++) {
		// Generate critic sample
		criticRehearsalSamples[i]._inputs.resize(_critic.getNumInputs());

		for (size_t j = 0; j < _critic.getNumInputs(); j++) {
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
	//std::cout << criticTarget << std::endl;
	// Add discounted reward to eligibility traces. Propagate reward downward
	float prevChainValue = criticTarget;

	for (std::list<EligibilityTrace>::iterator it = _eligibilityTraceChain.begin(); it != _eligibilityTraceChain.end(); it++) {
		it->_value = (1.0f - _lambda) * it->_value + _lambda * (it->_reward + _gamma * prevChainValue);
		prevChainValue = it->_reward + _gamma * prevChainValue;
	}

	// Add sample
	EligibilityTrace trace;
	trace._inputs = _prevInputs;
	trace._value = criticTarget;
	trace._reward = reward;

	_eligibilityTraceChain.push_front(trace);

	while (_eligibilityTraceChain.size() > _maxEligibilityTraceChainSize)
		_eligibilityTraceChain.pop_back();

	// Accumulate all samples into one buffer
	std::vector<IOSet> traces(_eligibilityTraceChain.size() + criticRehearsalSamples.size());

	size_t traceIndex = 0;

	for (std::list<EligibilityTrace>::iterator it = _eligibilityTraceChain.begin(); it != _eligibilityTraceChain.end(); it++, traceIndex++) {
		traces[traceIndex]._inputs = it->_inputs;
		traces[traceIndex]._outputs = std::vector<float>(1, it->_value);
	}

	for (size_t i = 0; i < criticRehearsalSamples.size(); i++, traceIndex++)
		traces[traceIndex] = criticRehearsalSamples[i];

	std::uniform_int_distribution<int> sampleDistCritic(0, traces.size() - 1);

	for (size_t p = 0; p < _numCriticBackpropPasses; p++) {
		size_t randIndex = static_cast<size_t>(sampleDistCritic(_generator));

		for (size_t i = 0; i < _critic.getNumInputs(); i++)
			_critic.setInput(i, traces[randIndex]._inputs[i]);

		_critic.activateLinearOutputLayer();

		FeedForwardNeuralNetwork::Gradient gradient;
		_critic.getGradientLinearOutputLayer(traces[randIndex]._outputs, gradient);
		_critic.moveAlongGradientMomentum(gradient, _criticAlpha, _criticMomentum);
	}

	_critic.decayWeights(_criticDecay);

	// Update actor if did better than before
	if (tdError > 0.0f) {
		//std::cout << "t";
		std::uniform_int_distribution<int> sampleDistActor(0, actorRehearsalSamples.size() - 1);
		std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

		for (size_t p = 0; p < _numActorBackpropPasses; p++) {
			if (actorRehearsalSamples.empty() || dist01(_generator) < _actorRehearseNewSampleChance) {
				for (size_t i = 0; i < _actor.getNumInputs(); i++)
					_actor.setInput(i, _prevInputs[i]);

				_actor.activateLinearOutputLayer();

				FeedForwardNeuralNetwork::Gradient gradient;
				_actor.getGradientLinearOutputLayer(_outputs, gradient);
				_actor.moveAlongGradientMomentum(gradient, _actorAlpha, _actorMomentum);
			}
			else {
				size_t randIndex = static_cast<size_t>(sampleDistActor(_generator));

				for (size_t i = 0; i < _actor.getNumInputs(); i++)
					_actor.setInput(i, actorRehearsalSamples[randIndex]._inputs[i]);

				_actor.activateLinearOutputLayer();

				FeedForwardNeuralNetwork::Gradient gradient;
				_actor.getGradientLinearOutputLayer(actorRehearsalSamples[randIndex]._outputs, gradient);
				_actor.moveAlongGradientMomentum(gradient, _actorAlpha, _actorMomentum);
			}
		}
	}

	_actor.decayWeights(_actorDecay);

	for (size_t i = 0; i < currentInputs.size(); i++)
		_actor.setInput(i, currentInputs[i]);

	_actor.activateLinearOutputLayer();

	_prevOutputs = _outputs;

	for (size_t i = 0; i < _actor.getNumOutputs(); i++) {
		_outputOffsets[i].update(_generator, dt);
		_outputs[i] = std::max(-1.0f, std::min(1.0f, _actor.getOutput(i))) + _outputOffsets[i]._position;
	}

	_prevPrevInputs = _prevInputs;
	_prevInputs = currentInputs;

	_prevValue = value;
	_prevReward = reward;
}

void Cacla::writeToStream(std::ostream &stream) {
	stream << _outputOffsetScalar << std::endl;
	stream << _actorAlpha << " " << _criticAlpha << " " << _lambda << " " << _gamma << std::endl;
	stream << _numCriticBackpropPasses << " " << _numActorBackpropPasses << std::endl;
	stream << _actorMomentum << " " << _criticMomentum << std::endl;
	stream << _varianceDecay << std::endl;
	stream << _numPseudoRehearsalSamplesActor << std::endl;
	stream << _numPseudoRehearsalSamplesCritic << std::endl;
	stream << _pseudoRehearsalSampleStdDev << std::endl;
	stream << _pseudoRehearsalSampleMean << std::endl;

	_actor.writeToStream(stream);
	_critic.writeToStream(stream);
}

void Cacla::readFromStream(std::istream &stream) {
	stream >> _outputOffsetScalar;

	stream >> _actorAlpha;
	stream >> _criticAlpha;
	stream >> _lambda;
	stream >> _gamma;

	stream >> _numCriticBackpropPasses;
	stream >> _numActorBackpropPasses;

	stream >> _actorMomentum;
	stream >> _criticMomentum;

	stream >> _varianceDecay;

	stream >> _numPseudoRehearsalSamplesActor;
	stream >> _numPseudoRehearsalSamplesCritic;
	stream >> _pseudoRehearsalSampleStdDev;
	stream >> _pseudoRehearsalSampleMean;

	_actor.readFromStream(stream);
	_critic.readFromStream(stream);

	_prevInputs.assign(_actor.getNumInputs(), 0.0f);

	_outputOffsets.assign(_actor.getNumOutputs(), nn::BrownianPerturbation());

	_outputs.assign(_actor.getNumOutputs(), 0.0f);
}