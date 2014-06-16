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

#include <nn/PSOAgent.h>

#include <algorithm>

#include <iostream>

using namespace nn;

PSOAgent::PSOAgent()
: _minWeight(-2.0f), _maxWeight(2.0f),
_minVelocity(-0.05f), _maxVelocity(0.05f),
_alpha(0.5f), _gamma(0.92f),
_minAttraction(0.05f), _maxAttraction(0.4f),
_particleVelocityDecay(0.2f),
_rewardDecay(0.3f),
_temperature(0.05f),
_numCriticBackpropPasses(8),
_maxEligibilityTraceChainSize(60),
_criticAlpha(0.02f), _criticMomentum(0.0f),
_numPseudoRehearsalSamplesCritic(8),
_pseudoRehearsalSampleStdDev(0.75f),
_pseudoRehearsalSampleMean(0.0f),
_criticErrorTolerance(0.05f),
_criticDecay(0.001f),
_teleportChance(0.0f),
_maxTeleportPerturbation(0.1f),
_greedExponent(2.0f),
_particleMassResistanceInv(0.001f),
_attractionOffset(-0.001f),
_prevParticleIndex(0),
_particleIndex(0)
{}

void PSOAgent::createRandom(size_t numInputs, size_t numOutputs,
	size_t actorNumHiddenLayers, size_t actorNumNeuronsPerHiddenLayer,
	size_t criticNumHiddenLayers, size_t criticNumNeuronsPerHiddenLayer,
	size_t numParticles, float initMinWeight, float initMaxWeight,
	unsigned long seed)
{
	_numInputs = numInputs;
	_numOutputs = numOutputs;

	_actor.createRandom(_numInputs, _numOutputs, actorNumHiddenLayers, actorNumNeuronsPerHiddenLayer, _minWeight, _maxWeight, seed + 1);

	_critic.createRandom(_numInputs, 1, criticNumHiddenLayers, criticNumNeuronsPerHiddenLayer, _minWeight, _maxWeight, seed + 2);

	_generator.seed(seed);

	// Create particles
	std::uniform_real_distribution<float> weightDistribution(initMinWeight, initMaxWeight);

	_particles.resize(numParticles);

	size_t particleWeightVectorSize = _actor.getWeightVectorSize();

	for (size_t i = 0; i < _particles.size(); i++) {
		_particles[i]._position.resize(particleWeightVectorSize);

		for (size_t j = 0; j < particleWeightVectorSize; j++)
			_particles[i]._position[j] = weightDistribution(_generator);

		_particles[i]._velocity.assign(particleWeightVectorSize, 0.0f);

		_particles[i]._reward = 0.0f;

		_particles[i]._evaluated = false;
	}

	_prevInputs.assign(_numInputs, 0.0f);
}

void PSOAgent::step(float reward, float dt) {
	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
	std::uniform_real_distribution<float> distAttraction(_minAttraction, _maxAttraction);
	std::uniform_real_distribution<float> distTemperature(-_temperature, _temperature);
	std::uniform_int_distribution<int> distParticle(0, _particles.size() - 1);

	std::vector<float> currentInputs(getNumInputs());

	for (size_t i = 0; i < getNumInputs(); i++)
		currentInputs[i] = _actor.getInput(i);

	_critic.activateLinearOutputLayer();

	float value = _critic.getOutput(0);

	for (size_t i = 0; i < getNumInputs(); i++)
		_critic.setInput(i, _prevInputs[i]);

	_critic.activateLinearOutputLayer();

	float prevValue = _critic.getOutput(0);

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
	float criticTarget = prevValue + _alpha * tdError;

	// Add discounted reward to eligibility traces. Propagate reward downward
	float prevChainValue = criticTarget;

	for (std::list<EligibilityTrace>::iterator it = _eligibilityTraceChain.begin(); it != _eligibilityTraceChain.end(); it++) {
		it->_value = (1.0f - _alpha) * it->_value + _alpha * (it->_reward + _gamma * prevChainValue);
		prevChainValue = it->_value;
	}

	// Track square error for early stopping
	float squareError = 99999.0f;

	for (size_t p = 0; p < _numCriticBackpropPasses && squareError > _criticErrorTolerance; p++) {
		float error;

		// Recalculate sum of square errors
		squareError = 0.0f;

		// Train on new sample
		for (size_t i = 0; i < _prevInputs.size(); i++)
			_critic.setInput(i, _prevInputs[i]);

		_critic.activateLinearOutputLayer();

		error = criticTarget - _critic.getOutput(0);

		squareError += error * error;

		FeedForwardNeuralNetwork::Gradient gradient;
		_critic.getGradientLinearOutputLayer(std::vector<float>(1, criticTarget), gradient);
		_critic.moveAlongGradientMomentum(gradient, _criticAlpha, _criticMomentum);

		// Train on previous samples
		for (std::list<EligibilityTrace>::iterator it = _eligibilityTraceChain.begin(); it != _eligibilityTraceChain.end(); it++) {
			for (size_t i = 0; i < it->_inputs.size(); i++)
				_critic.setInput(i, it->_inputs[i]);

			_critic.activateLinearOutputLayer();

			error = it->_value - _critic.getOutput(0);

			squareError += error * error;

			FeedForwardNeuralNetwork::Gradient gradient;
			_critic.getGradientLinearOutputLayer(std::vector<float>(1, it->_value), gradient);
			_critic.moveAlongGradientMomentum(gradient, _criticAlpha, _criticMomentum);
		}

		// Train on rehearsal samples
		for (size_t i = 0; i < criticRehearsalSamples.size(); i++) {
			for (size_t j = 0; j < _critic.getNumInputs(); j++)
				_critic.setInput(j, criticRehearsalSamples[i]._inputs[j]);

			_critic.activateLinearOutputLayer();

			error = criticRehearsalSamples[i]._outputs[0] - _critic.getOutput(0);

			squareError += error * error;

			FeedForwardNeuralNetwork::Gradient gradient;
			_critic.getGradientLinearOutputLayer(criticRehearsalSamples[i]._outputs, gradient);
			_critic.moveAlongGradientMomentum(gradient, _criticAlpha, _criticMomentum);
		}
	}

	_critic.decayWeights(_criticDecay);

	// Add sample
	EligibilityTrace trace;
	trace._inputs = _prevInputs;
	trace._value = criticTarget;
	trace._reward = reward;

	_eligibilityTraceChain.push_front(trace);

	while (_eligibilityTraceChain.size() > _maxEligibilityTraceChainSize)
		_eligibilityTraceChain.pop_back();

	size_t currentParticleIndex = _particleIndex;

	_actor.setWeightVector(_particles[currentParticleIndex]._position);

	_actor.activate();

	if (_particles[_prevParticleIndex]._evaluated)
		_particles[_prevParticleIndex]._reward += (tdError - _particles[_prevParticleIndex]._reward) * _rewardDecay * dt;
	else {
		_particles[_prevParticleIndex]._reward = reward;
		_particles[_prevParticleIndex]._evaluated = true;
	}

	// Find normalized rewardes
	float minreward = 999999.0f;
	float maxreward = -999999.0f;

	for (size_t i = 0; i < _particles.size(); i++) {
		if (_particles[i]._reward < minreward)
			minreward = _particles[i]._reward;

		if (_particles[i]._reward > maxreward)
			maxreward = _particles[i]._reward;
	}

	std::vector<float> normalizedrewards;

	if (maxreward <= minreward)
		normalizedrewards.assign(_particles.size(), 0.0f);
	else {
		normalizedrewards.resize(_particles.size());

		float offsetMaxrewardInv = 1.0f / (maxreward - minreward);

		// Rescale so that highest reward is at 1
		for (size_t i = 0; i < _particles.size(); i++)
			normalizedrewards[i] = std::powf((_particles[i]._reward - minreward) * offsetMaxrewardInv, _greedExponent);
	}

	// Move all particles
	for (size_t i = 0; i < _particles.size(); i++)
	for (size_t j = 0; j < _particles.size(); j++) {
		if (i == j)
			continue;

		// Calculate distance to this particle
		/*float distance = 0.0f;

		for (size_t k = 0; k < _particles[i]._velocity.size(); k++) {
			float delta = _particles[j]._velocity[k] - _particles[i]._velocity[k];
			distance += delta * delta;
		}

		distance = std::sqrtf(distance);

		float attractionDistanceScalar = 1.0f / (distance + 1.0f);*/

		for (size_t k = 0; k < _particles[i]._velocity.size(); k++) {
			_particles[i]._velocity[k] += (-_particleVelocityDecay * _particles[i]._velocity[k] + distAttraction(_generator) * (normalizedrewards[j] - normalizedrewards[i] + _attractionOffset) * (_particles[j]._position[k] - _particles[i]._position[k]) / (_particleMassResistanceInv + 1.0f - normalizedrewards[i]) +
				distTemperature(_generator)) * dt;

			// Bounds
			_particles[i]._velocity[k] = std::min(_maxVelocity, std::max(_minVelocity, _particles[i]._velocity[k]));

			_particles[i]._position[k] += _particles[i]._velocity[k] * dt;

			// Bounds
			_particles[i]._position[k] = std::min(_maxWeight, std::max(_minWeight, _particles[i]._position[k]));
		}
	}

	if (dist01(_generator) < _teleportChance) {
		std::uniform_real_distribution<float> teleportDist(-_maxTeleportPerturbation, _maxTeleportPerturbation);

		for (size_t i = 0; i < _particles[currentParticleIndex]._position.size(); i++) {
			_particles[currentParticleIndex]._position[i] += teleportDist(_generator);

			// Bounds
			_particles[currentParticleIndex]._position[i] = std::min(_maxWeight, std::max(_minWeight, _particles[currentParticleIndex]._position[i]));

			_particles[currentParticleIndex]._velocity[i] = 0.0f;
		}
	}

	_prevParticleIndex = currentParticleIndex;

	_particleIndex = (_particleIndex + 1) % _particles.size();

	_prevInputs = currentInputs;
}