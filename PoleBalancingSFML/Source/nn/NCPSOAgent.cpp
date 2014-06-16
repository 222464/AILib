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

#include <nn/NCPSOAgent.h>

#include <algorithm>

#include <iostream>

using namespace nn;

NCPSOAgent::NCPSOAgent()
: _minWeight(-4.0f), _maxWeight(4.0f),
_minVelocity(-0.2f), _maxVelocity(0.2f),
_alpha(0.01f), _gamma(0.99f),
_pFactor(0.1f), _gFactor(0.1f),
_particleVelocityDecay(0.001f),
_rewardDecay(0.0001f),
_bestDecay(0.0001f),
_numChainUpdateParticles(100),
_prevParticleIndex(0),
_particleIndex(0)
{}

void NCPSOAgent::createRandom(size_t numInputs, size_t numOutputs,
	size_t actorNumHiddenLayers, size_t actorNumNeuronsPerHiddenLayer,
	size_t numParticles, float initMinWeight, float initMaxWeight,
	unsigned long seed)
{
	_numInputs = numInputs;
	_numOutputs = numOutputs;

	_actor.createRandom(_numInputs, _numOutputs, actorNumHiddenLayers, actorNumNeuronsPerHiddenLayer, _minWeight, _maxWeight, seed + 1);

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

		_particles[i]._pBest = 0.0f;
		_particles[i]._pBestPosition = _particles[i]._position;
	}
}

void NCPSOAgent::step(float reward, float dt) {
	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

	size_t currentParticleIndex = _particleIndex;

	_actor.setWeightVector(_particles[currentParticleIndex]._position);

	_actor.activateLinearOutputLayer();

	// Update past particles with new reward
	float gamma = 1.0f;

	int particleIndex = static_cast<int>(_prevParticleIndex);

	for (size_t i = 0; i < _numChainUpdateParticles; i++) {
		_particles[particleIndex]._reward += -_rewardDecay * _particles[particleIndex]._reward + gamma * reward;

		particleIndex--;

		if (particleIndex < 0)
			particleIndex = static_cast<int>(_particles.size()) - 1;

		gamma *= _gamma;
	}

	// Decay bests
	_particles[_prevParticleIndex]._pBest += (_particles[_prevParticleIndex]._reward - _particles[_prevParticleIndex]._pBest) * _bestDecay * dt;

	// Update pBest
	if (_particles[_prevParticleIndex]._reward > _particles[_prevParticleIndex]._pBest) {
		_particles[_prevParticleIndex]._pBest = _particles[_prevParticleIndex]._reward;
		_particles[_prevParticleIndex]._pBestPosition = _particles[_prevParticleIndex]._position;
	}

	// Find best
	size_t gBestIndex = 0;

	for (size_t i = 1; i < _particles.size(); i++)
	if (_particles[i]._reward > _particles[gBestIndex]._reward)
		gBestIndex = i;

	// Move this particle
	for (size_t ci = 0; ci < _particles[_prevParticleIndex]._position.size(); ci++) {
		_particles[_prevParticleIndex]._velocity[ci] += (-_particleVelocityDecay * _particles[_prevParticleIndex]._velocity[ci] +
			_pFactor * dist01(_generator) * (_particles[_prevParticleIndex]._pBestPosition[ci] - _particles[_prevParticleIndex]._position[ci]) +
			_gFactor * dist01(_generator) * (_particles[gBestIndex]._position[ci] - _particles[_prevParticleIndex]._position[ci])) * dt;

		_particles[_prevParticleIndex]._position[ci] += _particles[_prevParticleIndex]._velocity[ci] * dt;
	}

	_prevParticleIndex = currentParticleIndex;

	_particleIndex = (_particleIndex + 1) % _particles.size();
}