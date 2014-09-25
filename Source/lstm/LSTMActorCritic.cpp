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

#include <lstm/LSTMActorCritic.h>

#include <algorithm>

#include <iostream>

using namespace lstm;

void LSTMActorCritic::createRandom(int numInputs, int numOutputs,
	int actorNumHiddenLayers, int actorHiddenLayerSize,
	int numActorMemoryCells, int numActorMemoryCellLayers,
	int numCriticHiddenLayers, int criticHiddenLayerSize,
	int numCriticMemoryCells, int numCriticMemoryCellLayers,
	float minWeight, float maxWeight, std::mt19937 &generator)
{
	_actor.createRandomLayered(numInputs, numOutputs, numActorMemoryCellLayers, numActorMemoryCells, actorNumHiddenLayers, actorHiddenLayerSize, minWeight, maxWeight, generator);
	_critic.createRandomLayered(numInputs, 1, numCriticMemoryCellLayers, numCriticMemoryCells, numCriticHiddenLayers, criticHiddenLayerSize, minWeight, maxWeight, generator);

	_currentInputs.assign(numInputs, 0.0f);
	_currentOutputs.assign(numOutputs, 0.0f);
	_outputOffsets.assign(numOutputs, 0.0f);

	for (size_t i = 0; i < _currentInputs.size(); i++) {
		_actor.setInput(i, _currentInputs[i]);
		_critic.setInput(i, _currentInputs[i]);
	}

	_actor.step(true);
	_critic.step(true);
}

void LSTMActorCritic::step(float reward, float qAlpha, float actorAlpha, float breakRate, float perturbationStdDev, float criticAlpha, float gamma, float eligibiltyDecayActor, float eligibiltyDecayCritic, float varianceDecay, float actorMomentum, float criticMomentum, float outputOffsetDecay, std::mt19937 &generator) {
	std::vector<float> prevInputs(_currentInputs.size());

	for (size_t i = 0; i < prevInputs.size(); i++)
		prevInputs[i] = _critic.getInput(i);

	LSTMG nextCritic = _critic;
	
	for (size_t i = 0; i < _currentInputs.size(); i++)
		nextCritic.setInput(i, _currentInputs[i]);

	nextCritic.step(true);

	float q = reward + gamma * nextCritic.getOutput(0);

	_error = q - _critic.getOutput(0);

	//std::cout << _outputOffsets[0] << std::endl;

	_critic.getDeltas(std::vector<float>(1, _critic.getOutput(0) + _error * qAlpha), eligibiltyDecayCritic, true);
	_critic.moveAlongDeltas(criticAlpha, criticMomentum);

	for (size_t i = 0; i < _currentInputs.size(); i++)
		_critic.setInput(i, _currentInputs[i]);

	_critic.step(true);

	if (_error > _variance) {
		//std::cout << "T";
		_actor.moveAlongDeltas(actorAlpha, actorMomentum);
	}

	for (size_t i = 0; i < _currentInputs.size(); i++)
		_actor.setInput(i, _currentInputs[i]);

	_actor.step(true);

	// Decay output offsets
	for (size_t i = 0; i < _outputOffsets.size(); i++)
		_outputOffsets[i] *= outputOffsetDecay;

	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
	std::normal_distribution<float> distPert(0.0f, perturbationStdDev);

	for (size_t i = 0; i < _currentOutputs.size(); i++) {
		if (dist01(generator) < breakRate) {
			_currentOutputs[i] = 2.0f * dist01(generator) - 1.0f;

			_outputOffsets[i] = _currentOutputs[i] - _outputOffsets[i];
		}
		else
			_currentOutputs[i] = std::min(1.0f, std::max(-1.0f, std::min(1.0f, std::max(-1.0f, _actor.getOutput(i))) + distPert(generator) + _outputOffsets[i]));
	}

	_actor.getDeltas(_currentOutputs, eligibiltyDecayActor, true);

	_variance += (_error - _variance) * varianceDecay;
}

void LSTMActorCritic::step(float reward, float qAlpha, float actorAlpha, float breakRate, float perturbationStdDev, float criticAlpha, float gamma, float eligibiltyDecayActor, float eligibiltyDecayCritic, float varianceDecay, float actorMomentum, float criticMomentum, float outputOffsetDecay, float hebbianAlphaActor, float hebbianAlphaCritic, std::mt19937 &generator) {
	std::vector<float> prevInputs(_currentInputs.size());

	for (size_t i = 0; i < prevInputs.size(); i++)
		prevInputs[i] = _critic.getInput(i);

	LSTMG nextCritic = _critic;

	for (size_t i = 0; i < _currentInputs.size(); i++)
		nextCritic.setInput(i, _currentInputs[i]);

	nextCritic.step(true);

	float q = reward + gamma * nextCritic.getOutput(0);

	_error = q - _critic.getOutput(0);

	std::cout << _outputOffsets[0] << std::endl;

	_critic.getDeltas(std::vector<float>(1, _critic.getOutput(0) + _error * qAlpha), eligibiltyDecayCritic, true);
	_critic.moveAlongDeltasAndHebbian(criticAlpha, hebbianAlphaCritic, criticMomentum);

	for (size_t i = 0; i < _currentInputs.size(); i++)
		_critic.setInput(i, _currentInputs[i]);

	_critic.step(true);

	if (_error > _variance) {
		std::cout << "T";
		_actor.moveAlongDeltasAndHebbian(actorAlpha, hebbianAlphaActor, actorMomentum);
	}

	for (size_t i = 0; i < _currentInputs.size(); i++)
		_actor.setInput(i, _currentInputs[i]);

	_actor.step(true);

	// Decay output offsets
	for (size_t i = 0; i < _outputOffsets.size(); i++)
		_outputOffsets[i] *= outputOffsetDecay;

	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
	std::normal_distribution<float> distPert(0.0f, perturbationStdDev);

	for (size_t i = 0; i < _currentOutputs.size(); i++) {
		if (dist01(generator) < breakRate) {
			_currentOutputs[i] = 2.0f * dist01(generator) - 1.0f;

			_outputOffsets[i] = _currentOutputs[i] - _outputOffsets[i];
		}
		else
			_currentOutputs[i] = std::min(1.0f, std::max(-1.0f, std::min(1.0f, std::max(-1.0f, _actor.getOutput(i))) + distPert(generator) + _outputOffsets[i]));
	}

	_actor.getDeltas(_currentOutputs, eligibiltyDecayActor, true);

	_variance += (_error - _variance) * varianceDecay;
}