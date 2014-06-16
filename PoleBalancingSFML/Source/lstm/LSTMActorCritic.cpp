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

	for (size_t i = 0; i < _currentInputs.size(); i++) {
		_actor.setInput(i, _currentInputs[i]);
		_critic.setInput(i, _currentInputs[i]);
	}

	_actor.step(true);
	_critic.step(true);
}

void LSTMActorCritic::step(float reward, float actorAlpha, float offsetStdDev, float criticAlpha, float gamma, float eligibiltyDecay, std::mt19937 &generator) {
	LSTMG nextCritic = _critic;

	for (size_t i = 0; i < _currentInputs.size(); i++)
		nextCritic.setInput(i, _currentInputs[i]);

	//for (size_t i = 0; i < _currentOutputs.size(); i++)
	//	nextCritic.setInput(i + _currentInputs.size(), _actor.getOutput(i));

	nextCritic.step(true);

	float q = reward + gamma * nextCritic.getOutput(0);

	_error = q - _prevValue;

	_critic.moveAlongDeltas(criticAlpha * _error);

	for (size_t i = 0; i < _currentInputs.size(); i++)
		_critic.setInput(i, _currentInputs[i]);

	//for (size_t i = 0; i < _currentOutputs.size(); i++)
	//	_critic.setInput(i + _currentInputs.size(), _actor.getOutput(i));

	_critic.step(true);
	_critic.getDeltas(std::vector<float>(1, _critic.getOutput(0) + 1.0f), eligibiltyDecay, true);
	
	_prevValue = nextCritic.getOutput(0);

	if (_error > -0.01f)
		_actor.moveAlongDeltas(actorAlpha);

	for (size_t i = 0; i < _currentInputs.size(); i++)
		_actor.setInput(i, _currentInputs[i]);

	_actor.step(true);

	std::normal_distribution<float> distOffset(0.0f, offsetStdDev);

	for (size_t i = 0; i < _currentOutputs.size(); i++)
		_currentOutputs[i] = std::min(1.0f, std::max(-1.0f, std::min(1.0f, std::max(-1.0f, _actor.getOutput(i))) + distOffset(generator)));

	_actor.getDeltas(_currentOutputs, eligibiltyDecay, true);

	std::cout << _prevValue << " " << _error << std::endl;
}