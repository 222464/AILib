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

#include <deep/EQNAC.h>

#include <algorithm>

#include <iostream>

using namespace deep;

EQNAC::EQNAC()
: _baseline(0.0f)
{}

void EQNAC::createRandom(int numState, int numAction, int numHidden, float minWeight, float maxWeight, std::mt19937 &generator) {
	_numState = numState;
	_numAction = numAction;
	
	_rbm.createRandom(numState, numHidden, minWeight, maxWeight, generator);

	_outputNodes.resize(numAction);

	std::uniform_real_distribution<float> weightDist(minWeight, maxWeight);

	for (int i = 0; i < numAction; i++) {
		_outputNodes[i]._bias._weight = weightDist(generator);

		_outputNodes[i]._connections.resize(numState + numHidden);

		for (int j = 0; j < _outputNodes[i]._connections.size(); j++)
			_outputNodes[i]._connections[j]._weight = weightDist(generator);
	}

	_qNode._bias._weight = weightDist(generator);

	_qNode._connections.resize(numState + numHidden);

	for (int i = 0; i < _qNode._connections.size(); i++)
		_qNode._connections[i]._weight = weightDist(generator);
}

void EQNAC::step(float reward, float rbmAlpha, std::mt19937 &generator) {
	//_rbm.activate(
}