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

#include <nn/TabularQ.h>

using namespace nn;

void TabularQ::create(size_t numStates, size_t numActions, unsigned long seed) {
	_numStates = numStates;
	_numActions = numActions;
	
	_q.assign(_numStates * _numActions, 0.0f);

	_generator.seed(seed);
}

void TabularQ::step(float fitness) {
	size_t maxQAction = 0;

	for (size_t a = 1; a < _numActions; a++)
	if (getQ(_state, a) > getQ(_state, maxQAction))
		maxQAction = a;

	float newPrevQ = getQ(_prevState, _prevAction) + _alpha * (fitness + _gamma * getQ(_state, maxQAction) - getQ(_prevState, _prevAction));
	
	setQ(_prevState, _prevAction, newPrevQ);

	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
	std::uniform_int_distribution<int> actionDist(0, _numActions - 1);

	_prevAction = _action;
	_action = dist01(_generator) < _randomActionChance ? actionDist(_generator) : maxQAction;

	_prevState = _state;
}