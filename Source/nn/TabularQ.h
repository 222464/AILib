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

#pragma once

#include <vector>
#include <random>

namespace nn {
	class TabularQ {
	private:
		size_t _numStates;
		size_t _numActions;

		std::vector<float> _q; // Dimension: state x action

		size_t _state;
		size_t _action;

		size_t _prevState;
		size_t _prevAction;

		float getQ(size_t s, size_t a) const {
			return _q[s + _numStates * a];
		}

		void setQ(size_t s, size_t a, float value) {
			_q[s + _numStates * a] = value;
		}

	public:
		std::mt19937 _generator;

		float _gamma;
		float _alpha;

		float _randomActionChance;

		TabularQ()
			: _prevState(0), _prevAction(0),
			_gamma(0.9f), _alpha(0.5f),
			_randomActionChance(0.2f)
		{}

		void create(size_t numStates, size_t numActions, unsigned long seed);

		void step(float fitness);

		void setState(size_t state) {
			_state = state;
		}

		size_t getAction() const {
			return _action;
		}

		size_t getNumStates() const {
			return _numStates;
		}

		size_t getNumActions() const {
			return _numActions;
		}
	};
}
