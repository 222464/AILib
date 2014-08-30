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

#include <nn/SOM.h>
#include <nn/BrownianPerturbation.h>

#include <list>
#include <random>

namespace nn {
	class SOMQAgent {
	public:
		struct StateActionTuple {
			std::vector<float> _input;
			std::vector<float> _output;
		};
	private:
		size_t _numInputs;
		size_t _numOutputs;
		size_t _numStates;
		size_t _numActions;

		std::vector<float> _input;
		std::vector<float> _output;

		std::vector<float> _prevInput;
		std::vector<float> _prevOutput;
		std::vector<float> _prevExploratoryOutput;

		std::vector<bool> _stateOnlyMask;
		std::vector<bool> _stateActionMask;
		std::vector<bool> _stateRewardMask;
		std::vector<bool> _stateActionRewardMask;

		float _prevQ;

	public:
		SOM _stateActionSOM;

		SOMQAgent();

		void createRandom(size_t numInputs, size_t numOutputs, size_t dimensions, size_t dimensionSize, const nn::BrownianPerturbation &perturbation, float minWeight, float maxWeight, std::mt19937 &generator);

		void step(float fitness, float alpha, float gamma, float traceDecay, float breakRate, float dt, std::mt19937 &generator);

		void setInput(size_t i, float value) {
			_input[i] = value;
		}

		float getInput(size_t i) const {
			return _input[i];
		}

		float getOutput(size_t i) const {
			return _output[i];
		}
	};
}