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

#include <lstm/LSTMG.h>
#include <deep/FA.h>

namespace lstm {
	class LSTMActorCritic {
	private:
		LSTMG _actor;
		LSTMG _critic;

		std::vector<float> _currentInputs;
		std::vector<float> _currentOutputs;
		std::vector<float> _prevOutputs;

		std::vector<float> _outputOffsets;

		float _prevValue;
		float _error;

		float _variance;

	public:
		LSTMActorCritic()
			: _prevValue(0.0f), _error(0.0f), _variance(0.0f)
		{}

		void createRandom(int numInputs, int numOutputs,
			int actorNumHiddenLayers, int actorHiddenLayerSize,
			int numActorMemoryCells, int numActorMemoryCellLayers,
			int numCriticHiddenLayers, int criticHiddenLayerSize,
			int numCriticMemoryCells, int numCriticMemoryCellLayers,
			float minWeight, float maxWeight, std::mt19937 &generator);

		void step(float reward, float qAlpha, float actorAlpha, float breakRate, float perturbationStdDev, float criticAlpha, float gamma, float eligibiltyDecayActor, float eligibiltyDecayCritic, float varianceDecay, float actorMomentum, float criticMomentum, float outputOffsetDecay, std::mt19937 &generator);
		void step(float reward, float qAlpha, float actorAlpha, float breakRate, float perturbationStdDev, float criticAlpha, float gamma, float eligibiltyDecayActor, float eligibiltyDecayCritic, float varianceDecay, float actorMomentum, float criticMomentum, float outputOffsetDecay, float hebbianAlphaActor, float hebbianAlphaCritic, std::mt19937 &generator);

		void setInput(size_t index, float value) {
			_currentInputs[index] = value;
		}

		float getInput(size_t index) const {
			return _currentInputs[index];
		}

		float getOutput(size_t index) const {
			return _currentOutputs[index];
		}
	};
}