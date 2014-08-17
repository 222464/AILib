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

#include <assert.h>

namespace lstm {
	class LSTMGQ {
	private:
		LSTMG _qNet;

		std::vector<float> _inputs;
		std::vector<float> _outputs;

	public:
		void createRandom(size_t numInputs, size_t numOutputs,
			size_t numMemoryLayers, size_t numMemoryCellsPerLayer,
			size_t numHiddenLayers, size_t numNeuronsPerHiddenLayer,
			float minWeight, float maxWeight, std::mt19937 &randomGenerator);

		void step(float reward, float gamma, float alpha, float deriveQAlpha, float eligibilityDecay, size_t policyDeriveIterations, float outputPerturbationStdDev, std::mt19937 &randomGenerator);

		void setInput(size_t index, float value) {
			_inputs[index] = value;
		}

		float getOutput(size_t index) const {
			return _outputs[index];
		}
	};
}