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

#include <raahn/AutoEncoder.h>
#include <raahn/HebbianLearner.h>

namespace raahn {
	class RAAHN {
	private:
		AutoEncoder _autoEncoder;

		HebbianLearner _hebbianLearner;

		size_t _numOutputs;

		std::vector<float> _inputs;
		std::vector<float> _features;
		std::vector<float> _hebbianInputs;
		std::vector<float> _outputs;

	public:
		void createRandom(size_t numInputs, size_t numFeatures, size_t numOutputs,
			size_t numRecurrentConnections, size_t numHebbianHidden, size_t numNeuronsPerHebbianHidden,
			float minWeight, float maxWeight, std::mt19937 &generator);
		void createFromParents(const RAAHN &parent1, const RAAHN &parent2, float averageChance, std::mt19937 &generator);
		void mutate(float perturbationChance, float perturbationStdDev, std::mt19937 &generator);

		void update(float autoEncoderAlpha, float modulation, float traceDecay, float outputDecay, float breakRate, std::mt19937 &generator);

		void setInput(size_t index, float value) {
			_inputs[index] = value;
		}

		float getOutput(size_t index) const {
			return _outputs[index];
		}

		size_t getNumInputs() const {
			return _inputs.size();
		}

		size_t getNumOutputs() const {
			return _numOutputs;
		}
	};
}