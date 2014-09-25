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

namespace deep {
	class DAutoEncoder {
	private:
		struct Neuron {
			float _bias;
			std::vector<float> _weights;
		};

		std::vector<Neuron> _hidden;
		std::vector<float> _inputBiases;
		std::vector<float> _inputErrorBuffer;

		float crossoverChooseWeight(float w1, float w2, float averageChance, std::mt19937 &generator);

	public:
		void createRandom(size_t numInputs, size_t numOutputs, float minWeight, float maxWeight, std::mt19937 &generator);
		void createFromParents(const DAutoEncoder &parent1, const DAutoEncoder &parent2, float averageChance, std::mt19937 &generator);
		void mutate(float perturbationChance, float perturbationStdDev, std::mt19937 &generator);

		void update(const std::vector<float> &inputs, std::vector<float> &outputs, float alpha);

		static float sigmoid(float x) {
			return 1.0f / (1.0f + std::exp(-x));
		}

		const std::vector<float> &getInputErrorBuffer() const {
			return _inputErrorBuffer;
		}

		size_t getNumInputs() const {
			return _inputBiases.size();
		}

		size_t getNumOutputs() const {
			return _hidden.size();
		}
	};
}