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

#include <nn/Sensor.h>
#include <nn/BrownianPerturbation.h>
#include <vector>

namespace nn {
	class Neuron : public Sensor {
	public:
		struct Synapse {
			Sensor* _pInput;
			float _weight;
			float _trace;
			float _traceAdditional;

			Synapse()
				: _trace(0.0f), _traceAdditional(0.0f)
			{}
		};

		static float sigmoid(float x) {
			return 1.0f / (1.0f + std::expf(-x));
		}

		static float plasticityFunc(float x) {
			return -std::sin(x * 0.159154f);
		}

		static float scaledSigmoid(float x) {
			return 2.0f / (1.0f + std::expf(-x)) - 1.0f;
		}

		static float logit(float x) {
			return std::log(x / (1.0f - x));
		}

		std::vector<Synapse> _synapses;

		float _bias;
		float _biasTrace;
		float _biasTraceAdditional;

		Neuron()
			: _biasTrace(0.0f), _biasTraceAdditional(0.0f)
		{}

		// Inherited from Sensor
		void activate(float activationMultiplier, float outputTraceDecay);

		void activateTraceless(float activationMultiplier);

		void activateAndReinforce(float activationMultiplier, float outputTraceDecay, float weightTraceDecay, float error);
		void activateAndReinforceTraceless(float activationMultiplier, float error);
		void activateLinear(float activationMultiplier);

		void activateArp(float activationMultiplier, float outputTraceDecay, std::mt19937 &generator);

		// Stand-alone reinforce
		void reinforce(float error, float weightTraceDecay);
		void reinforceTraceless(float error);

		// Associative reward-penalty (stochastic)
		void reinforceArp(float reward, float alpha, float lambda);
		void reinforceArpWithTraces(float reward, float alpha, float lambda, float weightTraceDecay);
		void reinforceArpMomentum(float reward, float alpha, float lambda, float momentum);
	};
}
