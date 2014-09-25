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

#include <nn/FeedForwardNeuralNetwork.h>
#include <list>

namespace nn {
	class PSOAgent
	{
	public:
		struct Particle {
			std::vector<float> _position; // Weight vector of action generator network
			std::vector<float> _velocity;

			float _reward;

			bool _evaluated;
		};

		struct IOSet {
			std::vector<float> _inputs;
			std::vector<float> _outputs;
		};

		struct EligibilityTrace {
			std::vector<float> _inputs;
			float _value;
			float _reward;
		};

	private:
		std::vector<Particle> _particles;

		size_t _numInputs, _numOutputs;

		size_t _prevParticleIndex;

		size_t _particleIndex;

		std::list<EligibilityTrace> _eligibilityTraceChain;

		std::vector<float> _prevInputs;

	public:
		nn::FeedForwardNeuralNetwork _critic; // Input to critic is state
		nn::FeedForwardNeuralNetwork _actor; // Inputs to actor is state

		float _minWeight, _maxWeight;
		float _minVelocity, _maxVelocity;

		std::mt19937 _generator;

		float _alpha;
		float _gamma;

		float _minAttraction, _maxAttraction;

		float _particleVelocityDecay;

		float _rewardDecay;

		float _temperature;

		size_t _numCriticBackpropPasses;
		size_t _maxEligibilityTraceChainSize;
		
		float _criticAlpha;
		float _criticMomentum;

		size_t _numPseudoRehearsalSamplesCritic; // Number of samples used for performing pseudorehearsal to avoid catastrophic forgetting
		float _pseudoRehearsalSampleStdDev; // Standard deviation of the pseudorehearsal sample input selection (normal distribution)
		float _pseudoRehearsalSampleMean; // Average value around which random pseudorehearsal values are centered

		float _criticErrorTolerance;
		float _criticDecay;

		float _teleportChance; // Chance that a particle is randomly reinitialized
		float _maxTeleportPerturbation; // Max teleportation coordinate distance

		float _greedExponent;
		float _particleMassResistanceInv;
		float _attractionOffset;

		PSOAgent();

		void createRandom(size_t numInputs, size_t numOutputs,
			size_t actorNumHiddenLayers, size_t actorNumNeuronsPerHiddenLayer,
			size_t criticNumHiddenLayers, size_t criticNumNeuronsPerHiddenLayer,
			size_t numParticles, float initMinWeight, float initMaxWeight,
			unsigned long seed);

		void step(float reward, float dt = 1.0f);

		void setInput(size_t index, float value) {
			_actor.setInput(index, value);
			_critic.setInput(index, value);
		}

		float getOutput(size_t index) const {
			return _actor.getOutput(index);
		}

		size_t getNumInputs() const {
			return _numInputs;
		}

		size_t getNumOutputs() const {
			return _numOutputs;
		}
	};
}