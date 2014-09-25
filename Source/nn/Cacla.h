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
#include <nn/BrownianPerturbation.h>
#include <list>
#include <assert.h>
#include <iostream>

namespace nn {
	class Cacla {
	public:
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
		std::vector<float> _prevInputs;
		std::vector<float> _prevPrevInputs;
		std::vector<float> _outputs;
		std::vector<float> _prevOutputs;

		float _prevReward;
		float _prevValue;

		float _variance;

		std::list<EligibilityTrace> _eligibilityTraceChain;

	public:
		std::mt19937 _generator;

		float _outputOffsetScalar;
		float _outputOffsetAbsErrorScalar;
		float _reverseOffsetErrorScalar;
		std::vector<BrownianPerturbation> _outputOffsets;

		FeedForwardNeuralNetwork _actor, _critic;

		float _actorAlpha; // Learning rate of actor
		float _criticAlpha; // Learning rate of critic
		float _actorDecay; // Decay of actor weights
		float _criticDecay; // Decay of critic weights
		float _lambda; // Value function movement rate
		float _gamma; // Discount factor

		float _actorErrorTolerance;
		float _criticErrorTolerance;

		size_t _numCriticBackpropPasses; // Number of passes to update the value function with backprop
		size_t _numActorBackpropPasses;

		float _actorMomentum;
		float _criticMomentum;

		float _varianceDecay; // Rate at which the variance value decays to current value

		size_t _numPseudoRehearsalSamplesActor; // Number of samples used for performing pseudorehearsal to avoid catastrophic forgetting (actor)
		size_t _numPseudoRehearsalSamplesCritic; // Number of samples used for performing pseudorehearsal to avoid catastrophic forgetting (critic)
		float _pseudoRehearsalSampleStdDev; // Standard deviation of the pseudorehearsal sample input selection (normal distribution)
		float _pseudoRehearsalSampleMean; // Average value around which random pseudorehearsal values are centered

		size_t _maxEligibilityTraceChainSize;

		float _actorRehearseNewSampleChance;

		Cacla();

		// Create an agent with randomly initialized weights
		void createRandom(size_t numInputs, size_t numOutputs,
			size_t numActorHiddenLayers, size_t numActorNeuronsPerHiddenLayer,
			size_t numCriticHiddenLayers, size_t numCriticNeuronsPerHiddenLayer,
			const BrownianPerturbation &perturbation,
			float minWeight, float maxWeight,
			unsigned long seed);

		// Steps agent to a new state in the POMDP
		void step(float reward, float dt);

		void writeToStream(std::ostream &stream);
		void readFromStream(std::istream &stream);

		// Get a (previously set) input
		float getInput(size_t i) const {
			return _actor.getInput(i);
		}

		// Set an input to the agent
		void setInput(size_t i, float value) {
			_actor.setInput(i, value);
			_critic.setInput(i, value);
		}

		// Get the output action of the agent
		float getOutput(size_t i) const {
			return _outputs[i];
		}

		size_t getNumInputs() const {
			return _actor.getNumInputs();
		}

		size_t getNumOutputs() const {
			return _actor.getNumOutputs();
		}
	};
}