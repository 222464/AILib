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
#include <random>
#include <assert.h>

namespace nn {
	class QAgent
	{
	public:
		struct IOSet {
			std::vector<float> _inputs;
			float _output;
		};
	private:
		float _qDegradeAccum;

		size_t _numInputs; // Values after _numInputs are action values

		std::vector<float> _prevInputs;

		struct QInputOutputTuple {
			std::vector<float> _inputs;
			float _output;
		};

		std::list<QInputOutputTuple> _replayBuffer;

		std::vector<float> _outputBuffer;

		std::vector<float> _outputVelocities;
		std::vector<float> _outputOffsets;

		void findMaxQGradient();

	public:
		std::mt19937 _generator;

		FeedForwardNeuralNetwork _qNetwork; // Q value storage

		float _alpha; // Learning rate
		float _gamma; // Discount factor
		float _maxFitness; // Highest fitness value possible
		float _qUpdateAlpha; // Alpha of Q backprop updates
		float _findMaxQAlpha; // Alpha of max Q search
		float _findMaxQMometum; // Momentum for max Q search
		float _momentum; // Momentum for Q backprop
		float _randInitOutputRange; // Random range that the output is set to before backpropagating to inputs

		size_t _findMaxQPasses; // Number of backpropagation passes used to find action with maximum Q
		size_t _findMaxQSamples; // Number of times passes are repeated to find global optimum
		size_t _numBackpropPasses; // Number of backpropagation passes used to update Q

		size_t _numPseudoRehearsalSamples; // Number of pseudo-rehearsal samples used to maintain Q 
		float _pseudoRehearsalSampleStdDev; // Standard deviation of the pseudorehearsal sample input selection (normal distribution)
		float _pseudoRehearsalSampleMean; // Average value around which random pseudorehearsal values are centered

		float _weightDecay; // Decay of the weights over time in the Q network

		float _qDecay; // Decay of Q values over time to prevent state looping

		float _stdDev;

		QAgent();

		// Create an agent with randomly initialized weights
		void createRandom(size_t numInputs, size_t numOutputs,
			size_t numHiddenLayers, size_t numNeuronsPerHiddenLayer,
			float minWeight, float maxWeight, unsigned long seed);

		void writeToStream(std::ostream &stream);
		void readFromStream(std::istream &stream);

		void setInput(size_t i, float value) {
			assert(i < getNumInputs());
			_qNetwork.setInput(i, value);
		}

		// Get the output action of the agent
		float getOutput(size_t i) const {
			return _outputBuffer[i];
		}

		void step(float fitness);

		size_t getNumInputs() const {
			return _numInputs;
		}

		size_t getNumOutputs() const {
			return _outputBuffer.size();
		}
	};
}
