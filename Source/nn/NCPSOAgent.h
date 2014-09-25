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
	class NCPSOAgent
	{
	public:
		struct Particle {
			std::vector<float> _position; // Weight vector of action generator network
			std::vector<float> _velocity;

			float _reward;
			float _pBest;
			std::vector<float> _pBestPosition;
		};

	private:
		std::vector<Particle> _particles;

		size_t _numInputs, _numOutputs;

		size_t _prevParticleIndex;

		size_t _particleIndex;

	public:
		nn::FeedForwardNeuralNetwork _actor; // Inputs to actor is state

		float _minWeight, _maxWeight;
		float _minVelocity, _maxVelocity;

		std::mt19937 _generator;

		float _alpha;
		float _gamma;

		float _pFactor, _gFactor;

		float _particleVelocityDecay;

		float _rewardDecay;
		float _bestDecay;

		size_t _numChainUpdateParticles;

		NCPSOAgent();

		void createRandom(size_t numInputs, size_t numOutputs,
			size_t actorNumHiddenLayers, size_t actorNumNeuronsPerHiddenLayer,
			size_t numParticles, float initMinWeight, float initMaxWeight,
			unsigned long seed);

		void step(float reward, float dt = 1.0f);

		void setInput(size_t index, float value) {
			_actor.setInput(index, value);
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