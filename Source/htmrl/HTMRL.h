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

#include <htm/Region.h>

#include <deep/FA.h>

#include <algorithm>
#include <list>

#include <assert.h>

namespace htmrl {
	float defaultBoostFunction(float active, float minimum);

	class HTMRL {
	public:
		struct RegionDesc {
			// Init
			int _regionWidth;
			int _regionHeight;
			int _columnSize;
			int _connectionRadius;
			float _initInhibitionRadius;
			int _initNumSegments;
			float _permanenceDistanceBias;
			float _permanenceDistanceFalloff;
			float _permanenceBiasFloor;
			float _connectionPermanenceTarget;
			float _connectionPermanenceStdDev;

			// Spatial pooling
			float _minPermanence;
			float _minOverlap;
			int _desiredLocalActivity;
			float _spatialPermanenceIncrease;
			float _spatialPermanenceDecrease;
			float _minDutyCycleRatio;
			float _activeDutyCycleDecay;
			float _overlapDutyCycleDecay;
			float _subOverlapPermanenceIncrease;
			std::function<float(float, float)> _boostFunction;

			// Temporal pooling
			int _learningRadius;
			int _minLearningThreshold;
			int _activationThreshold;
			int _newNumConnections;
			float _temporalPermanenceIncrease;
			float _temporalPermanenceDecrease;
			float _newConnectionPermanence;

			int _maxSteps;

			RegionDesc()
				: _regionWidth(32),
				_regionHeight(32),
				_columnSize(4),
				_connectionRadius(6),
				_initInhibitionRadius(6.0f),
				_initNumSegments(0),
				_permanenceDistanceBias(0.2f),
				_permanenceDistanceFalloff(2.0f),
				_permanenceBiasFloor(-0.1f),
				_connectionPermanenceTarget(0.3f),
				_connectionPermanenceStdDev(0.1f),
				_minPermanence(0.3f),
				_minOverlap(3.0f),
				_desiredLocalActivity(18),
				_spatialPermanenceIncrease(0.04f),
				_spatialPermanenceDecrease(0.03f),
				_minDutyCycleRatio(0.01f),
				_activeDutyCycleDecay(0.01f),
				_overlapDutyCycleDecay(0.01f),
				_subOverlapPermanenceIncrease(0.02f),
				_boostFunction(std::bind(defaultBoostFunction, std::placeholders::_1, std::placeholders::_2)),
				_learningRadius(4),
				_minLearningThreshold(1),
				_activationThreshold(8),
				_newNumConnections(32),
				_temporalPermanenceIncrease(0.03f),
				_temporalPermanenceDecrease(0.025f),
				_newConnectionPermanence(0.31f),
				_maxSteps(3)
			{}
		};

		struct ReplaySample {
			std::vector<bool> _actorInputsb;
			std::vector<float> _actorOutputsExploratory;
			float _criticOutput;
			float _reward;

			float _optimalQ;

			std::vector<float> _prevDAction;
		};

		static float sigmoid(float x) {
			return 1.0f / (1.0f + std::exp(-x));
		}

	private:
		int _inputWidth;
		int _inputHeight;
		int _inputDotsWidth;
		int _inputDotsHeight;
		int _inputMax;

		std::vector<float> _inputf;
		std::vector<bool> _inputb;

		std::vector<bool> _prevLayerInputb;

		std::vector<RegionDesc> _regionDescs;
		std::vector<htm::Region> _regions;

		std::vector<float> _outputs;
		std::vector<float> _exploratoryOutputs;

		std::vector<float> _prevOutputs;
		std::vector<float> _prevExploratoryOutputs;

		deep::FA _actor;
		deep::FA _critic;

		float _prevMaxQ;
		float _prevValue;

		std::list<ReplaySample> _replayChain;

		void decodeInput();

	public:
		int _encodeBlobRadius;
		int _replaySampleFrames;
		int _maxReplayChainSize;
		int _backpropPassesActor;
		int _backpropPassesCritic;
		int _approachPasses;
		float _actionInputVocalness;

		HTMRL();

		void createRandom(int inputWidth, int inputHeight, int inputDotsWidth, int inputDotsHeight, int numOutputs, int actorNumHiddenLayers, int actorNumNodesPerHiddenLayer, int criticNumHiddenLayers, int criticNumNodesPerHiddenLayer, float actorCriticInitWeightStdDev, const std::vector<RegionDesc> &regionDescs, std::mt19937 &generator);

		void setInput(int x, int y, int axis, float value) {
			_inputf[x + y * _inputWidth + axis * _inputWidth * _inputHeight] = std::min(1.0f, std::max(-1.0f, value));
		}

		float getInputf(int x, int y, int axis) const {
			assert(axis == 0 || axis == 1);

			return _inputf[x + y * _inputWidth + axis * _inputWidth * _inputHeight];
		}

		bool getInputb(int x, int y) const {
			return _inputb[x + _inputDotsWidth * _inputWidth * y];
		}

		float getOutput(int index) const {
			return _exploratoryOutputs[index];
		}

		const htm::Region &getRegion(int index) const {
			return _regions[index];
		}

		void step(float reward, float backpropAlphaActor, float backpropAlphaCritic, float alphaActor, float alphaCritic, float momentumActor, float momentumCritic, float gamma, float lambda, float tauInv, float perturbationStdDev, float breakRate, float policySearchStdDev, float actionMomentum, float varianceDecay, std::mt19937 &generator);
	
		int getInputWidth() const {
			return _inputWidth;
		}

		int getInputHeight() const {
			return _inputHeight;
		}

		int getInputDotsWidth() const {
			return _inputDotsWidth;
		}

		int getInputDotsHeight() const {
			return _inputDotsHeight;
		}

		int getInputMax() const {
			return _inputMax;
		}
	};
}