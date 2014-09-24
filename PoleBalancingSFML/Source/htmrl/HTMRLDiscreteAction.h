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

#include <assert.h>

namespace htmrl {
	float defaultBoostFunctionDiscreteAction(float active, float minimum);

	class HTMRLDiscreteAction {
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
				_subOverlapPermanenceIncrease(0.03f),
				_boostFunction(std::bind(defaultBoostFunctionDiscreteAction, std::placeholders::_1, std::placeholders::_2)),
				_learningRadius(4),
				_minLearningThreshold(1),
				_activationThreshold(6),
				_newNumConnections(32),
				_temporalPermanenceIncrease(0.03f),
				_temporalPermanenceDecrease(0.025f),
				_newConnectionPermanence(0.31f),
				_maxSteps(6)
			{}
		};

		struct ReplaySample {
			std::vector<bool> _actorInputsb;
			int _actionExploratory;
			int _actionOptimal;
			float _reward;

			std::vector<float> _actionQValues;
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

		deep::FA _critic;

		int _prevMaxQAction;
		int _prevChooseAction;

		std::vector<float> _prevQValues;

		std::list<ReplaySample> _replayChain;

		void decodeInput();

	public:
		int _encodeBlobRadius;
		int _replaySampleFrames;
		int _maxReplayChainSize;
		int _backpropPassesCritic;

		HTMRLDiscreteAction();

		void createRandom(int inputWidth, int inputHeight, int inputDotsWidth, int inputDotsHeight, int numOutputs, int criticNumHiddenLayers, int criticNumNodesPerHiddenLayer, float criticInitWeightStdDev, const std::vector<RegionDesc> &regionDescs, std::mt19937 &generator);

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

		const htm::Region &getRegion(int index) const {
			return _regions[index];
		}

		int step(float reward, float backpropAlphaCritic, float momentumCritic, float gamma, float lambda, float tauInv, float epsilon, float weightDecayMultiplier, std::mt19937 &generator);

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