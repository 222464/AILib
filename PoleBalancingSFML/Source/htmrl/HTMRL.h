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

#include <algorithm>

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
				_permanenceDistanceBias(0.1f),
				_permanenceDistanceFalloff(2.0f),
				_permanenceBiasFloor(-0.05f),
				_connectionPermanenceTarget(0.3f),
				_connectionPermanenceStdDev(0.1f),
				_minPermanence(0.3f),
				_minOverlap(2.0f),
				_desiredLocalActivity(8),
				_spatialPermanenceIncrease(0.05f),
				_spatialPermanenceDecrease(0.04f),
				_minDutyCycleRatio(0.01f),
				_activeDutyCycleDecay(0.01f),
				_overlapDutyCycleDecay(0.01f),
				_subOverlapPermanenceIncrease(0.04f),
				_boostFunction(std::bind(defaultBoostFunction, std::placeholders::_1, std::placeholders::_2)),
				_learningRadius(4),
				_minLearningThreshold(1),
				_activationThreshold(6),
				_newNumConnections(32),
				_temporalPermanenceIncrease(0.05f),
				_temporalPermanenceDecrease(0.04f),
				_newConnectionPermanence(0.31f),
				_maxSteps(4)
			{}
		};

		static float sigmoid(float x) {
			return 1.0f / (1.0f + std::exp(-x));
		}

	private:
		struct StateOutput {
			std::vector<float> _cellWeights;
			std::vector<float> _columnWeights;
			std::vector<float> _columnBiases;

			std::vector<float> _columnPrevPrevOutputs;
			std::vector<float> _columnPrevOutputs;
			std::vector<float> _columnOutputs;
			std::vector<float> _columnErrors;
			float _outputError;

			float _bias;
			float _prevPrevOutput;
			float _prevOutput;
			float _output;
		};

		int _inputWidth;
		int _inputHeight;
		int _inputDotsWidth;
		int _inputDotsHeight;
		int _inputMax;

		std::vector<float> _inputf;
		std::vector<bool> _inputb;

		htm::Region _region;

		StateOutput _criticOutput;
		std::vector<StateOutput> _actorOutputs;

		std::vector<bool> _prevPrevState;
		std::vector<bool> _prevState;

		std::vector<float> _prevPrevOutputs;
		std::vector<float> _prevOutputs;
		std::vector<float> _outputs;
		std::vector<float> _outputOffsets;

		float _prevVariance;
		float _variance;
		float _prevError;

		float _prevQ;
		float _prevNextQ;
		bool _firstStep;

		bool _explore;

		void decodeInput();

	public:
		HTMRL();

		void createRandom(int inputWidth, int inputHeight, int inputDotsWidth, int inputDotsHeight, int numOutputs, const RegionDesc &regionDesc, float stateOutputWeightStdDev, std::mt19937 &generator);

		void setInput(int x, int y, float value) {
			_inputf[x + y * _inputWidth] = std::min(1.0f, std::max(-1.0f, value));
		}

		float getInputf(int x, int y) const {
			return _inputf[x + y * _inputWidth];
		}

		bool getInputb(int x, int y) const {
			return _inputb[x + _inputDotsWidth * _inputWidth * y];
		}

		float getOutput(int index) const {
			return _outputs[index];
		}

		const htm::Region &getRegion() const {
			return _region;
		}

		void step(float reward, float gamma, float qAlpha, float hebbianAlphaActor, float backpropAlphaCritic, RegionDesc &regionDesc, float outputPerturbationStdDev, float breakRate, float exploreErrorTolerance, std::mt19937 &generator);
	};
}