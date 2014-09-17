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

#include <htm/Column.h>

#include <random>
#include <functional>

namespace htm {
	class Region {
	private:
		std::vector<Column> _columns;

		int _regionWidth;
		int _regionHeight;
		int _inputWidth;
		int _inputHeight;
		int _connectionRadius;

		std::vector<int> _activeColumnIndices;

	public:
		void createRandom(int inputWidth, int inputHeight, int connectionRadius, float initInhibitionRadius, int initNumSegments,
			int regionWidth, int regionHeight, int columnSize, float permanenceDistanceBias, float permanenceDistanceFalloff, float permanenceBiasFloor,
			float connectionPermanenceTarget, float connectionPermanenceStdDev, std::mt19937 &generator);

		void spatialPooling(const std::vector<bool> &inputs, float minPermanence, float minOverlap, int desiredLocalActivity,
			float permanenceIncrease, float permanenceDecrease, float minDutyCycleRatio, float activeDutyCycleDecay,
			float overlapDutyCycleDecay, float subOverlapPermanenceIncrease,
			std::function<float(float, float)> &boostFunction);

		void stepBegin();
		void temporalPoolingNoLearn(float minPermanence, int activationThreshold);
		void temporalPoolingLearn(float minPermanence, int learningRadius, int minLearningThreshold, int activationThreshold, int newNumConnections, float permanenceIncrease, float permanenceDecrease, float newConnectionPermanence, int maxSteps, std::mt19937 &generator);

		const Column &getColumn(int i) const {
			return _columns[i];
		}

		const Column &getColumn(int x, int y) const {
			return _columns[x + y * _regionWidth];
		}

		bool getOutput(int i) const;
		bool getOutput(int x, int y) const;
		bool getPrediction(int i, int t) const;
		bool getPrediction(int x, int y, int t) const;
		void setColumnsToOutput();
		bool hasLearningCell(int x, int y) const;
		bool hasSegments(int x, int y) const;
		bool hasConnections(int x, int y) const;
		void getReconstruction(std::vector<bool> &output, float minOverlap, float minPermanence, bool fromPrediction) const;
		void getReconstructionAtTime(std::vector<bool> &output, float minOverlap, float minPermanence, int t) const;
	};
}