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

#include <chtm/CHTMRegion.h>

namespace chtm {
	class CHTMRL {
	public:

	private:
		CHTMRegion _region;

		float _prevValue;

	public:
		CHTMRL()
			: _prevValue(0.0f)
		{}

		void createRandom(int inputWidth, int inputHeight, int columnsWidth, int columnsHeight, int cellsPerColumn, int receptiveRadius, int cellRadius,
			float minCenter, float maxCenter, float minWidth, float maxWidth, float minInputWeight, float maxInputWeight, float minReconWeight, float maxReconWeight,
			float minCellWeight, float maxCellWeight, float minOutputWeight, float maxOutputWeight, std::mt19937 &generator);

		void step(float reward, const std::vector<float> &input, const std::vector<bool> &actionMask, std::vector<float> &action, float optimizationAlpha, int optimizationSteps, float optimizationPerturbationStdDev, float optimizationDecay, float indecisivnessIntensity, float perturbationIntensity, float intentSparsity, float intentIntensity, int inhibitionRadius, float sparsity, float cellIntensity, float predictionIntensity, float weightAlphaQ, float reconAlpha, float centerAlpha, float widthAlpha, float widthScalar,
			float minDistance, float minLearningThreshold, float cellAlpha, float qAlpha, float gamma, float lambda, float tauInv, std::mt19937 &generator);

		const CHTMRegion &getRegion() const {
			return _region;
		}
	};
}