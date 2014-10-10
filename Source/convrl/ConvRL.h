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

#include <lstm/LSTMG.h>

#include <deep/FA.h>

#include <deep/ConvNet2D.h>

#include <deep/FERL.h>

#include <algorithm>

#include <assert.h>

namespace deep {
	class ConvRL {
	public:
		struct ReplaySample {
			std::vector<float> _inputs;
		};

	private:
		ConvNet2D _convNet;

		FERL _ferl;

		float _prevMaxQ;
		float _prevValue;

		std::list<ReplaySample> _replayChain;

		std::vector<float> _outputs;

	public:
		ConvRL();

		void createRandom(int inputMapWidth, int inputMapHeight, int inputNumMaps, const std::vector<ConvNet2D::LayerPairDesc> &layerDescs, float convMinWeight, float convMaxWeight, int numOutputs, int ferlNumHidden, float ferlWeightStdDev, std::mt19937 &generator);

		void setInput(int x, int y, int m, float value) {
			_convNet.setInput(x, y, m, value);
		}

		float getOutput(int i) const {
			return _outputs[i];
		}

		const ConvNet2D &getConvNet() const {
			return _convNet;
		}

		int getNumOutputs() const {
			return _outputs.size();
		}

		void step(float reward, float qAlpha, float gamma, float lambdaGamma, float tauInv,
			float rbmAlpha, int convMaxNumReplaySamples, int convNumReplayIterations,
			int actionSearchIterations, int actionSearchSamples, float actionSearchAlpha,
			float breakChance, float perturbationStdDev,
			int ferlMaxNumReplaySamples, int ferlReplayIterations,
			float gradientAlpha, float gradientMomentum,
			std::mt19937 &generator, std::vector<float> &convBuff);
	};
}