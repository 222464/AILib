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

#include <vector>
#include <list>
#include <random>

namespace deep {
	class FERL {
	public:
		static float sigmoid(float x) {
			return 1.0f / (1.0f + std::exp(-x));
		}

		struct ReplaySample {
			std::vector<float> _visible;

			float _q;
		};

	private:
		struct Connection {
			float _weight;

			float _prevDWeight;

			Connection()
				: _prevDWeight(0.0f)
			{}
		};

		struct Hidden {
			Connection _bias;
			std::vector<Connection> _connections;
			float _state;

			Hidden()
				: _state(0.0f)
			{}
		};

		struct Visible {
			Connection _bias;
			float _state;

			Visible()
				: _state(0.0f)
			{}
		};

		std::vector<Hidden> _hidden;
		std::vector<Visible> _visible;
		int _numState;
		int _numAction;

		float _zInv;

		float _prevMax;
		float _prevValue;

		std::vector<float> _prevVisible;

		std::list<ReplaySample> _replaySamples;

	public:
		FERL();

		void createRandom(int numState, int numAction, int numHidden, float weightStdDev, std::mt19937 &generator);

		void createFromParents(const FERL &parent1, const FERL &parent2, float averageChance, std::mt19937 &generator);

		void mutate(float perturbationStdDev, std::mt19937 &generator);

		// Returns action index
		void step(const std::vector<float> &state, std::vector<float> &action,
			float reward, float qAlpha, float gamma, float lambdaGamma, float tauInv,
			int actionSearchIterations, int actionSearchSamples, float actionSearchAlpha,
			float breakChance, float perturbationStdDev,
			int maxNumReplaySamples, int replayIterations, float gradientAlpha, float gradientMomentum,
			std::mt19937 &generator);

		void activate();
		void updateOnError(float error, float momentum);

		float freeEnergy() const;

		float value() const {
			return -freeEnergy() * _zInv;
		}

		int getNumState() const {
			return _numState;
		}

		int getNumAction() const {
			return _numAction;
		}

		int getNumVisible() const {
			return _visible.size();
		}

		int getNumHidden() const {
			return _hidden.size();
		}

		float getZInv() const {
			return _zInv;
		}

		const std::list<ReplaySample> &getSamples() const {
			return _replaySamples;
		}
	};
}