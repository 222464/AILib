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

#include <deep/RecurrentSparseAutoencoder.h>
#include <unordered_map>
#include <unordered_set>
#include <string>

namespace text {
	class Word2SDR {
	public:
		struct Settings {
			float _sparsity;
			int _hiddenSize;
			float _dutyCycleDecay;
			float _stateLeak;
			float _alpha;
			float _beta;
			float _gamma;
			float _momentum;

			Settings()
				: _sparsity(33.0f / 256.0f),
				_hiddenSize(256),
				_dutyCycleDecay(0.01f),
				_stateLeak(0.0f),
				_alpha(0.1f),
				_beta(0.1f * 0.5f),
				_gamma(0.0f),
				_momentum(0.0f)
			{}
		};

		struct WordFeatures {
			std::vector<float> _features;
	
			//void update(float sparsity);

			float distance(const WordFeatures &other) const;
		};

	private:
		std::unordered_map<std::string, WordFeatures> _words;

		deep::RecurrentSparseAutoencoder _rsa;

		std::string _predictedWord;
		WordFeatures _predictedFeatures;

		Settings _settings;

	public:
		void createRandom(int featureSize, const Settings &settings, float minInitWeight, float maxInitWeight, float recurrentScalar, std::mt19937 &generator);

		void show(const std::string &word);

		void read(std::istream &is);
		
		const std::string &getPrediction() {
			return _predictedWord;
		}

		void clearMemory() {
			_rsa.clearMemory();
		}

		void clearWords() {
			_words.clear();
		}
	};
}