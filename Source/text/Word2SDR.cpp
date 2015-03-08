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

#include <text/Word2SDR.h>

#include <algorithm>

using namespace text;

/*void Word2SDR::WordFeatures::update(float sparsity) {
	int localActivity = std::round(_features.size() * sparsity);

	std::vector<float> newFeatures(_features.size());

	float dist2 = 0.0f;

	for (int i = 0; i < _features.size(); i++) {
		float numHigher = 0.0f;

		for (int j = 0; j < _features.size(); j++) {
			if (_features[j] > _features[i])
				numHigher++;
		}

		if (numHigher < localActivity)
			newFeatures[i] = _features[i];
		else
			newFeatures[i] = 0.0f;

		dist2 += newFeatures[i] * newFeatures[i];
	}

	float magInv = 1.0f / std::sqrt(std::max(0.00001f, dist2));

	for (int i = 0; i < newFeatures.size(); i++)
		newFeatures[i] *= magInv;

	_features = newFeatures;
}*/

float Word2SDR::WordFeatures::distance(const WordFeatures &other) const {
	float dist2 = 0.0f;

	for (int i = 0; i < _features.size(); i++) {
		float delta = _features[i] - other._features[i];
		dist2 += delta * delta;
	}

	return std::sqrt(dist2);
}

void Word2SDR::createRandom(int featureSize, const Settings &settings, float minInitWeight, float maxInitWeight, float recurrentScalar, std::mt19937 &generator) {
	_settings = settings;

	_rsa.createRandom(featureSize, _settings._hiddenSize, _settings._sparsity, minInitWeight, maxInitWeight, recurrentScalar, generator);

	_predictedWord = "";
	_predictedFeatures._features.resize(featureSize);

	std::uniform_real_distribution<float> uniformDist(0.0f, 1.0f);

	for (int i = 0; i < featureSize; i++)
		_predictedFeatures._features[i] = uniformDist(generator);

	//_predictedFeatures.update(settings._sparsity);
}

void Word2SDR::show(const std::string &word) {
	_rsa.stepBegin();

	std::unordered_map<std::string, WordFeatures>::iterator fit = _words.find(word);

	// If could not find word, create the word and give it the predicted features
	if (fit == _words.end()) {
		_words[word] = _predictedFeatures;

		for (int v = 0; v < _rsa.getNumVisibleNodes(); v++)
			_rsa.setVisibleNodeState(v, _predictedFeatures._features[v]);
	}
	else { // Has word, update to point to its features
		for (int v = 0; v < _rsa.getNumVisibleNodes(); v++)
			_rsa.setVisibleNodeState(v, fit->second._features[v]);
	}

	_rsa.activate(_settings._sparsity, _settings._dutyCycleDecay);

	_rsa.learn(_settings._sparsity, _settings._stateLeak, _settings._alpha, _settings._beta, _settings._gamma, 1.0f, _settings._momentum, 1.0f, 1.0f);

	// Get new predicted word
	for (int v = 0; v < _rsa.getNumVisibleNodes(); v++)
		_predictedFeatures._features[v] = _rsa.getVisibleNodeReconstruction(v);

	//_predictedFeatures.update(_settings._sparsity);

	// Find closest word
	float minDistance = 999999.0f;

	for (std::unordered_map<std::string, WordFeatures>::iterator it = _words.begin(); it != _words.end(); it++) {
		float dist = _predictedFeatures.distance(it->second);

		if (dist < minDistance) {
			minDistance = dist;

			_predictedWord = it->first;
		}
	}
}

void Word2SDR::read(std::istream &is) {
	while (is.good() && !is.eof()) {
		std::string word;
		is >> word;

		show(word);
	}
}