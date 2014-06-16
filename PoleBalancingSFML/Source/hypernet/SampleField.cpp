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

#include <hypernet/SampleField.h>

#include <iostream>

#include <assert.h>

using namespace hn;

SampleField::SampleField()
: _xSize(0), _ySize(0)
{
	_kernel = std::bind(hn::kernelSquaredExponential, std::placeholders::_1, std::placeholders::_2, 0.1f);
}

void SampleField::create(size_t xSize, size_t ySize) {
	assert(_xSize == 0 && _ySize == 0);

	_xSize = xSize;
	_ySize = ySize;
}

void SampleField::addSample(const Sample &sample) {
	assert(_xSize != 0 && _ySize != 0);
	assert(sample._x.size() == _xSize && sample._y.size() == _ySize);

	_samples.push_back(sample);
}

std::vector<float> SampleField::getYAtX(const std::vector<float> &x) const {
	assert(_xSize != 0 && _ySize != 0);

	std::vector<float> result(_ySize, 0.0f);

	float totalInfluence = 0.0f;

	for (size_t s = 0; s < _samples.size(); s++) {
		float influence = _kernel(x, _samples[s]._x);

		for (size_t yi = 0; yi < result.size(); yi++)
			result[yi] += _samples[s]._y[yi] * influence;

		totalInfluence += influence;
	}

	float totalInfluenceInv = 1.0f / totalInfluence;

	for (size_t yi = 0; yi < result.size(); yi++)
		result[yi] *= totalInfluenceInv;

	return result;
}

float SampleField::getInfluenceAtX(const std::vector<float> &x) const {
	float totalInfluence = 0.0f;

	for (size_t s = 0; s < _samples.size(); s++) {
		float influence = _kernel(x, _samples[s]._x);

		totalInfluence += influence;
	}

	return totalInfluence;
}

float SampleField::getVarianceAtX(const std::vector<float> &x) const {
	float variance = 0.0f;

	float totalInfluence = 0.0f;

	std::vector<float> yAtX = getYAtX(x);

	for (size_t s = 0; s < _samples.size(); s++) {
		float influence = _kernel(x, _samples[s]._x);

		float distance = 0.0f;

		for (size_t yi = 0; yi < yAtX.size(); yi++) {
			float difference = _samples[s]._y[yi] - yAtX[yi];

			distance += difference * difference;
		}

		distance = std::sqrt(distance);

		variance += distance * influence;

		totalInfluence += influence;
	}

	return variance / totalInfluence;
}

float hn::kernelSquaredExponential(const std::vector<float> &x1, const std::vector<float> &x2, float invThetaSquared) {
	assert(x1.size() == x2.size());

	float magnitudeSquared = 0.0f;

	for (size_t xi = 0; xi < x1.size(); xi++) {
		float difference = x1[xi] - x2[xi];
		magnitudeSquared += difference * difference;
	}

	return std::exp(-0.5f * invThetaSquared * std::sqrt(magnitudeSquared));
}