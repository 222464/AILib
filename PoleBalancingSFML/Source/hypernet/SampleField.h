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
#include <memory>
#include <functional>

namespace hn {
	class SampleField {
	public:
		struct Sample {
			std::vector<float> _x;
			std::vector<float> _y;

			Sample() {}
			Sample(const std::vector<float> &x, std::vector<float> &y)
				: _x(x), _y(y)
			{}
		};

	private:
		std::vector<Sample> _samples;

		size_t _xSize, _ySize;

	public:
		std::function<float(const std::vector<float> &, const std::vector<float> &)> _kernel;

		SampleField();

		void create(size_t xSize, size_t ySize);

		void addSample(const Sample &sample);

		std::vector<float> getYAtX(const std::vector<float> &x) const;
		float getInfluenceAtX(const std::vector<float> &x) const;
		float getVarianceAtX(const std::vector<float> &x) const;

		size_t getXSize() const {
			return _xSize;
		}

		size_t getYSize() const {
			return _ySize;
		}

		size_t getNumSamples() const {
			return _samples.size();
		}

		const Sample &getSample(size_t index) const {
			return _samples[index];
		}

		void clearSamples() {
			_samples.clear();
		}
	};

	float kernelSquaredExponential(const std::vector<float> &x1, const std::vector<float> &x2, float invThetaSquared);
}