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

#include <hypernet/Link.h>

#include <unordered_map>
#include <memory>

namespace hn {
	class Boid {
	private:
		std::vector<float> _outputs;

		std::vector<float> _boidMemory;

	public:
		std::vector<float> _inputs;

		std::unordered_map<int, std::shared_ptr<Link>> _links;

		Boid(const Config &config, class HyperNet &hypernet, std::mt19937 &generator);

		void gatherInput(const Config &config, float reward, class HyperNet &hypernet, std::mt19937 &generator, int myInputOffset, float activationMultiplier);
		void update(const Config &config, float reward, class HyperNet &hypernet, std::mt19937 &generator, int myInputOffset, float activationMultiplier);

		size_t getNumOutputs() const {
			return _outputs.size();
		}

		float getOutput(size_t index) const {
			return _outputs[index];
		}

		size_t getMemorySize() const {
			return _boidMemory.size();
		}

		float getMemory(size_t index) const {
			return _boidMemory[index];
		}
	};
}