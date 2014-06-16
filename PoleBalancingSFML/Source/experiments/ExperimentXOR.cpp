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

#include <experiments/ExperimentXOR.h>

#include <SFML/System.hpp>

#include <iostream>

float ExperimentXOR::evaluate(hn::HyperNet &hypernet, const hn::Config &config, std::mt19937 &generator) {
	float inputs[4][2] {
		{ -10.0f, -10.0f },
		{ -10.0f, 10.0f },
		{ 10.0f, -10.0f },
		{ 10.0f, 10.0f }
	};

	float outputs[4] {
		0.0f,
		1.0f,
		1.0f,
		0.0f
	};

	hn::Config subConfig = config;

	subConfig._numInputGroups = 2;
	subConfig._numInputsPerGroup = 1;
	subConfig._numOutputGroups = 1;
	subConfig._numOutputsPerGroup = 1;

	std::vector<int> dimensions(2);

	dimensions[0] = 6;
	dimensions[1] = 3;

	hypernet.generateFeedForward(subConfig, 1, 4, generator);

	float reward = 0.0f;
	float prevReward = 0.0f;
	float totalReward = 0.0f;

	std::vector<float> values(4);

	for (size_t i = 0; i < 100; i++) {
		float newReward = 1.0f;

		float average = 0.0f;

		for (size_t j = 0; j < 4; j++) {
			hypernet.setInput(0, inputs[j][0]);
			hypernet.setInput(1, inputs[j][1]);

			hypernet.step(config, (reward - prevReward) * 50.0f, generator, 4, 6.0f);

			newReward -= std::abs(outputs[j] - hypernet.getOutput(0)) * 0.25f;

			values[j] = hypernet.getOutput(0);
			average += values[j];
		}

		average *= 0.25f;

		//for (size_t j = 0; j < 4; j++) {
		//	newReward -= 0.05f / (0.125f + 8.0f * std::abs(average - values[j]));
		//}
		
		prevReward = reward;
		reward = newReward;

		totalReward += newReward;
	}

	std::cout << "Finished XOR experiment with total reward of " << totalReward << "." << std::endl;

	return totalReward;
}