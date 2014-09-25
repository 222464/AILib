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

#include <experiments/ExperimentPoleBalancing.h>

#include <SFML/System.hpp>

#include <iostream>

float ExperimentPoleBalancing::evaluate(hn::HyperNet &hypernet, const hn::Config &config, std::mt19937 &generator) {
	hn::Config subConfig = config;

	subConfig._numInputGroups = 4;
	subConfig._numInputsPerGroup = 1;
	subConfig._numOutputGroups = 1;
	subConfig._numOutputsPerGroup = 1;

	std::vector<int> dimensions(2);

	dimensions[0] = 8;
	dimensions[1] = 3;

	hypernet.generateFeedForward(subConfig, 1, 12, generator);

	float pixelsPerMeter = 128.0f;
	float poleLength = 1.0f;
	float g = -2.8f;
	float massMass = 20.0f;
	float cartMass = 2.0f;
	sf::Vector2f massPos(0.0f, poleLength);
	sf::Vector2f massVel(0.0f, 0.0f);
	float poleAngle = static_cast<float>(PI) * 0.0f;
	float poleAngleVel = 0.0f;
	float poleAngleAccel = 0.0f;
	float cartX = 0.0f;
	float cartVelX = 0.0f;
	float cartAccelX = 0.0f;
	float poleRotationalFriction = 0.008f;
	float cartMoveRadius = 1.8f;
	float cartFriction = 0.02f;
	float maxSpeed = 3.0f;

	float dt = 0.03f;

	float fitness = 0.0f;
	float prevFitness = 0.0f;

	float totalFitness = 0.0f;

	for (size_t i = 0; i < 600; i++) {
		//std::cout << "Step " << i << std::endl;

		// Update fitness
		prevFitness = fitness;

		if (poleAngle < static_cast<float>(PI))
			fitness = -(static_cast<float>(PI) * 0.5f - poleAngle);
		else
			fitness = -(static_cast<float>(PI) * 0.5f - (static_cast<float>(PI) * 2.0f - poleAngle));

		//fitness = fitness - std::abs(poleAngleVel * 40.0f);

		totalFitness += fitness * 0.1f;

		// ------------------------------ AI -------------------------------

		float dFitness = fitness - prevFitness;

		float error = dFitness * 10.0f;

		//agent.reinforceArp(std::min(1.0f, std::max(-1.0f, error)) * 0.5f + 0.5f, 0.1f, 0.05f);

		hypernet.setInput(0, cartX * 0.25f);
		hypernet.setInput(1, cartVelX);
		hypernet.setInput(2, std::fmod(poleAngle + static_cast<float>(PI), 2.0f * static_cast<float>(PI)));
		hypernet.setInput(3, poleAngleVel);

		hypernet.step(subConfig, dFitness * 10.0f, generator, 6, 6.0f);

		float dir = std::min(1.0f, std::max(-1.0f, hypernet.getOutput(0)));

		//dir = 1.4f * (dir * 2.0f - 1.0f);

		float agentForce = 4000.0f * dir;
		//float agentForce = 2000.0f * agent.getOutput(0);

		// ---------------------------- Physics ----------------------------

		float pendulumCartAccelX = cartAccelX;

		if (cartX < -cartMoveRadius)
			pendulumCartAccelX = 0.0f;
		else if (cartX > cartMoveRadius)
			pendulumCartAccelX = 0.0f;

		poleAngleAccel = pendulumCartAccelX * std::cos(poleAngle) + g * std::sin(poleAngle);
		poleAngleVel += -poleRotationalFriction * poleAngleVel + poleAngleAccel * dt;
		poleAngle += poleAngleVel * dt;

		massPos = sf::Vector2f(cartX + std::cos(poleAngle + static_cast<float>(PI) * 0.5f) * poleLength, std::sin(poleAngle + static_cast<float>(PI) * 0.5f) * poleLength);

		float force = 0.0f;

		if (std::abs(cartVelX) < maxSpeed)
			force = std::max(-4000.0f, std::min(4000.0f, agentForce));

		if (cartX < -cartMoveRadius) {
			cartX = -cartMoveRadius;

			cartAccelX = -cartVelX / dt;
			cartVelX = -0.5f * cartVelX;
		}
		else if (cartX > cartMoveRadius) {
			cartX = cartMoveRadius;

			cartAccelX = -cartVelX / dt;
			cartVelX = -0.5f * cartVelX;
		}

		cartAccelX = 0.25f * (force + massMass * poleLength * poleAngleAccel * std::cos(poleAngle) - massMass * poleLength * poleAngleVel * poleAngleVel * std::sin(poleAngle)) / (massMass + cartMass);
		cartVelX += -cartFriction * cartVelX + cartAccelX * dt;
		cartX += cartVelX * dt;

		poleAngle = std::fmod(poleAngle, (2.0f * static_cast<float>(PI)));

		if (poleAngle < 0.0f)
			poleAngle += static_cast<float>(PI) * 2.0f;
	}

	std::cout << "Pole balancing experiment finished with total fitness of " << totalFitness << "." << std::endl;

	return totalFitness;
}