/*
AI Lib
Copyright (C) 4014 Eric Laukien

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

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include <SFML/Audio.hpp>

#include <Consts.h>

#include <nn/QAgent.h>
#include <nn/ActorCriticAgent.h>
#include <nn/Cacla.h>
#include <nn/MemoryActor.h>
#include <nn/RLLSTMAgent.h>
#include <nn/PSOAgent.h>
#include <nn/MultiQ.h>
#include <nn/SOMQAgent.h>

#include <lstm/LSTMActorCritic.h>

#include <deep/AutoLSTM.h>

#include <deep/FERL.h>

#include <rbf/SDRRBFNetwork.h>

#include <convrl/ConvRL.h>

#include <hypernet/BayesianOptimizerTrainer.h>
#include <hypernet/EvolutionaryTrainer.h>

#include <experiments/ExperimentPoleBalancing.h>
#include <experiments/ExperimentXOR.h>
#include <experiments/ExperimentOR.h>
#include <experiments/ExperimentAND.h>

#include <hypernet/SampleField.h>
#include <hypernet/BayesianOptimizer.h>

#include <dnf/Field.h>

#include <ctrnn/GeneticAlgorithm.h>

#include <chtm/CHTMRegion.h>
#include <chtm/CHTMRL.h>

#include <nn/GeneticAlgorithm.h>

#include <falcon/Falcon.h>

#include <elman/ElmanNetwork.h>

#include <htm/Region.h>
#include <htmrl/HTMRL.h>
#include <htmrl/HTMRLDiscreteAction.h>

#include <rbf/RBFNetwork.h>

#include <plot/plot.h>

#include <deep/RBM.h>
#include <deep/DBN.h>
#include <deep/ConvNet2D.h>
#include <deep/SharpFA.h>

#include <raahn/AutoEncoder.h>

#include <time.h>
#include <iostream>
#include <random>
#include <array>
#include <fstream>
#include <sstream>

/*int main() {
	std::mt19937 generator(time(nullptr));*/
	
	/*hn::FunctionApproximator approx;
	approx.createRandom(2, 1, 1, 4, -0.2f, 0.2f, generator);

	float inputs[4][2] {
		{ 0.0f, 0.0f },
		{ 0.0f, 10.0f },
		{ 10.0f, 0.0f },
		{ 10.0f, 10.0f }
	};

	float outputs[4] {
		0.0f,
		0.0f,
		1.0f,
		0.0f
	};

	float reward = 0.0f;
	float prevReward = 0.0f;
	float totalReward = 0.0f;

	std::vector<float> inputBuffer(2);
	std::vector<float> outputBuffer(1);

	for (size_t i = 0; i < 1000; i++) {
		float newReward = 1.0f;

		for (size_t j = 0; j < 4; j++) {
			inputBuffer[0] = (inputs[j][0]);
			inputBuffer[1] = (inputs[j][1]);

			std::vector<std::vector<float>> layerOutputs;

			approx.process(inputBuffer, layerOutputs);

			newReward -= std::abs(outputs[j] - layerOutputs.back()[0]) * 0.25f;

			approx.backpropagate(inputBuffer, layerOutputs, std::vector<float>(1, outputs[j]), 0.1f);
		}

		prevReward = reward;
		reward = newReward;

		totalReward += newReward;
	}

	for (size_t j = 0; j < 4; j++) {
		inputBuffer[0] = (inputs[j][0]);
		inputBuffer[1] = (inputs[j][1]);

		approx.process(inputBuffer, outputBuffer);

		std::cout << outputBuffer[0] << std::endl;
	}

	system("pause");*/

	/*hn::EvolutionaryTrainer trainer;
	hn::Config config;
	
	trainer.create(config, 32, generator, 6.0f);

	trainer.addExperiment(std::shared_ptr<hn::Experiment>(new ExperimentXOR()));
	trainer.addExperiment(std::shared_ptr<hn::Experiment>(new ExperimentOR()));
	trainer.addExperiment(std::shared_ptr<hn::Experiment>(new ExperimentAND()));

	for (size_t g = 0; g < 10000; g++) {
		std::cout << "Evaluating generation " << (g + 1) << "." << std::endl;

		trainer.evaluate(config, generator);

		std::cout << "Reproducing generation " << (g + 1) << "." << std::endl;

		trainer.reproduce(config, generator);

		std::cout << "Saving best to \"hypernet1.txt\"" << std::endl;

		std::ofstream toFile("hypernet1.txt");

		trainer.writeBestToStream(toFile);

		toFile.close();

		std::cout << "Generation completed." << std::endl;

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
			break;
	}

	return 0;
}*/

/*float testFunc(float x) {
	return 0.2f * x * x * x * x + 0.5f * x * x * x - 1.2f * x * x + 0.2f * x + 0.3f;
}*/

/*int main() {*/
	/*sf::RenderWindow window;

	window.create(sf::VideoMode(800, 600), "Pole Balancing");

	window.setVerticalSyncEnabled(true);

	hn::SampleField sampleField;
	sampleField.create(2, 3);
	sampleField._kernel = std::bind(hn::kernelSquaredExponential, std::placeholders::_1, std::placeholders::_2, 128.0f);

	std::mt19937 generator(time(nullptr));

	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

	// Add some samples
	for (size_t i = 0; i < 100; i++) {
		hn::SampleField::Sample s;

		s._x.resize(2);
		s._x[0] = dist01(generator);
		s._x[1] = dist01(generator);
		s._y.resize(3);
		s._y[0] = dist01(generator);
		s._y[1] = dist01(generator);
		s._y[2] = dist01(generator);

		sampleField.addSample(s);
	}
	
	sf::Image image;

	image.create(800, 600);

	for (unsigned int x = 0; x < image.getSize().x; x++)
	for (unsigned int y = 0; y < image.getSize().y; y++) {
		std::vector<float> sx(2);
		sx[0] = static_cast<float>(x) / static_cast<float>(image.getSize().x);
		sx[1] = static_cast<float>(y) / static_cast<float>(image.getSize().y);

		std::vector<float> sy = sampleField.getYAtX(sx);

		sf::Color c;

		c.r = sy[0] * 255.0f;
		c.g = sy[1] * 255.0f;
		c.b = sy[2] * 255.0f;

		image.setPixel(x, y, c);
	}

	sf::Texture texture;
	texture.loadFromImage(image);

	// ----------------------------- Game Loop -----------------------------

	bool quit = false;

	sf::Clock clock;

	float dt = 0.017f;

	do {
		clock.restart();

		// ----------------------------- Input -----------------------------

		sf::Event windowEvent;

		while (window.pollEvent(windowEvent))
		{
			switch (windowEvent.type)
			{
			case sf::Event::Closed:
				quit = true;
				break;
			}
		}

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
			quit = true;

		// ---------------------------- Rendering ----------------------------

		sf::Sprite s;
		s.setTexture(texture);

		window.draw(s);

		// -------------------------------------------------------------------

		window.display();

		dt = clock.getElapsedTime().asSeconds();
	} while (!quit);

	return 0;*/

	/*std::mt19937 generator(time(nullptr));
	hn::BayesianOptimizer bo;

	bo.create(1, std::vector<float>(1, -3.0f), std::vector<float>(1, 1.0f));

	for (size_t i = 0; i < 6; i++) {
		bo.generateNewVariables(generator);

		bo.update(testFunc(bo.getCurrentVariables()[0]));
	}

	std::cout << bo.getCurrentVariables()[0] << std::endl;

	system("pause");*/

	/*std::mt19937 generator(time(nullptr));

	hn::Config config;
	hn::BayesianOptimizerTrainer bot;

	bot.create(config, -2.0f, 2.0f, generator);

	bot.addExperiment(std::shared_ptr<hn::Experiment>(new ExperimentXOR()));
	bot.addExperiment(std::shared_ptr<hn::Experiment>(new ExperimentOR()));
	bot.addExperiment(std::shared_ptr<hn::Experiment>(new ExperimentAND()));

	for (size_t g = 0; g < 1000; g++) {
		bot.evaluate(config, generator);

		std::cout << "Current fitness: " << bot.getCurrentFitness() << std::endl;

		bot.update(config, generator);
	}

	return 0;
}*/

/*int main() {
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

	hn::Config config;
	config._numInputGroups = 2;
	config._numInputsPerGroup = 1;
	config._numOutputGroups = 1;
	config._numOutputsPerGroup = 1;

	std::mt19937 generator(time(nullptr));
	hn::HyperNet hypernet;
	hypernet.readFromStream(std::ifstream("hypernet1.txt"));

	std::vector<int> dims(2);
	dims[0] = 12;
	dims[1] = 6;
	hypernet.generateFeedForward(config, 1, 4, generator);

	float reward = 0.0f;
	float prevReward = 0.0f;
	float totalReward = 0.0f;

	for (size_t i = 0; i < 400; i++) {
		float newReward = 1.0f;

		for (size_t j = 0; j < 4; j++) {
			hypernet.setInput(0, inputs[j][0]);
			hypernet.setInput(1, inputs[j][1]);

			hypernet.step(config, (reward - prevReward) * 50.0f, generator, 6, 6.0f);

			newReward -= std::abs(outputs[j] - hypernet.getOutput(0)) * 0.25f;
		}

		prevReward = reward;
		reward = newReward;

		totalReward += newReward;
	}

	for (size_t j = 0; j < 4; j++) {
		hypernet.setInput(0, inputs[j][0]);
		hypernet.setInput(1, inputs[j][1]);

		hypernet.step(config, (reward - prevReward) * 50.0f, generator, 6, 6.0f);

		std::cout << hypernet.getOutput(0) << std::endl;
	}

	std::cout << "totalReward: " << totalReward << std::endl;

	system("pause");

	return 0;
}*/

/*
int main() {
	std::mt19937 generator(time(nullptr));

	raahn::AutoEncoder ae;

	ae.create(10, 3, -0.3f, 0.3f, generator);

	std::vector<float> inputs(10, 0.0f);
	std::vector<float> outputs(4, 0.0f);

	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

	for (size_t j = 0; j < 10; j++) {
		inputs[j] = dist01(generator);
	}

	for (size_t i = 0; i < 1000; i++) {
		

		ae.update(inputs, outputs, 0.1f);

		float error = 0.0f;

		for (size_t j = 0; j < 10; j++)
			error += ae.getInputErrorBuffer()[j];

		std::cout << "Total Error: " << error << std::endl;

		
	}

	system("pause");

	return 0;
}*/

float evaluatePoleBalancing(ctrnn::CTRNN &net, std::mt19937 &generator) {
	float pixelsPerMeter = 128.0f;
	float poleLength = 1.0f;
	float g = -2.8f;
	float massMass = 40.0f;
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

	for (size_t i = 0; i < 1500; i++) {
		//std::cout << "Step " << i << std::endl;

		// Update fitness
		prevFitness = fitness;

		if (poleAngle < static_cast<float>(PI))
			fitness = -(static_cast<float>(PI) * 0.5f - poleAngle);
		else
			fitness = -(static_cast<float>(PI) * 0.5f - (static_cast<float>(PI) * 2.0f - poleAngle));

		//fitness = fitness - std::abs(poleAngleVel * 40.0f);

		totalFitness += fitness * 0.01f;

		// ------------------------------ AI -------------------------------

		float dFitness = fitness - prevFitness;

		float error = dFitness * 10.0f;

		//agent.reinforceArp(std::min(1.0f, std::max(-1.0f, error)) * 0.5f + 0.5f, 0.1f, 0.05f);

		net.setInput(0, cartX * 0.25f);
		net.setInput(1, cartVelX);
		net.setInput(2, std::fmod(poleAngle + static_cast<float>(PI), 2.0f * static_cast<float>(PI)));
		net.setInput(3, poleAngleVel);

		net.setInput(4, dFitness * 0.1f);

		for (size_t s = 0; s < 18; s++)
			net.step(1.0f, dFitness * 0.1f, 0.1f, generator);

		float dir = std::min(1.0f, std::max(-1.0f, 2.0f * net.getOutput(0) - 1.0f));

		//dir = 1.4f * (dir * 2.0f - 1.0f);

		float agentForce = 4000.0f * dir;
		//float agentForce = 4000.0f * agent.getOutput(0);

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

	//std::cout << "Pole balancing experiment finished with total fitness of " << totalFitness << "." << std::endl;

	return totalFitness;
}

float evaluateAND(ctrnn::CTRNN &net, std::mt19937 &generator) {
	float inputs[4][2] {
		{ -1.0f, -1.0f },
		{ -1.0f, 1.0f },
		{ 1.0f, -1.0f },
		{ 1.0f, 1.0f }
	};

	float outputs[4] {
		0.0f,
		0.0f,
		0.0f,
		1.0f
	};

	float reward = 0.0f;
	float prevReward = 0.0f;

	float initReward = 0.0f;

	float totalReward = 0.0f;

	std::vector<float> values(4);

	for (size_t i = 0; i < 100; i++) {
		float newReward = 1.0f;

		float average = 0.0f;

		for (size_t j = 0; j < 4; j++) {
			net.setInput(0, inputs[j][0]);
			net.setInput(1, inputs[j][1]);
			net.setInput(2, inputs[j][0]);
			net.setInput(3, inputs[j][1]);
			net.setInput(4, (reward - prevReward) * 0.1f);

			for(size_t s = 0; s < 18; s++)
				net.step(1.0f, (reward - prevReward) * 0.1f, 0.1f, generator);

			newReward -= std::pow(std::abs(outputs[j] - net.getOutput(0)), 1.0f) * 0.25f;

			values[j] = net.getOutput(0);
			average += values[j];
		}

		average *= 0.25f;

		//for (size_t j = 0; j < 4; j++) {
		//	newReward -= 0.05f / (0.125f + 8.0f * std::abs(average - values[j]));
		//}

		prevReward = reward;
		reward = newReward;

		if (i == 0)
			initReward = std::min(1.0f, std::max(0.0f, reward));

		totalReward += reward * 0.1f;
	}

	//std::cout << "Finished AND experiment with total reward of " << (reward - initReward) * std::min(1.0f, std::max(0.0f, (reward * reward))) << "." << std::endl;

	return totalReward;
}

float evaluateOR(ctrnn::CTRNN &net, std::mt19937 &generator) {
	float inputs[4][2] {
		{ -1.0f, -1.0f },
		{ -1.0f, 1.0f },
		{ 1.0f, -1.0f },
		{ 1.0f, 1.0f }
	};

	float outputs[4] {
		0.0f,
		1.0f,
		1.0f,
		1.0f
	};

	float reward = 0.0f;
	float prevReward = 0.0f;

	float initReward = 0.0f;

	float totalReward = 0.0f;

	std::vector<float> values(4);

	for (size_t i = 0; i < 100; i++) {
		float newReward = 1.0f;

		float average = 0.0f;

		for (size_t j = 0; j < 4; j++) {
			net.setInput(0, inputs[j][0]);
			net.setInput(1, inputs[j][1]);
			net.setInput(2, inputs[j][0]);
			net.setInput(3, inputs[j][1]);
			net.setInput(4, (reward - prevReward) * 0.1f);

			for (size_t s = 0; s < 18; s++)
				net.step(1.0f, (reward - prevReward) * 0.1f, 0.1f, generator);

			newReward -= std::pow(std::abs(outputs[j] - net.getOutput(0)), 1.0f) * 0.25f;

			values[j] = net.getOutput(0);
			average += values[j];
		}

		average *= 0.25f;

		//for (size_t j = 0; j < 4; j++) {
		//	newReward -= 0.05f / (0.125f + 8.0f * std::abs(average - values[j]));
		//}

		prevReward = reward;
		reward = newReward;

		if (i == 0)
			initReward = std::min(1.0f, std::max(0.0f, reward));

		totalReward += reward * 0.5f;
	}

	//std::cout << "Finished AND experiment with total reward of " << (reward - initReward) * std::min(1.0f, std::max(0.0f, (reward * reward))) << "." << std::endl;

	return totalReward;
}

float evaluateXOR(ctrnn::CTRNN &net, std::mt19937 &generator) {
	float inputs[4][2] {
		{ -1.0f, -1.0f },
		{ -1.0f, 1.0f },
		{ 1.0f, -1.0f },
		{ 1.0f, 1.0f }
	};

	float outputs[4] {
		0.0f,
		1.0f,
		1.0f,
		0.0f
	};

	float reward = 0.0f;
	float prevReward = 0.0f;

	float initReward = 0.0f;

	float totalReward = 0.0f;

	std::vector<float> values(4);

	for (size_t i = 0; i < 100; i++) {
		float newReward = 1.0f;

		float average = 0.0f;

		for (size_t j = 0; j < 4; j++) {
			net.setInput(0, inputs[j][0]);
			net.setInput(1, inputs[j][1]);
			net.setInput(2, inputs[j][0]);
			net.setInput(3, inputs[j][1]);
			net.setInput(4, (reward - prevReward) * 0.1f);

			for (size_t s = 0; s < 18; s++)
				net.step(1.0f, (reward - prevReward) * 0.1f, 0.1f, generator);

			newReward -= std::pow(std::abs(outputs[j] - net.getOutput(0)), 1.0f) * 0.25f;

			values[j] = net.getOutput(0);
			average += values[j];
		}

		average *= 0.25f;

		//for (size_t j = 0; j < 4; j++) {
		//	newReward -= 0.05f / (0.125f + 8.0f * std::abs(average - values[j]));
		//}

		prevReward = reward;
		reward = newReward;

		if (i == 0)
			initReward = std::min(1.0f, std::max(0.0f, reward));

		totalReward += reward * 0.5f;
	}

	//std::cout << "Finished AND experiment with total reward of " << (reward - initReward) * std::min(1.0f, std::max(0.0f, (reward * reward))) << "." << std::endl;

	return totalReward;
}

/*int main() {
	std::mt19937 generator(time(nullptr));

	ctrnn::GeneticAlgorithm ga;

	ga.create(40, 5, 1, 14, -0.5f, 0.5f, 1.01f, 2.0f, 0.05f, 1.0f, generator);

	for (size_t g = 0; g < 550; g++) {
		float maxFitness = -99999.0f;

		for (size_t i = 0; i < ga.getPopulationSize(); i++) {
			//ctrnn::CTRNN memberPoleBalancing = ga.getPopulationMember(i);
			ctrnn::CTRNN memberAND = ga.getPopulationMember(i);
			ctrnn::CTRNN memberOR = ga.getPopulationMember(i);
			ctrnn::CTRNN memberXOR = ga.getPopulationMember(i);

			ga.setFitness(i, evaluateAND(memberAND, generator) + evaluateOR(memberOR, generator) + evaluateXOR(memberXOR, generator));

			if (ga.getFitness(i) > maxFitness)
				maxFitness = ga.getFitness(i);
		}

		ga.generation(0.3f, 0.05f, 0.2f, 0.3f, 0.05f, 0.2f, 0.3f, 0.05f, 0.2f, 3.0f, 4, generator);

		std::cout << "Finished generation " << g << ". Peak fitness: " << maxFitness << std::endl;
	}

	float maxFitness = -99999.0f;
	size_t maxFitnessIndex = 0;

	for (size_t i = 0; i < ga.getPopulationSize(); i++) {
		//ctrnn::CTRNN memberPoleBalancing = ga.getPopulationMember(i);
		ctrnn::CTRNN memberAND = ga.getPopulationMember(i);
		ctrnn::CTRNN memberOR = ga.getPopulationMember(i);
		ctrnn::CTRNN memberXOR = ga.getPopulationMember(i);

		float f = evaluateAND(memberAND, generator) + evaluateOR(memberOR, generator) + evaluateXOR(memberXOR, generator);

		if (f > maxFitness) {
			maxFitness = f;
			maxFitnessIndex = i;
		}
	}

	ctrnn::CTRNN best = ga.getPopulationMember(maxFitnessIndex);

	best.clear();

	float inputs[4][2] {
		{ -1.0f, -1.0f },
		{ -1.0f, 1.0f },
		{ 1.0f, -1.0f },
		{ 1.0f, 1.0f }
	};

	float outputs[4] {
		0.0f,
		1.0f,
		1.0f,
		0.0f
	};

	float reward = 0.0f;
	float prevReward = 0.0f;

	float initReward = 0.0f;

	float totalReward = 0.0f;

	std::vector<float> values(4);

	for (size_t i = 0; i < 100; i++) {
		float newReward = 1.0f;

		float average = 0.0f;

		for (size_t j = 0; j < 4; j++) {
			best.setInput(0, inputs[j][0]);
			best.setInput(1, inputs[j][1]);
			best.setInput(2, inputs[j][0]);
			best.setInput(3, inputs[j][1]);
			best.setInput(4, (reward - prevReward) * 0.1f);

			for (size_t s = 0; s < 16; s++)
				best.step(1.0f, (reward - prevReward) * 0.1f, 0.1f, generator);

			newReward -= std::pow(std::abs(outputs[j] - best.getOutput(0)), 2.0f) * 0.25f;

			if (i == 49)
				std::cout << best.getOutput(0) << std::endl;

			values[j] = best.getOutput(0);
			average += values[j];
		}

		average *= 0.25f;

		//for (size_t j = 0; j < 4; j++) {
		//	newReward -= 0.05f / (0.125f + 8.0f * std::abs(average - values[j]));
		//}

		prevReward = reward;
		reward = newReward;

		if (i == 0)
			initReward = std::min(1.0f, std::max(0.0f, reward));

		totalReward += reward * 0.5f;
	}

	best.clear();

	sf::RenderWindow window;

	window.create(sf::VideoMode(800, 600), "Pole Balancing");

	window.setVerticalSyncEnabled(true);

	//window.setFramerateLimit(60);

	// -------------------------- Load Resources --------------------------

	sf::Texture backgroundTexture;
	sf::Texture cartTexture;
	sf::Texture poleTexture;

	backgroundTexture.loadFromFile("Resources/background.png");
	cartTexture.loadFromFile("Resources/cart.png");
	poleTexture.loadFromFile("Resources/pole.png");

	// --------------------------------------------------------------------

	sf::Sprite backgroundSprite;
	sf::Sprite cartSprite;
	sf::Sprite poleSprite;

	backgroundSprite.setTexture(backgroundTexture);
	cartSprite.setTexture(cartTexture);
	poleSprite.setTexture(poleTexture);

	backgroundSprite.setPosition(sf::Vector2f(0.0f, 0.0f));

	cartSprite.setOrigin(sf::Vector2f(static_cast<float>(cartSprite.getTexture()->getSize().x) * 0.5f, static_cast<float>(cartSprite.getTexture()->getSize().y)));
	poleSprite.setOrigin(sf::Vector2f(static_cast<float>(poleSprite.getTexture()->getSize().x) * 0.5f, static_cast<float>(poleSprite.getTexture()->getSize().y)));

	// ----------------------------- Physics ------------------------------

	float pixelsPerMeter = 128.0f;
	float poleLength = 1.0f;
	float g = -2.8f;
	float massMass = 40.0f;
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

	// ---------------------------- Game Loop -----------------------------

	bool quit = false;

	sf::Clock clock;

	float dt = 0.017f;

	float fitness = 0.0f;
	
	float lowPassFitness = 0.0f;

	bool reverseDirection = false;

	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

	do {
		clock.restart();

		// ----------------------------- Input -----------------------------

		sf::Event windowEvent;

		while (window.pollEvent(windowEvent))
		{
			switch (windowEvent.type)
			{
			case sf::Event::Closed:
				quit = true;
				break;
			}
		}

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
			quit = true;

		// Update fitness
		if (poleAngle < static_cast<float>(PI))
			fitness = -(static_cast<float>(PI) * 0.5f - poleAngle);
		else
			fitness = -(static_cast<float>(PI) * 0.5f - (static_cast<float>(PI) * 2.0f - poleAngle));

		//fitness = fitness - std::abs(poleAngleVel * 40.0f);

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::A))
			fitness = -cartX;
		else if (sf::Keyboard::isKeyPressed(sf::Keyboard::D))
			fitness = cartX;

		// ------------------------------ AI -------------------------------

		float dFitness = fitness - lowPassFitness;

		float error = dFitness * 10.0f;

		//agent.reinforceArp(std::min(1.0f, std::max(-1.0f, error)) * 0.5f + 0.5f, 0.1f, 0.05f);

		best.setInput(0, cartX * 0.25f);
		best.setInput(1, cartVelX);
		best.setInput(2, std::fmod(poleAngle + static_cast<float>(PI), 2.0f * static_cast<float>(PI)));
		best.setInput(3, poleAngleVel);

		best.setInput(4, dFitness * 10.0f);

		for (size_t s = 0; s < 16; s++)
			best.step(1.0f, dFitness, 0.1f, generator);

		lowPassFitness += (fitness - lowPassFitness) * 0.2f;

		//std::cout << dFitness * 0.5f - 0.001f << std::endl;

		float dir = std::min(1.0f, std::max(-1.0f, 2.0f * best.getOutput(0) - 1.0f));

		//dir = 1.4f * (dir * 2.0f - 1.0f);

		float agentForce = 4000.0f * dir;
		//float agentForce = 4000.0f * agent.getOutput(0);

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

		if (std::abs(cartVelX) < maxSpeed) {
			force = std::max(-4000.0f, std::min(4000.0f, agentForce));

			if (sf::Keyboard::isKeyPressed(sf::Keyboard::Left))
				force = -4000.0f;

			if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right))
				force = 4000.0f;
		}

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

		// ---------------------------- Rendering ----------------------------

		window.clear();

		window.draw(backgroundSprite);

		cartSprite.setPosition(sf::Vector2f(static_cast<float>(window.getSize().x) * 0.5f + pixelsPerMeter * cartX, static_cast<float>(window.getSize().y) * 0.5f + 3.0f));

		window.draw(cartSprite);

		poleSprite.setPosition(cartSprite.getPosition() + sf::Vector2f(0.0f, -45.0f));
		poleSprite.setRotation(poleAngle * 180.0f / static_cast<float>(PI) + 180.0f);

		window.draw(poleSprite);

		// -------------------------------------------------------------------

		window.display();

		dt = clock.getElapsedTime().asSeconds();
	} while (!quit);
}*/

/*int main() {
	std::mt19937 generator(time(nullptr));

	sf::RenderWindow window;
	window.create(sf::VideoMode(512, 512), "AILib", sf::Style::Default);

	dnf::Field field;
	field.createRandom(2, 32, 2, 0.75f, 0.0f, 1.0f, generator);

	float size = std::min(window.getSize().x / static_cast<float>(field.getSize()), window.getSize().y / static_cast<float>(field.getSize()));

	sf::Vector2f boxSize(size, size);

	// -------------------------- Simulation Loop -------------------------

	bool quit = false;

	sf::Clock clock;

	float dt = 0.017f;

	float switchPatternTimer = 0.0f;
	float switchPatternTime = 280.0f;

	int pattern = 0;

	float inputPattern[4][2] = {
		{ 0.0f, 0.0f },
		{ 0.0f, 1.0f },
		{ 1.0f, 0.0f },
		{ 1.0f, 1.0f }
	};

	float outputPattern[4] = {
		0.0f,
		0.0f,
		0.0f,
		0.0f
	};

	float reward = 0.0f;
	float prevReward = 0.0f;

	float averageReward = 0.0f;

	float fieldAverage = 0.0f;

	do {
		clock.restart();

		// ----------------------------- Input -----------------------------

		sf::Event windowEvent;

		while (window.pollEvent(windowEvent))
		{
			switch (windowEvent.type)
			{
			case sf::Event::Closed:
				quit = true;
				break;
			}
		}

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
			quit = true;

		std::vector<float> input(4);

		input[0] = inputPattern[pattern][0];
		input[1] = inputPattern[pattern][1];

		// -------------------------------------------------------------------

		window.clear();

		for (int i = 0; i < 1; i++)
		field.step(input, 0.1f, 0.2f, 0.3f, (reward - prevReward) * 0.001f, 0.05f, 0.001f, 0.001f, 0.002f, 0.5f);

		// Draw grid
		int lIndex = 0;

		for (int x = 0; x < field.getSize(); x++)
		for (int y = 0; y < field.getSize(); y++) {
			sf::RectangleShape rectShape;

			rectShape.setPosition(x * boxSize.x, y * boxSize.y);
			rectShape.setSize(boxSize);

			float positivePotentialColor = field.getOutputE(x, y);
			float negativePotentialColor = field.getOutputE(x, y);

			rectShape.setFillColor(sf::Color(positivePotentialColor * 255.0f, negativePotentialColor * 255.0f, 0));

			window.draw(rectShape);

			lIndex++;
		}

		switchPatternTimer += 1.0f;

		if (switchPatternTimer > switchPatternTime) {
			prevReward = reward;

			reward = -std::pow(std::abs(outputPattern[pattern] - std::min(1.0f, std::max(0.0f, field.getOutputE(16, 16)))), 2.0f);

			averageReward += (reward - averageReward) * 0.1f;

			std::cout << field.getOutputE(16, 16) << "    " << averageReward << "     " << reward << std::endl;

			switchPatternTimer -= switchPatternTime;

			pattern = (pattern + 1) % 4;

			fieldAverage = 0.0f;
		}

		// -------------------------------------------------------------------

		window.display();

		dt = clock.getElapsedTime().asSeconds();
	} while (!quit);

	return 0;
}*/

/*int main() {
	std::mt19937 generator(time(nullptr));

	float inputs[4][2] {
		{ 0.0f, 0.0f },
		{ 0.0f, 1.0f },
		{ 1.0f, 0.0f },
		{ 1.0f, 1.0f }
	};

	float outputs[4] {
		0.0f,
		1.0f,
		1.0f,
		0.0f
	};

	rbf::RBFNetwork fa;

	fa.createRandom(2, 32, 1, -0.5f, 0.5f, 0.01f, 2.0f, -0.1f, 0.1f, generator);

	std::vector<float> in(2);
	std::vector<float> out(1);

	for (int i = 0; i < 50; i++) {
		//fa.clearGradient();

		for (int k = 0; k < 4; k++) {
			in[0] = inputs[k][0];
			in[1] = inputs[k][1];

			//fa.process(in, out);

			//fa.accumulateGradient(in, std::vector<float>(1, outputs[k]));

			fa.getOutput(in, out);

			fa.learnFeatures(in, 0.2f, 0.2f, 0.5f);

			fa.update(in, out, std::vector<float>(1, outputs[k]), 0.1f);
		}

		//fa.scaleGradient(0.25f);

		//fa.moveAlongGradientRMS(0.1f, 0.1f, 0.8f, 0.0f, 0.01f);
	}

	for (int k = 0; k < 4; k++) {
		in[0] = inputs[k][0];
		in[1] = inputs[k][1];

		//fa.process(in, out);

		fa.getOutput(in, out);

		std::cout << out[0] << std::endl;
	}

	system("pause");

	float reward = 0.0f;
	float prevReward = 0.0f;

	float initReward = 0.0f;

	float totalReward = 0.0f;

	sf::RenderWindow window;

	window.create(sf::VideoMode(1600, 600), "Pole Balancing");

	window.setVerticalSyncEnabled(true);

	//window.setFramerateLimit(60);

	// -------------------------- Load Resources --------------------------

	sf::Texture backgroundTexture;
	sf::Texture cartTexture;
	sf::Texture poleTexture;

	backgroundTexture.loadFromFile("Resources/background.png");
	cartTexture.loadFromFile("Resources/cart.png");
	poleTexture.loadFromFile("Resources/pole.png");

	// --------------------------------------------------------------------

	sf::Sprite backgroundSprite;
	sf::Sprite cartSprite;
	sf::Sprite poleSprite;

	backgroundSprite.setTexture(backgroundTexture);
	cartSprite.setTexture(cartTexture);
	poleSprite.setTexture(poleTexture);

	backgroundSprite.setPosition(sf::Vector2f(0.0f, 0.0f));

	cartSprite.setOrigin(sf::Vector2f(static_cast<float>(cartSprite.getTexture()->getSize().x) * 0.5f, static_cast<float>(cartSprite.getTexture()->getSize().y)));
	poleSprite.setOrigin(sf::Vector2f(static_cast<float>(poleSprite.getTexture()->getSize().x) * 0.5f, static_cast<float>(poleSprite.getTexture()->getSize().y)));

	// ----------------------------- Physics ------------------------------

	float pixelsPerMeter = 128.0f;
	float poleLength = 1.0f;
	float g = -2.8f;
	float massMass = 40.0f;
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

	// ---------------------------- Game Loop -----------------------------

	bool quit = false;

	sf::Clock clock;

	float dt = 0.017f;

	float fitness = 0.0f;
	float prevFitness = 0.0f;

	float lowPassFitness = 0.0f;

	bool reverseDirection = false;

	bool trainMode = true;

	bool tDownLastFrame = false;

	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

	//deep::AutoLSTM alstm;

	//elman::ElmanNetwork en;

	//en.createRandom(4, 1, 10, -0.5f, 0.5f, generator);

	//lstm::LSTMActorCritic lstmAC;

	//lstmAC.createRandom(4, 1, 1, 24, 2, 1, 1, 24, 2, 1, -0.5f, 0.5f, generator);

	htmrl::HTMRLDiscreteAction htmRL;
	std::vector<htmrl::HTMRLDiscreteAction::RegionDesc> regionDescs(1);
	regionDescs[0]._regionWidth = 32;
	regionDescs[0]._regionHeight = 16;

	std::vector<float> condensed;

	htmRL.createRandom(2, 1, 32, 32, 2, 2, 3, 1, 32, 0.1f, regionDescs, generator);

	//falcon::Falcon fal;
	//fal.create(4, 1);

	//deep::FERL ferl;
	//ferl.createRandom(4, 1, 24, 0.1f, generator);

	//nn::SOMQAgent sqa;

	//sqa.createRandom(4, 1, 2, 60, nn::BrownianPerturbation(), -0.5f, 0.5f, generator);

	sf::Font font;

	font.loadFromFile("Resources/pixelated.ttf");

	sf::Image image;

	image.create(htmRL.getCondenseBufferWidth(), htmRL.getCondenseBufferHeight());

	sf::RenderTexture rt;

	rt.create(800, 600);

	sf::plot::Plot plot;

	plot.setSize(sf::Vector2f(rt.getSize().x, rt.getSize().y));
	plot.setTitle("Avg. Reward");
	plot.setFont("Resources/arial.ttf");
	plot.setXLabel("Time (seconds)");
	plot.setYLabel("Avg. Reward");
	plot.setBackgroundColor(sf::Color::White);
	plot.setTitleColor(sf::Color::Black);
	plot.setPosition(sf::Vector2f(0.0f, 0.0f));

	sf::plot::Curve &cAverage = plot.createCurve("Avg. Reward", sf::Color::Red);

	cAverage.setFill(false);

	plot.prepare();

	float plotMin = 99999.0f;
	float plotMax = -99999.0f;

	rt.draw(plot);
	rt.display();

	float avgReward = 0.0f;
	float avgRewardDecay = 0.003f;

	float totalTime = 0.0f;

	float plotUpdateTimer = 0.0f;

	do {
		clock.restart();

		// ----------------------------- Input -----------------------------

		sf::Event windowEvent;

		while (window.pollEvent(windowEvent))
		{
			switch (windowEvent.type)
			{
			case sf::Event::Closed:
				quit = true;
				break;
			}
		}

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
			quit = true;

		// Update fitness
		if (poleAngle < static_cast<float>(PI))
			fitness = -(static_cast<float>(PI) * 0.5f - poleAngle);
		else
			fitness = -(static_cast<float>(PI) * 0.5f - (static_cast<float>(PI) * 2.0f - poleAngle));

		fitness += static_cast<float>(PI) * 0.5f;

		//fitness = fitness - std::abs(poleAngleVel * 1.0f);

		//fitness = -std::abs(cartX);

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::A))
			fitness = -cartX;
		else if (sf::Keyboard::isKeyPressed(sf::Keyboard::D))
			fitness = cartX;

		// ------------------------------ AI -------------------------------

		float dFitness = fitness - prevFitness;

		//reward = dFitness * 5.0f;

		reward = fitness * 0.2f;

		if (totalTime == 0.0f)
			avgReward = reward;
		else
			avgReward = (1.0f - avgRewardDecay) * avgReward + avgRewardDecay * reward;

		//agent.reinforceArp(std::min(1.0f, std::max(-1.0f, error)) * 0.5f + 0.5f, 0.1f, 0.05f);

		//agent.setInput(0, 0, cartX * 0.5f);
		//agent.setInput(0, 1, cartVelX);
		//agent.setInput(1, 0, std::fmod(poleAngle + static_cast<float>(PI), 2.0f * static_cast<float>(PI) / PI * 2.0f - 1.0f));
		//agent.setInput(1, 1, poleAngleVel * 0.1f);

		//agent.step(dFitness * 6.0f, 0.97f, 0.5f, 0.1f, 0.001f, regionDesc, 0.2f, 0.3f, 0.05f, generator);

		//lstmAC.setInput(0, cartX * 0.25f);
		//lstmAC.setInput(1, cartVelX);
		//lstmAC.setInput(2, std::fmod(poleAngle + static_cast<float>(PI), 2.0f * static_cast<float>(PI)));
		//lstmAC.setInput(3, poleAngleVel);

		//lstmAC.step(dFitness * 10.0f, 0.2f, 0.01f, 0.02f, 0.4f, 0.0001f, 0.97f, 0.5f, 0.5f, 0.05f, 0.2f, 0.2f, 0.99f, generator);

		//fal.update(fitness, 0.1f, 0.9f, 0.2f, fieldParams, 1.5f, 0.8f, 0.07f, generator);

		//sqa.step(fitness, 0.3f, 0.96f, 0.5f, 0.07f, dt, generator);

		std::vector<float> state(4);

		state[0] = std::min(1.0f, std::max(-1.0f, cartX * 0.333f));
		state[1] = std::min(1.0f, std::max(-1.0f, cartVelX * 0.25f));
		state[2] = std::fmod(poleAngle + static_cast<float>(PI), 2.0f * static_cast<float>(PI)) / (2.0f * static_cast<float>(PI)) * 2.0f - 1.0f;
		state[3] = std::min(1.0f, std::max(-1.0f, poleAngleVel * 0.25f));

		std::vector<float> output;

		htmRL.setInput(0, 0, 0, state[0]);
		htmRL.setInput(0, 0, 1, state[1]);
		htmRL.setInput(1, 0, 0, state[2]);
		htmRL.setInput(1, 0, 1, state[3]);

		int action;

		action = htmRL.step(reward, 0.5f, 0.05f, 0.005f, 0.2f, 0.997f, 0.99f, 1.0f, 0.15f, 0.0f, 0.0f, 0.0f, 0.05f, generator, condensed);

		output.resize(1);
		output[0] = action - 1;

		//ferl.step(state, output, reward, 0.0001f, 0.97f, 0.8f, 6.0f, 8, 8, 0.05f, 0.05f, 0.1f, generator);

		lowPassFitness += (fitness - lowPassFitness) * 0.2f;

		float agentForce = 0.0f;

		//if (!trainMode) {
		//float dir = std::min(1.0f, std::max(-1.0f, agent.getOutput(0)));

		float dir = output[0];

			//dir = 1.4f * (dir * 2.0f - 1.0f);

			agentForce = 4000.0f * dir;
		//}
		//float agentForce = 4000.0f * agent.getOutput(0);

			prevFitness = fitness;

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

		if (std::abs(cartVelX) < maxSpeed) {
			force = std::max(-4000.0f, std::min(4000.0f, agentForce));

			if (sf::Keyboard::isKeyPressed(sf::Keyboard::Left))
				force = -4000.0f;

			if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right))
				force = 4000.0f;
		}

		//if (trainMode) {
		//	en.calculateGradient(std::vector<float>(1, force / 4000.0f));

		//	en.moveAlongGradient(0.01f);
		//}

		//en.updateContext();

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

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::T)) {
			if (!tDownLastFrame) {
				trainMode = !trainMode;
			}
			
			tDownLastFrame = true;
		}
		else
			tDownLastFrame = false;

		// ---------------------------- Rendering ----------------------------

		window.clear();

		window.draw(backgroundSprite);

		cartSprite.setPosition(sf::Vector2f(800.0f * 0.5f + pixelsPerMeter * cartX, 600.0f * 0.5f + 3.0f));

		window.draw(cartSprite);

		poleSprite.setPosition(cartSprite.getPosition() + sf::Vector2f(0.0f, -45.0f));
		poleSprite.setRotation(poleAngle * 180.0f / static_cast<float>(PI) + 180.0f);

		window.draw(poleSprite);

		if (plotUpdateTimer >= 1.0f) {
			plotUpdateTimer -= 1.0f;

			plotMin = std::min<float>(avgReward, plotMin);
			plotMax = std::max<float>(avgReward, plotMax);

			cAverage.addValue(avgReward);

			plot.prepare();

			cAverage.prepare(sf::Vector2f(0.0f, totalTime), sf::Vector2f(plotMin, plotMax));

			rt.draw(plot);

			rt.display();
		}

		sf::Sprite plotSprite;

		plotSprite.setPosition(800.0f, 0.0f);

		plotSprite.setTexture(rt.getTexture());

		window.draw(plotSprite);

		if (trainMode) {
			sf::Text text;

			text.setFont(font);

			text.setString("Train");

			text.setPosition(10.0f, 10.0f);

			text.setColor(sf::Color::Blue);

			window.draw(text);
		}

		for (size_t i = 0; i < htmRL.getCondenseBufferWidth() * htmRL.getCondenseBufferHeight(); i++) {
			int x = i % htmRL.getCondenseBufferWidth();
			int y = i / htmRL.getCondenseBufferWidth();

			sf::Color c;
			
			c.r = condensed[i] * 255.0f;
			c.g = 0;
			c.b = 0;

			image.setPixel(x, y, c);
		}

		sf::Texture t;

		t.loadFromImage(image);

		sf::Sprite s;

		s.setPosition(800.0f - 256.0f, 0.0f);

		s.setScale(256.0f / htmRL.getCondenseBufferWidth(), 256.0f / htmRL.getCondenseBufferWidth());

		s.setTexture(t);

		window.draw(s);
		
		// -------------------------------------------------------------------

		window.display();

		//dt = clock.getElapsedTime().asSeconds();

		totalTime += dt;
		plotUpdateTimer += dt;
	} while (!quit);
}*/

/*int main() {
	std::mt19937 generator(time(nullptr));

	float reward = 0.0f;
	float prevReward = 0.0f;

	float initReward = 0.0f;

	float totalReward = 0.0f;

	sf::RenderWindow window;

	window.create(sf::VideoMode(1600, 600), "Pole Balancing");

	window.setVerticalSyncEnabled(true);

	//window.setFramerateLimit(60);

	// -------------------------- Load Resources --------------------------

	sf::Texture backgroundTexture;
	sf::Texture cartTexture;
	sf::Texture poleTexture;

	backgroundTexture.loadFromFile("Resources/background.png");
	cartTexture.loadFromFile("Resources/cart.png");
	poleTexture.loadFromFile("Resources/pole.png");

	sf::Texture inputCartTexture;
	sf::Texture inputPoleTexture;

	inputCartTexture.loadFromFile("Resources/inputCart.png");
	inputPoleTexture.loadFromFile("Resources/inputPole.png");

	// --------------------------------------------------------------------

	sf::Sprite backgroundSprite;
	sf::Sprite cartSprite;
	sf::Sprite poleSprite;

	backgroundSprite.setTexture(backgroundTexture);
	cartSprite.setTexture(cartTexture);
	poleSprite.setTexture(poleTexture);

	backgroundSprite.setPosition(sf::Vector2f(0.0f, 0.0f));

	cartSprite.setOrigin(sf::Vector2f(static_cast<float>(cartSprite.getTexture()->getSize().x) * 0.5f, static_cast<float>(cartSprite.getTexture()->getSize().y)));
	poleSprite.setOrigin(sf::Vector2f(static_cast<float>(poleSprite.getTexture()->getSize().x) * 0.5f, static_cast<float>(poleSprite.getTexture()->getSize().y)));

	sf::Sprite inputCartSprite;
	sf::Sprite inputPoleSprite;

	inputCartSprite.setTexture(inputCartTexture);
	inputPoleSprite.setTexture(inputPoleTexture);

	inputCartSprite.setOrigin(sf::Vector2f(static_cast<float>(inputCartSprite.getTexture()->getSize().x) * 0.5f, static_cast<float>(inputCartSprite.getTexture()->getSize().y)));
	inputPoleSprite.setOrigin(sf::Vector2f(static_cast<float>(inputPoleSprite.getTexture()->getSize().x) * 0.5f, static_cast<float>(inputPoleSprite.getTexture()->getSize().y)));


	// ----------------------------- Physics ------------------------------

	float pixelsPerMeter = 128.0f;
	float inputPixelsPerMeter = 8.0f;
	float poleLength = 1.0f;
	float g = -2.8f;
	float massMass = 40.0f;
	float cartMass = 2.0f;
	sf::Vector2f massPos(0.0f, poleLength);
	sf::Vector2f massVel(0.0f, 0.0f);
	float poleAngle = static_cast<float>(PI)* 0.0f;
	float poleAngleVel = 0.0f;
	float poleAngleAccel = 0.0f;
	float cartX = 0.0f;
	float cartVelX = 0.0f;
	float cartAccelX = 0.0f;
	float poleRotationalFriction = 0.008f;
	float cartMoveRadius = 1.8f;
	float cartFriction = 0.02f;
	float maxSpeed = 3.0f;

	// ---------------------------- Game Loop -----------------------------

	bool quit = false;

	sf::Clock clock;

	float dt = 0.017f;

	float fitness = 0.0f;
	float prevFitness = 0.0f;

	float lowPassFitness = 0.0f;

	bool reverseDirection = false;

	bool trainMode = true;

	bool tDownLastFrame = false;

	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

	//deep::AutoLSTM alstm;

	//elman::ElmanNetwork en;

	//en.createRandom(4, 1, 10, -0.5f, 0.5f, generator);

	//lstm::LSTMActorCritic lstmAC;

	//lstmAC.createRandom(4, 1, 1, 24, 2, 1, 1, 24, 2, 1, -0.5f, 0.5f, generator);

	htmrl::HTMRLDiscreteAction htmRL;
	std::vector<htmrl::HTMRLDiscreteAction::RegionDesc> regionDescs(3);
	regionDescs[0]._connectionRadius = 1;
	regionDescs[0]._initInhibitionRadius = 1.0f;
	regionDescs[0]._desiredLocalActivity = 1;
	regionDescs[0]._learningRadius = 1;
	regionDescs[0]._regionWidth = 128;
	regionDescs[0]._regionHeight = 128;

	regionDescs[1]._connectionRadius = 2;
	regionDescs[1]._initInhibitionRadius = 2.0f;
	regionDescs[1]._desiredLocalActivity = 2;
	regionDescs[1]._learningRadius = 2;
	regionDescs[1]._regionWidth = 64;
	regionDescs[1]._regionHeight = 64;

	regionDescs[2]._connectionRadius = 2;
	regionDescs[2]._initInhibitionRadius = 2.0f;
	regionDescs[2]._desiredLocalActivity = 2;
	regionDescs[2]._learningRadius = 2;
	regionDescs[2]._regionWidth = 32;
	regionDescs[2]._regionHeight = 32;

	std::vector<float> condensed;

	htmRL.createRandom(32, 32, 4, 4, 2, 2, 3, 1, 32, 0.1f, regionDescs, generator);

	//falcon::Falcon fal;
	//fal.create(4, 1);

	//deep::FERL ferl;
	//ferl.createRandom(4, 1, 24, 0.1f, generator);

	//nn::SOMQAgent sqa;

	//sqa.createRandom(4, 1, 2, 60, nn::BrownianPerturbation(), -0.5f, 0.5f, generator);

	sf::Font font;

	font.loadFromFile("Resources/pixelated.ttf");

	sf::Image image;

	image.create(htmRL.getCondenseBufferWidth(), htmRL.getCondenseBufferHeight());

	sf::RenderTexture inputRT;

	inputRT.create(64, 32);

	sf::RenderTexture rt;

	rt.create(800, 600);

	sf::plot::Plot plot;

	plot.setSize(sf::Vector2f(rt.getSize().x, rt.getSize().y));
	plot.setTitle("Avg. Reward");
	plot.setFont("Resources/arial.ttf");
	plot.setXLabel("Time (seconds)");
	plot.setYLabel("Avg. Reward");
	plot.setBackgroundColor(sf::Color::White);
	plot.setTitleColor(sf::Color::Black);
	plot.setPosition(sf::Vector2f(0.0f, 0.0f));

	sf::plot::Curve &cAverage = plot.createCurve("Avg. Reward", sf::Color::Red);

	cAverage.setFill(false);

	plot.prepare();

	float plotMin = 99999.0f;
	float plotMax = -99999.0f;

	rt.draw(plot);
	rt.display();

	float avgReward = 0.0f;
	float avgRewardDecay = 0.003f;

	float totalTime = 0.0f;

	float plotUpdateTimer = 0.0f;

	do {
		clock.restart();

		// ----------------------------- Input -----------------------------

		sf::Event windowEvent;

		while (window.pollEvent(windowEvent))
		{
			switch (windowEvent.type)
			{
			case sf::Event::Closed:
				quit = true;
				break;
			}
		}

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
			quit = true;

		// Update fitness
		if (poleAngle < static_cast<float>(PI))
			fitness = -(static_cast<float>(PI)* 0.5f - poleAngle);
		else
			fitness = -(static_cast<float>(PI)* 0.5f - (static_cast<float>(PI)* 2.0f - poleAngle));

		fitness += static_cast<float>(PI) * 0.5f;

		//fitness = fitness - std::abs(poleAngleVel * 1.0f);

		//fitness = -std::abs(cartX);

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::A))
			fitness = -cartX;
		else if (sf::Keyboard::isKeyPressed(sf::Keyboard::D))
			fitness = cartX;

		// ------------------------------ AI -------------------------------

		float dFitness = fitness - prevFitness;

		//reward = dFitness * 5.0f;

		reward = fitness * 0.2f;

		if (totalTime == 0.0f)
			avgReward = reward;
		else
			avgReward = (1.0f - avgRewardDecay) * avgReward + avgRewardDecay * reward;

		//agent.reinforceArp(std::min(1.0f, std::max(-1.0f, error)) * 0.5f + 0.5f, 0.1f, 0.05f);

		//agent.setInput(0, 0, cartX * 0.5f);
		//agent.setInput(0, 1, cartVelX);
		//agent.setInput(1, 0, std::fmod(poleAngle + static_cast<float>(PI), 2.0f * static_cast<float>(PI) / PI * 2.0f - 1.0f));
		//agent.setInput(1, 1, poleAngleVel * 0.1f);

		//agent.step(dFitness * 6.0f, 0.97f, 0.5f, 0.1f, 0.001f, regionDesc, 0.2f, 0.3f, 0.05f, generator);

		//lstmAC.setInput(0, cartX * 0.25f);
		//lstmAC.setInput(1, cartVelX);
		//lstmAC.setInput(2, std::fmod(poleAngle + static_cast<float>(PI), 2.0f * static_cast<float>(PI)));
		//lstmAC.setInput(3, poleAngleVel);

		//lstmAC.step(dFitness * 10.0f, 0.2f, 0.01f, 0.02f, 0.4f, 0.0001f, 0.97f, 0.5f, 0.5f, 0.05f, 0.2f, 0.2f, 0.99f, generator);

		//fal.update(fitness, 0.1f, 0.9f, 0.2f, fieldParams, 1.5f, 0.8f, 0.07f, generator);

		//sqa.step(fitness, 0.3f, 0.96f, 0.5f, 0.07f, dt, generator);

		//std::vector<float> state(4);

		//state[0] = std::min(1.0f, std::max(-1.0f, cartX * 0.333f));
		//state[1] = std::min(1.0f, std::max(-1.0f, cartVelX * 0.25f));
		//state[2] = std::fmod(poleAngle + static_cast<float>(PI), 2.0f * static_cast<float>(PI)) / (2.0f * static_cast<float>(PI)) * 2.0f - 1.0f;
		//state[3] = std::min(1.0f, std::max(-1.0f, poleAngleVel * 0.25f));

		//htmRL.setInput(0, 0, 0, state[0]);
		//htmRL.setInput(0, 0, 1, state[1]);
		//htmRL.setInput(1, 0, 0, state[2]);
		//htmRL.setInput(1, 0, 1, state[3]);

		sf::Image img = inputRT.getTexture().copyToImage();

		for (int i = 0; i < htmRL.getInputWidth() * htmRL.getInputHeight(); i++) {
			int x = i % htmRL.getInputWidth();
			int y = i / htmRL.getInputWidth();

			int xl = x * 2;
			int yl = y;

			htmRL.setInput(x, y, 0, img.getPixel(xl, yl).r / 255.0f * 2.0f - 1.0f);
			htmRL.setInput(x, y, 1, img.getPixel(xl + 1, yl).r / 255.0f * 2.0f - 1.0f);
		}

		std::vector<float> output;

		int action;

		action = htmRL.step(reward, 0.5f, 0.05f, 0.005f, 0.2f, 0.994f, 0.99f, 1.0f, 0.15f, 0.0f, 0.0f, 0.0f, 0.05f, generator, condensed);

		output.resize(1);
		output[0] = action - 1;

		//ferl.step(state, output, reward, 0.0001f, 0.97f, 0.8f, 6.0f, 8, 8, 0.05f, 0.05f, 0.1f, generator);

		lowPassFitness += (fitness - lowPassFitness) * 0.2f;

		float agentForce = 0.0f;

		//if (!trainMode) {
		//float dir = std::min(1.0f, std::max(-1.0f, agent.getOutput(0)));

		float dir = output[0];

		//dir = 1.4f * (dir * 2.0f - 1.0f);

		agentForce = 4000.0f * dir;
		//}
		//float agentForce = 4000.0f * agent.getOutput(0);

		prevFitness = fitness;

		// ---------------------------- Physics ----------------------------

		float pendulumCartAccelX = cartAccelX;

		if (cartX < -cartMoveRadius)
			pendulumCartAccelX = 0.0f;
		else if (cartX > cartMoveRadius)
			pendulumCartAccelX = 0.0f;

		poleAngleAccel = pendulumCartAccelX * std::cos(poleAngle) + g * std::sin(poleAngle);
		poleAngleVel += -poleRotationalFriction * poleAngleVel + poleAngleAccel * dt;
		poleAngle += poleAngleVel * dt;

		massPos = sf::Vector2f(cartX + std::cos(poleAngle + static_cast<float>(PI)* 0.5f) * poleLength, std::sin(poleAngle + static_cast<float>(PI)* 0.5f) * poleLength);

		float force = 0.0f;

		if (std::abs(cartVelX) < maxSpeed) {
			force = std::max(-4000.0f, std::min(4000.0f, agentForce));

			if (sf::Keyboard::isKeyPressed(sf::Keyboard::Left))
				force = -4000.0f;

			if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right))
				force = 4000.0f;
		}

		//if (trainMode) {
		//	en.calculateGradient(std::vector<float>(1, force / 4000.0f));

		//	en.moveAlongGradient(0.01f);
		//}

		//en.updateContext();

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
			poleAngle += static_cast<float>(PI)* 2.0f;

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::T)) {
			if (!tDownLastFrame) {
				trainMode = !trainMode;
			}

			tDownLastFrame = true;
		}
		else
			tDownLastFrame = false;

		// ---------------------------- Rendering ----------------------------

		// Render to input buffer
		inputRT.clear();

		inputCartSprite.setPosition(sf::Vector2f(inputRT.getSize().x * 0.5f + inputPixelsPerMeter * cartX, inputRT.getSize().y * 0.5f + 4.0f));

		inputRT.draw(inputCartSprite);

		inputPoleSprite.setPosition(inputCartSprite.getPosition() + sf::Vector2f(0.0f, -4.0f));
		inputPoleSprite.setRotation(poleAngle * 180.0f / static_cast<float>(PI) + 180.0f);

		inputRT.draw(inputPoleSprite);

		inputRT.display();

		window.clear();

		window.draw(backgroundSprite);

		cartSprite.setPosition(sf::Vector2f(800.0f * 0.5f + pixelsPerMeter * cartX, 600.0f * 0.5f + 3.0f));

		window.draw(cartSprite);

		poleSprite.setPosition(cartSprite.getPosition() + sf::Vector2f(0.0f, -45.0f));
		poleSprite.setRotation(poleAngle * 180.0f / static_cast<float>(PI)+180.0f);

		window.draw(poleSprite);

		if (plotUpdateTimer >= 1.0f) {
			plotUpdateTimer -= 1.0f;

			plotMin = std::min<float>(avgReward, plotMin);
			plotMax = std::max<float>(avgReward, plotMax);

			cAverage.addValue(avgReward);

			plot.prepare();

			cAverage.prepare(sf::Vector2f(0.0f, totalTime), sf::Vector2f(plotMin, plotMax));

			rt.draw(plot);

			rt.display();
		}

		sf::Sprite plotSprite;

		plotSprite.setPosition(800.0f, 0.0f);

		plotSprite.setTexture(rt.getTexture());

		window.draw(plotSprite);

		if (trainMode) {
			sf::Text text;

			text.setFont(font);

			text.setString("Train");

			text.setPosition(10.0f, 10.0f);

			text.setColor(sf::Color::Blue);

			window.draw(text);
		}

		for (size_t i = 0; i < htmRL.getCondenseBufferWidth() * htmRL.getCondenseBufferHeight(); i++) {
			int x = i % htmRL.getCondenseBufferWidth();
			int y = i / htmRL.getCondenseBufferWidth();

			sf::Color c;

			c.r = condensed[i] * 255.0f;
			c.g = 0;
			c.b = 0;

			image.setPixel(x, y, c);
		}

		sf::Texture t;

		t.loadFromImage(image);

		sf::Sprite s;

		s.setPosition(800.0f - 256.0f, 0.0f);

		s.setScale(256.0f / htmRL.getCondenseBufferWidth(), 256.0f / htmRL.getCondenseBufferWidth());

		s.setTexture(t);

		window.draw(s);

		sf::Sprite inputSprite;

		inputSprite.setTexture(inputRT.getTexture());

		inputSprite.setPosition(0, 0);
		inputSprite.setScale(4.0f, 4.0f);

		window.draw(inputSprite);

		// -------------------------------------------------------------------

		window.display();

		//dt = clock.getElapsedTime().asSeconds();

		totalTime += dt;
		plotUpdateTimer += dt;
	} while (!quit);
}*/

/*int main() {
	sf::RenderWindow window;

	window.create(sf::VideoMode(800, 600), "Mountain Car");

	//window.setVerticalSyncEnabled(true);

	window.setFramerateLimit(60);

	// -------------------------- Load Resources --------------------------

	sf::Texture backgroundTexture;
	sf::Texture cartTexture;

	cartTexture.loadFromFile("Resources/car.png");

	// --------------------------------------------------------------------

	sf::Sprite cartSprite;

	cartSprite.setTexture(cartTexture);

	cartSprite.setOrigin(sf::Vector2f(static_cast<float>(cartSprite.getTexture()->getSize().x) * 0.5f, static_cast<float>(cartSprite.getTexture()->getSize().y) * 0.5f));
	cartSprite.setScale(sf::Vector2f(0.25f, 0.25f));

	// ----------------------------- Physics ------------------------------

	float pixelsPerMeter = 128.0f;

	float velocity = 0.0f;
	float position = -0.5f;

	// ------------------------------- AI ---------------------------------

	std::mt19937 generator(time(nullptr));

	chtm::CHTMRL htm;

	htm.createRandom(2, 2, 16, 16, 3, 1, 3, -0.75f, 0.75f, 0.01f, 0.1f, -0.1f, 0.1f, -0.1f, 0.1f, -0.1f, 0.1f, -0.1f, 0.1f, generator);

	// ---------------------------- Game Loop -----------------------------

	bool quit = false;

	sf::Clock clock;

	float dt = 0.017f;

	float fitness = 0.0f;
	float prevFitness = 0.0f;

	prevFitness = fitness;
	fitness = 0.0f;

	std::vector<sf::Vertex> hills;

	for (float p = -3.141f; p < 3.141f;) {
		sf::Vertex v1;
		v1.position = sf::Vector2f(static_cast<float>(window.getSize().x) * 0.5f + p * pixelsPerMeter, static_cast<float>(window.getSize().y) * 0.5f + std::sin(p * 3.0f) * pixelsPerMeter * -0.333f);
		v1.color = sf::Color::Red;

		p += 0.1f;

		sf::Vertex v2;
		v2.position = sf::Vector2f(static_cast<float>(window.getSize().x) * 0.5f + p * pixelsPerMeter, static_cast<float>(window.getSize().y) * 0.5f + std::sin(p * 3.0f) * pixelsPerMeter * -0.333f);
		v2.color = sf::Color::Red;

		hills.push_back(v1);
		hills.push_back(v2);
	}

	std::vector<float> prevState(4, 0.0f);

	sf::Image image;

	image.create(16, 16);

	do {
		clock.restart();

		// ----------------------------- Input -----------------------------

		sf::Event windowEvent;

		while (window.pollEvent(windowEvent))
		{
			switch (windowEvent.type)
			{
			case sf::Event::Closed:
				quit = true;
				break;
			}
		}

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
			quit = true;

		// ---------------------------- Fitness ------------------------------

		prevFitness = fitness;
		fitness = (std::sin(position * 3.0f) + 1.0f) * 0.5f;

		// ------------------------------ AI -------------------------------

		float dFitness = (fitness - prevFitness) * 100.0f;

		std::vector<float> state(4);

		state[0] = 1.2f * (position + 0.52f);
		state[1] = velocity * 15.0f;
		state[2] = prevState[2];
		state[3] = prevState[3];

		std::vector<float> action(4);
		std::vector<bool> actionMask(4, false);

		actionMask[2] = true;
		actionMask[3] = true;

		htm.step(fitness, state, actionMask, action, 0.5f, 16, 0.1f, 0.9f, 0.0f, 0.1f, 16.0f, 1.0f, 8, 16.0f, 16.0f, 16.0f, 0.0001f, 0.01f, 0.01f, 0.01f, 0.1f, 0.1f, 0.0f, 0.05f, 0.5f, 0.99f, 0.97f, 1.0f, 0.05f, 0.05f, generator);

		float actionf;

		prevState[2] = action[2];
		prevState[3] = action[3];

		actionf = (prevState[2]) * 1.0f;

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::Left))
			actionf = -1.0f;
		else if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right))
			actionf = 1.0f;

		velocity += (-velocity * 0.01f + actionf * 0.001f + std::cos(3.0f * position) * -0.0025f) * dt / 0.017f;
		position += velocity * dt / 0.017f;

		// ---------------------------- Rendering ----------------------------

		window.clear(sf::Color::White);

		// Draw hills
		window.draw(&hills[0], hills.size(), sf::Lines);

		cartSprite.setPosition(sf::Vector2f(static_cast<float>(window.getSize().x) * 0.5f + pixelsPerMeter * position, static_cast<float>(window.getSize().y) * 0.5f + pixelsPerMeter * -0.333f * std::sinf(3.0f * position)));

		float slope = std::cos(3.0f * position);

		float angle = std::atan(slope);

		cartSprite.setRotation(360.0f - 180.0f / static_cast<float>(std::_Pi) * angle);

		window.draw(cartSprite);

		for (int x = 0; x < 16; x++)
		for (int y = 0; y < 16; y++) {
			sf::Color c;

			c.r = htm.getRegion().getColumn(x, y)._prediction * 255.0f;
			c.g = 0;
			c.b = 0;

			image.setPixel(x, y, c);
		}

		sf::Texture t;

		t.loadFromImage(image);

		sf::Sprite s;

		s.setPosition(800.0f - 256.0f, 0.0f);

		s.setScale(256.0f / image.getSize().x, 256.0f / image.getSize().y);

		s.setTexture(t);

		window.draw(s);

		// -------------------------------------------------------------------

		window.display();

		//dt = clock.getElapsedTime().asSeconds();
	} while (!quit);
}*/

/*const size_t entrySize = 30; // 31 if you include the weight

struct DataEntry {
	std::array<double, entrySize> _data;

	unsigned int _eventID;

	double _weight;

	bool _s;
};

int main() {
	std::mt19937 generator(time(nullptr));

	std::ifstream fromFile("Resources/training/training.csv");

	if (!fromFile.is_open()) {
		std::cerr << "Could not find training data!" << std::endl;
	}

	std::array<bool, entrySize> hasNulls;

	hasNulls.assign(false);

	std::array<double, entrySize> mins;

	mins.assign(999999.0);

	std::array<double, entrySize> maxs;

	maxs.assign(-999999.0);

	std::string line;

	std::getline(fromFile, line);

	std::vector<DataEntry> entries;

	int t = 0;

	while (!fromFile.eof() && fromFile.good()) {
		std::getline(fromFile, line);

		if (line == "")
			continue;
		
		DataEntry entry;

		std::istringstream is(line);

		std::string value;

		std::getline(is, value, ',');

		entry._eventID = std::stoi(value) - 100000;

		for (size_t i = 0; i < entrySize; i++) {
			std::getline(is, value, ',');

			entry._data[i] = std::stof(value);

			if (entry._data[i] == -999.0) {
				hasNulls[i] = true;
			}
			else {
				mins[i] = std::min(mins[i], entry._data[i]);
				maxs[i] = std::max(maxs[i], entry._data[i]);
			}
		}

		std::getline(is, value, ',');

		entry._weight = std::stof(value);

		std::getline(is, value, ',');

		entry._s = value == "s";

		entries.push_back(entry);

		t++;
	}

	int numNulls = 0;

	for (size_t i = 0; i < entrySize; i++)
	if (hasNulls[i])
		numNulls++;

	// Normalize the data
	for (size_t i = 0; i < entries.size(); i++) {
		for (size_t j = 0; j < entrySize; j++) {
			if (entries[i]._data[j] != -999.0)
				entries[i]._data[j] = (entries[i]._data[j] - mins[j]) / (maxs[j] - mins[j]);
		}
	}

	deep::DBN dbn;

	//nn::FeedForwardNeuralNetwork ffnn;

	//ffnn.createRandom(entrySize + numNulls, 1, 1, 60, -0.1f, 0.1f, generator);

	std::vector<size_t> rbmHiddenSizes(2);

	rbmHiddenSizes[0] = 60;
	rbmHiddenSizes[1] = 60;

	dbn.createRandom(entrySize + numNulls, 1, rbmHiddenSizes, -0.1, 0.1, generator);

	double rbmAlpha = 0.1;
	double ffnnAlphaTune = 0.05;
	double alphaMultiplier = 1.0;
	double momentum = 0.0;
	double weightDecayMultiplier = 0.999;
	size_t miniBatchSize = 1;

	// Pre-train layers
	for (size_t l = 0; l < rbmHiddenSizes.size(); l++) {
		std::cout << "Pre-training layer " << l + 1 << std::endl;

		for (size_t i = 0; i < 1000000; i++) {
			// Select random
			std::uniform_int_distribution<int> distEntry(0, entries.size() - 1);

			int entryIndex = distEntry(generator);

			std::vector<double> data(entrySize + numNulls);

			size_t nIndex = 0;

			for (size_t j = 0; j < entrySize; j++) {
				if (hasNulls[j]) {
					if (entries[entryIndex]._data[j] == -999.0) {
						data[entrySize + nIndex] = 1.0;
						data[j] = 0.0f;
					}
					else {
						data[entrySize + nIndex] = 0.0;
						data[j] = entries[entryIndex]._data[j];
					}

					nIndex++;
				}
				else {
					data[j] = entries[entryIndex]._data[j];
				}

				assert(data[j] >= 0.0 && data[j] <= 1.0);
			}

			// Execute layers up till here
			std::vector<double> input;

			dbn.getOutputMeanThroughLayers(l, data, input);

			dbn.trainLayerUnsupervised(l, input, rbmAlpha, generator);

			if (i % 1000 == 0)
				std::cout << "Pre-training iteration " << i + 1 << std::endl;
		}
	}

	dbn.prepareForGradientDescent();

	// Fine tuning
	for (size_t i = 0; i < 1000000; i++) {
		// Select random
		std::uniform_int_distribution<int> distEntry(0, entries.size() - 1);

		int entryIndex = distEntry(generator);

		std::vector<double> data(entrySize + numNulls);

		size_t nIndex = 0;

		for (size_t j = 0; j < entrySize; j++) {
			if (hasNulls[j]) {
				if (entries[entryIndex]._data[j] == -999.0) {
					data[entrySize + nIndex] = 1.0;
					data[j] = 0.0f;
				}
				else {
					data[entrySize + nIndex] = 0.0;
					data[j] = entries[entryIndex]._data[j];
				}

				nIndex++;
			}
			else {
				data[j] = entries[entryIndex]._data[j];
			}

			assert(data[j] >= 0.0 && data[j] <= 1.0);
		}

		std::vector<double> output;

		dbn.execute(data, output);

		std::vector<double> target(1);

		target[0] = entries[entryIndex]._s ? 1.0 : -1.0;

		dbn.getError(target);

		//dbn.signError();

		dbn.accumulateGradient();

		if ((i % miniBatchSize) == (miniBatchSize - 1)) {
			dbn.moveAlongAccumulatedGradient(ffnnAlphaTune);

			dbn.decayWeights(weightDecayMultiplier);
		}

		if (i % 1000 == 0)
			std::cout << "Fine-tuning iteration " << i + 1 << std::endl;
	}

	int numAccurate = 0;

	// Train
	int numTestTrials = entries.size();

	float sRate = 0.0;
	float bRate = 0.0;

	for (size_t i = 0; i < numTestTrials; i++) {
		// Select random
		//std::uniform_int_distribution<int> distEntry(0, entries.size() - 1);

		size_t entryIndex = i;

		std::vector<double> data(entrySize + numNulls);

		size_t nIndex = 0;

		for (size_t j = 0; j < entrySize; j++) {
			if (hasNulls[j]) {
				if (entries[entryIndex]._data[j] == -999.0) {
					data[entrySize + nIndex] = 1.0;
					data[j] = 0.0f;
				}
				else {
					data[entrySize + nIndex] = 0.0;
					data[j] = entries[entryIndex]._data[j];
				}

				nIndex++;
			}
			else {
				data[j] = entries[entryIndex]._data[j];
			}

			//assert(data[j] >= 0.0 && data[j] <= 1.0);
		}

		std::vector<double> output;

		dbn.execute(data, output);

		int label = output[0] > 0.0 ? 1 : 0;

		if ((label == 1) == entries[entryIndex]._s) {
			std::cout << entryIndex << ": Accurate. Output: " << output[0] << " Result: " << (entries[entryIndex]._s ? "s" : "b") << std::endl;
		}
		else {
			std::cout << entryIndex << ": Innaccurate. Output: " << output[0] << " Result: " << (entries[entryIndex]._s ? "s" : "b") << std::endl;
		}

		if ((label == 1) == entries[entryIndex]._s) {
			numAccurate++;
		}

		if (label == 1) { // Predicted s
			if (entries[entryIndex]._s)
				sRate += entries[entryIndex]._weight;
			else
				bRate += entries[entryIndex]._weight;
		}

		if (i % 1000 == 0)
			std::cout << "Testing iteration " << i + 1 << std::endl;
	}

	int numInnaccurate = numTestTrials - numAccurate;

	std::cout << "Error: " << numInnaccurate / static_cast<float>(numTestTrials)* 100.0 << "%" << std::endl;

	std::cout << "AMS: " << std::sqrt(2.0 * ((sRate + bRate + 10.0) * std::log(1.0 + sRate / (bRate + 10.0)) - sRate)) << std::endl;

	system("pause");

	std::cout << "Loading test data..." << std::endl;

	std::ifstream fromFileTest("Resources/test/test.csv");

	if (!fromFileTest.is_open()) {
		std::cerr << "Could not find test data!" << std::endl;
	}


	std::getline(fromFileTest, line);

	entries.clear();

	t = 0;

	while (!fromFileTest.eof() && fromFileTest.good()) {
		std::getline(fromFileTest, line);

		if (line == "")
			continue;

		DataEntry entry;

		std::istringstream is(line);

		std::string value;

		std::getline(is, value, ',');

		entry._eventID = std::stoi(value);

		for (size_t i = 0; i < entrySize; i++) {
			std::getline(is, value, ',');

			entry._data[i] = std::stof(value);
		}

		entries.push_back(entry);

		t++;
	}

	fromFileTest.close();

	// Normalize the data
	for (size_t i = 0; i < entries.size(); i++) {
		for (size_t j = 0; j < entrySize; j++) {
			if (entries[i]._data[j] != -999.0)
				entries[i]._data[j] = (entries[i]._data[j] - mins[j]) / (maxs[j] - mins[j]);
		}
	}

	std::cout << "Number of entries: " << entries.size() << std::endl;;

	system("pause");

	// Output file
	std::cout << "Evaluating test data..." << std::endl;

	std::ofstream evaluationToFile("evaluation.txt");

	evaluationToFile << "EventId,RankOrder,Class" << std::endl;

	for (size_t i = 0; i < entries.size(); i++) {
		// Select random
		//std::uniform_int_distribution<int> distEntry(0, entries.size() - 1);

		size_t entryIndex = i;

		std::vector<double> data(entrySize + numNulls);

		size_t nIndex = 0;

		for (size_t j = 0; j < entrySize; j++) {
			if (hasNulls[j]) {
				if (entries[entryIndex]._data[j] == -999.0) {
					data[entrySize + nIndex] = 1.0;
					data[j] = 0.0f;
				}
				else {
					data[entrySize + nIndex] = 0.0;
					data[j] = entries[entryIndex]._data[j];
				}

				nIndex++;
			}
			else {
				data[j] = entries[entryIndex]._data[j];
			}

			//assert(data[j] >= 0.0 && data[j] <= 1.0);
		}

		std::vector<double> output;

		dbn.execute(data, output);

		int label = output[0] > 0.0 ? 1 : 0;

		//size_t rankOrder = static_cast<size_t>(std::min(1.0, std::max(0.0, output[0] * 0.5 + 0.5)) * 550000.0) + 1;

		evaluationToFile << entries[entryIndex]._eventID << "," << (i + 1) << "," << (label == 1 ? "s" : "b") << std::endl;

		if (i % 1000 == 0)
			std::cout << "Evaluated iteration " << i + 1 << std::endl;
	}

	evaluationToFile.close();

	std::cout << "Done." << std::endl;

	system("pause");

	return 0;
}*/

/*int main() {
	std::mt19937 generator(time(nullptr));

	sf::Image img;

	img.loadFromFile("testImage.png");

	deep::ConvNet2D convNet;

	std::vector<deep::ConvNet2D::LayerPairDesc> descs(2);

	convNet.createRandom(img.getSize().x, img.getSize().y, 1, descs, -0.1f, 0.1f, generator);

	float bInv = 1.0f / 255.0f;

	for (size_t x = 0; x < img.getSize().x; x++)
	for (size_t y = 0; y < img.getSize().y; y++) {
		float greyscale = ((img.getPixel(x, y).r * bInv + img.getPixel(x, y).g * bInv + img.getPixel(x, y).b * bInv) * 0.3333f - 0.5f) * 6.0f;

		convNet.setInput(x, y, 0, greyscale);
	}

	sf::RenderWindow window;
	window.create(sf::VideoMode(convNet.getOutputWidth() * 8, convNet.getOutputHeight() * 8), "ConvNetRBM", sf::Style::Default);

	for (size_t i = 0; i < 50; i++) {
		convNet.activateAndLearn(0.01f, generator);

		sf::Image result;

		result.create(convNet.getOutputWidth(), convNet.getOutputHeight());

		for (size_t x = 0; x < convNet.getOutputWidth(); x++)
		for (size_t y = 0; y < convNet.getOutputHeight(); y++) {
			sf::Color c;

			c.r = 255.0f * std::min(1.0f, std::max(0.0f, ((convNet.getOutput(x, y, 0) - 0.5f) * 2.0f + 0.5f)));;
			c.g = 255.0f * std::min(1.0f, std::max(0.0f, ((convNet.getOutput(x, y, 1) - 0.5f) * 2.0f + 0.5f)));;
			c.b = 255.0f * std::min(1.0f, std::max(0.0f, ((convNet.getOutput(x, y, 2) - 0.5f) * 2.0f + 0.5f)));;

			result.setPixel(x, y, c);
		}

		sf::Texture t;

		t.loadFromImage(result);

		sf::Sprite s;

		s.setScale(8.0f, 8.0f);

		s.setTexture(t);

		window.draw(s);

		window.display();
	}

	return 0;
}*/

/*float boostFunction(float active, float minimum) {
	return (1.0f - minimum) + std::max(0.0f, -(minimum - active));
}

int main() {
	std::mt19937 generator(time(nullptr));

	std::function<float(float, float)> boostFunc = std::bind(boostFunction, std::placeholders::_1, std::placeholders::_2);

	htm::Region region;

	region.createRandom(40, 40, 8, 6, 0, 40, 40, 5, 0.02f, 2.0f, -0.02f, 0.301f, 0.1f, generator);

	sf::RenderWindow window;

	window.create(sf::VideoMode(1200, 600), "HTM", sf::Style::Default);

	window.setFramerateLimit(40);

	float squareSize = window.getSize().y / 40.0f;

	// ---------------------------- Game Loop -----------------------------

	bool quit = false;

	sf::Clock clock;

	float dt = 0.1f;

	sf::Font font;

	font.loadFromFile("Resources/pixelated.ttf");

	sf::Image image;

	image.create(40, 40);

	sf::Image reconstruction;

	reconstruction.create(40, 40);

	std::vector<bool> input(40 * 40, false);

	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

	float timer = 0.0f;
	float time = PI * 2.0f;

	do {
		clock.restart();

		// ----------------------------- Input -----------------------------

		sf::Event windowEvent;

		while (window.pollEvent(windowEvent))
		{
			switch (windowEvent.type)
			{
			case sf::Event::Closed:
				quit = true;
				break;
			}
		}

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
			quit = true;

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::F))
			window.setFramerateLimit(600);
		else
			window.setFramerateLimit(40);

		// Clear input
		for (int i = 0; i < input.size(); i++)
			input[i] = false;

		// Set one dot
		int dotX = std::round(20.0f + std::cos(timer) * 16.0f);
		int dotY = std::round(20.0f + std::sin(timer * 2.0f) * 8.0f);

		for (int dx = -2; dx <= 2; dx++)
		for (int dy = -2; dy <= 2; dy++) {
			int x = dotX + dx;
			int y = dotY + dy;

			if (x >= 0 && y >= 0 && x < 40 && y < 40)
				input[x + 40 * y] = true;
		}

		region.stepBegin();
		
		region.spatialPooling(input, 0.3f, 3.0f, 18, 0.02f, 0.015f, 0.01f, 0.05f, 0.05f, 0.015f, boostFunc);

		region.temporalPoolingLearn(0.3f, 4, 1, 10, 32, 0.02f, 0.015f, 0.301f, 6, generator);

		std::vector<bool> reconstructionData;

		int steps = 2;

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::Num3))
			steps = 3;

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::Num4))
			steps = 4;

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::Num5))
			steps = 5;

		region.getReconstructionAtTime(reconstructionData, 3.0f, 0.3f, steps);

		
		// ---------------------------- Rendering ----------------------------

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::T)) {
			for (int x = 0; x < 40; x++)
			for (int y = 0; y < 40; y++) {
				sf::Color c;

				if (input[x + 40 * y])
					c = sf::Color::Blue;
				else
					c = sf::Color::White;

				image.setPixel(x, y, c);
			}
		}
		else {
			for (int x = 0; x < 40; x++)
			for (int y = 0; y < 40; y++) {
				sf::Color c;

				if (region.getColumn(x, y).isActive())
					c = sf::Color::Red;
				else
					c = sf::Color::White;

				image.setPixel(x, y, c);
			}
		}

		for (int x = 0; x < 40; x++)
		for (int y = 0; y < 40; y++) {
			sf::Color c;

			if (reconstructionData[x + y * 40])
				c = sf::Color::Red;
			else
				c = sf::Color::White;

			reconstruction.setPixel(x, y, c);
		}

		sf::Texture t;

		t.loadFromImage(image);

		sf::Sprite s;

		s.setTexture(t);

		s.setScale(squareSize, squareSize);

		window.draw(s);

		sf::Texture t2;

		t2.loadFromImage(reconstruction);

		sf::Sprite s2;

		s2.setPosition(600, 0);

		s2.setTexture(t2);

		s2.setScale(squareSize, squareSize);

		window.draw(s2);

		// -------------------------------------------------------------------

		window.display();

		//dt = clock.getElapsedTime().asSeconds();

		timer += dt * 1.0f;

		if (timer >= time)
			timer -= time;

	} while (!quit);

	return 0;
}*/

/*int main() {
	std::mt19937 generator(time(nullptr));

	lstm::LSTMG lstmg;

	lstmg.createRandomLayered(1, 1, 1, 2, 1, 4, -0.1f, 0.1f, generator);

	float prevY = 0.0f;

	for (int i = 0; i < 2000; i++)
	for (float x = 0.0f; x < 6.28f; x += 0.2f) {
		float y = std::sin(x);

		lstmg.setInput(0, 0.0f);

		lstmg.step(true);

		lstmg.getDeltas(std::vector<float>(1, y), 0.0f, true);

		lstmg.moveAlongDeltas(0.01f);

		prevY = y;
	}

	prevY = 0.0f;

	for (float x = 0.0f; x < 6.28f; x += 0.2f) {
		float y = std::sin(x);

		lstmg.setInput(0, 0.0f);

		lstmg.step(true);

		std::cout << lstmg.getOutput(0) << std::endl;

		prevY = y;
	}

	return 0;
}*/

/*float testFunc(float x) {
	return std::sin(x * 3.0f);
}

void addPlotData(std::ostream &os, int plotIndex, const std::vector<std::tuple<float, float>> &data) {
	os << "local c" << plotIndex << " = { " << std::endl;

	for (int i = 0; i < data.size(); i++) {
		os << "    { " << std::get<0>(data[i]) << ", " << std::get<1>(data[i]) << " }";

		if (i != data.size() - 1)
			os << ",";

		os << std::endl;
	}

	os << "}" << std::endl << std::endl;
}

int main() {
	int iter = 64;

	std::mt19937 generator(time(nullptr));

	std::ofstream toFile("output.txt");

	std::vector<std::tuple<float, float>> data;

	data.clear();

	for (float x = 0.0f; x < 6.28f; x += 0.1f) {
		float y = testFunc(x);

		data.push_back(std::make_tuple(x, y));
	}

	addPlotData(toFile, 1, data);

	data.clear();

	//std::cout << "SDR RBF:" << std::endl;

	chtm::CHTMRegion htm;

	htm.createRandom(2, 2, 16, 16, 3, 2, 3, 1, -0.1f, 0.1f, 0.01f, 0.1f, -0.1f, 0.1f, -0.1f, 0.1f, -0.1f, 0.1f, generator);

	for (int i = 0; i < iter; i++) {
		float prevY = 0.0f;

		for (float x = 0.0f; x < 6.28f; x += 0.1f) {
			float y = testFunc(x);

			std::vector<float> input(4, x);
			std::vector<float> output(1);
			std::vector<float> target(1, y);

			htm.stepBegin();

			htm.getOutput(input, output, 3, 32.0f, 8.0f, 8.0f, generator);

			htm.learn(input, output, target, 0.01f, 0.05f, 0.05f, 2.0f, 0.01f, 0.2f, 0.01f);

			prevY = y;
		}

		std::cout << i << std::endl;
	}

	float error = 0.0f;

	sf::Image img;

	img.create(16, 16);

	float prevY = 0.0f;

	for (float x = 0.0f; x < 6.28f; x += 0.1f) {
		float y = testFunc(x);

		std::vector<float> input(4, x);
		std::vector<float> output(1);
		std::vector<float> target(1, y);

		htm.stepBegin();

		htm.getOutput(input, output, 3, 32.0f, 8.0f, 8.0f, generator);

		if (x == 0.0f) {
			for (int x = 0; x < 16; x++)
			for (int y = 0; y < 16; y++) {
				sf::Color c = sf::Color::Black;
				c.r = 255.0f * htm.getColumn(x, y)._output;

				img.setPixel(x, y, c);
			}
		}

		error += std::abs(output[0] - y);

		//sdrrbf.update(input, output, target, 0.05f, 0.05f, 0.05f, 1.0f);

		std::cout << x << " " << output[0] << std::endl;

		data.push_back(std::make_tuple(x, output[0]));

		prevY = y;

		system("pause");
	}

	addPlotData(toFile, 2, data);

	data.clear();

	img.saveToFile("region.png");

	//std::cout << "Error: " << error << std::endl;

	//system("pause");

	//std::cout << "MLP:" << std::endl;
	deep::FA fans;

	fans.createRandom(1, 1, 1, 256, 0.1f, generator);

	std::uniform_real_distribution<float> angleDist(0.0f, 6.28f);

	for (int i = 0; i < iter; i++)
	for (float x = 0.0f; x < 6.28f; x += 0.01f) {

		float y = testFunc(x);

		std::vector<float> input(1, x);
		std::vector<float> output(1);
		std::vector<float> target(1, y);

		fans.clearGradient();

		fans.process(input, output);

		fans.accumulateGradient(input, target);

		fans.moveAlongGradientRMS(0.05f, 0.01f, 0.1f, 0.0f, 0.0f);
	}

	error = 0.0f;

	for (float x = 0.0f; x < 6.28f; x += 0.1f) {
		float y = testFunc(x);

		std::vector<float> input(1, x);
		std::vector<float> output(1);
		std::vector<float> target(1, y);

		fans.process(input, output);

		error += std::abs(output[0] - y);

		data.push_back(std::make_tuple(x, output[0]));

		std::cout << output[0] << std::endl;
	}

	addPlotData(toFile, 3, data);

	data.clear();

	deep::SharpFA fa;

	fa.createRandom(1, 1, 1, 256, 0.1f, generator);

	for (int i = 0; i < iter; i++)
	for (float x = 0.0f; x < 6.28f; x += 0.01f) {

		float y = testFunc(x);

		std::vector<float> input(1, x);
		std::vector<float> output(1);
		std::vector<float> target(1, y);

		fa.clearGradient();

		fa.process(input, output, 0.5f, 3);

		fa.accumulateGradient(input, target);

		fa.moveAlongGradientRMS(0.05f, 0.01f, 0.1f, 0.0f, 0.0f);
	}

	error = 0.0f;

	for (float x = 0.0f; x < 6.28f; x += 0.1f) {
		float y = testFunc(x);

		std::vector<float> input(1, x);
		std::vector<float> output(1);
		std::vector<float> target(1, y);

		fa.process(input, output, 0.5f, 3);

		error += std::abs(output[0] - y);

		data.push_back(std::make_tuple(x, output[0]));

		std::cout << output[0] << std::endl;
	}

	addPlotData(toFile, 4, data);

	data.clear();

	deep::DBN dbn;

	dbn.createRandom(1, 1, std::vector<int>(1, 256), -0.1f, 0.1f, generator);

	for (int i = 0; i < 10000; i++) {
		float x2 = angleDist(generator);

		float y = testFunc(x2);

		std::vector<float> input(1, x2);
		std::vector<float> output(1);
		std::vector<float> mean(16);
		std::vector<float> target(1, y);

		dbn.trainLayerUnsupervised(0, input, 0.01f, generator);
	}

	error = 0.0f;

	dbn.prepareForGradientDescent();

	for (int i = 0; i < iter; i++)
	for (float x = 0.0f; x < 6.28f; x += 0.01f) {
		float y = testFunc(x);

		std::vector<float> input(1, x);
		std::vector<float> output(1);
		std::vector<float> target(1, y);

		dbn.execute(input, output);
	
		dbn.getError(target);

		dbn.accumulateGradient();

		dbn.moveAlongAccumulatedGradient(0.001f);
	}
	
	for (float x = 0.0f; x < 6.28f; x += 0.1f) {
		float y = testFunc(x);

		std::vector<float> input(1, x);
		std::vector<float> output(1);
		std::vector<float> target(1, y);

		dbn.execute(input, output);

		error += std::abs(output[0] - y);

		data.push_back(std::make_tuple(x, output[0]));

		std::cout << output[0] << std::endl;
	}

	addPlotData(toFile, 5, data);

	data.clear();

	deep::FA fanss;

	fanss.createRandom(1, 1, 1, 256, 0.1f, generator);

	for (int i = 0; i < iter; i++)
	for (float x = 0.0f; x < 6.28f; x += 0.01f) {

		float x2 = angleDist(generator);

		float y = testFunc(x2);

		std::vector<float> input(1, x2);
		std::vector<float> output(1);
		std::vector<float> target(1, y);

		fanss.clearGradient();

		fanss.process(input, output);

		fanss.accumulateGradient(input, target);

		fanss.moveAlongGradientRMS(0.05f, 0.01f, 0.1f, 0.0f, 0.0f);
	}

	error = 0.0f;

	for (float x = 0.0f; x < 6.28f; x += 0.1f) {
		float y = testFunc(x);

		std::vector<float> input(1, x);
		std::vector<float> output(1);
		std::vector<float> target(1, y);

		fanss.process(input, output);

		error += std::abs(output[0] - y);

		data.push_back(std::make_tuple(x, output[0]));

		std::cout << output[0] << std::endl;
	}

	addPlotData(toFile, 6, data);

	data.clear();

	toFile.close();

	//std::cout << "Error: " << error << std::endl;

	//system("pause");

	//c1.prepare(sf::Vector2f(0.0f, 6.28f), sf::Vector2f(-2.0f, 2.0f));

	return 0;
}*/

/*int main() {
	std::mt19937 generator(time(nullptr));

	sf::RenderWindow renderWindow;

	renderWindow.create(sf::VideoMode(512, 512), "Circles", sf::Style::Default);

	sf::Clock clock;

	bool quit = false;

	bool aPressedLastFrame = false;

	do {
		clock.restart();

		// ----------------------------- Input -----------------------------

		sf::Event windowEvent;

		while (renderWindow.pollEvent(windowEvent)) {
			switch (windowEvent.type)
			{
			case sf::Event::Closed:
				quit = true;
				break;
			}
		}

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
			quit = true;

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::A) && !aPressedLastFrame) {

		}

		aPressedLastFrame = sf::Keyboard::isKeyPressed(sf::Keyboard::A);

	} while (!quit);

	return 0;
}*/

int main() {
	std::mt19937 generator(time(nullptr));

	float reward = 0.0f;
	float prevReward = 0.0f;

	float initReward = 0.0f;

	float totalReward = 0.0f;

	sf::RenderWindow window;

	window.create(sf::VideoMode(1600, 600), "Pole Balancing");

	window.setVerticalSyncEnabled(true);

	//window.setFramerateLimit(60);

	// -------------------------- Load Resources --------------------------

	sf::Texture backgroundTexture;
	sf::Texture cartTexture;
	sf::Texture poleTexture;

	backgroundTexture.loadFromFile("Resources/background.png");
	cartTexture.loadFromFile("Resources/cart.png");
	poleTexture.loadFromFile("Resources/pole.png");

	sf::Texture inputCartTexture;
	sf::Texture inputPoleTexture;

	inputCartTexture.loadFromFile("Resources/inputCart.png");
	inputPoleTexture.loadFromFile("Resources/inputPole.png");

	// --------------------------------------------------------------------

	sf::Sprite backgroundSprite;
	sf::Sprite cartSprite;
	sf::Sprite poleSprite;

	backgroundSprite.setTexture(backgroundTexture);
	cartSprite.setTexture(cartTexture);
	poleSprite.setTexture(poleTexture);

	backgroundSprite.setPosition(sf::Vector2f(0.0f, 0.0f));

	cartSprite.setOrigin(sf::Vector2f(static_cast<float>(cartSprite.getTexture()->getSize().x) * 0.5f, static_cast<float>(cartSprite.getTexture()->getSize().y)));
	poleSprite.setOrigin(sf::Vector2f(static_cast<float>(poleSprite.getTexture()->getSize().x) * 0.5f, static_cast<float>(poleSprite.getTexture()->getSize().y)));

	sf::Sprite inputCartSprite;
	sf::Sprite inputPoleSprite;

	inputCartSprite.setTexture(inputCartTexture);
	inputPoleSprite.setTexture(inputPoleTexture);

	inputCartSprite.setOrigin(sf::Vector2f(static_cast<float>(inputCartSprite.getTexture()->getSize().x) * 0.5f, static_cast<float>(inputCartSprite.getTexture()->getSize().y)));
	inputPoleSprite.setOrigin(sf::Vector2f(static_cast<float>(inputPoleSprite.getTexture()->getSize().x) * 0.5f, static_cast<float>(inputPoleSprite.getTexture()->getSize().y)));


	// ----------------------------- Physics ------------------------------

	float pixelsPerMeter = 128.0f;
	float inputPixelsPerMeter = 8.0f;
	float poleLength = 1.0f;
	float g = -2.8f;
	float massMass = 40.0f;
	float cartMass = 2.0f;
	sf::Vector2f massPos(0.0f, poleLength);
	sf::Vector2f massVel(0.0f, 0.0f);
	float poleAngle = static_cast<float>(PI)* 0.0f;
	float poleAngleVel = 0.0f;
	float poleAngleAccel = 0.0f;
	float cartX = 0.0f;
	float cartVelX = 0.0f;
	float cartAccelX = 0.0f;
	float poleRotationalFriction = 0.008f;
	float cartMoveRadius = 1.8f;
	float cartFriction = 0.02f;
	float maxSpeed = 3.0f;

	// ---------------------------- Game Loop -----------------------------

	bool quit = false;

	sf::Clock clock;

	float dt = 0.017f;

	float fitness = 0.0f;
	float prevFitness = 0.0f;

	float lowPassFitness = 0.0f;

	bool reverseDirection = false;

	bool trainMode = true;

	bool tDownLastFrame = false;

	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

	sf::Font font;

	font.loadFromFile("Resources/pixelated.ttf");

	sf::RenderTexture inputRT;

	inputRT.create(64, 32);

	sf::RenderTexture rt;

	rt.create(800, 600);

	sf::plot::Plot plot;

	plot.setSize(sf::Vector2f(rt.getSize().x, rt.getSize().y));
	plot.setTitle("Avg. Reward");
	plot.setFont("Resources/arial.ttf");
	plot.setXLabel("Time (seconds)");
	plot.setYLabel("Avg. Reward");
	plot.setBackgroundColor(sf::Color::White);
	plot.setTitleColor(sf::Color::Black);
	plot.setPosition(sf::Vector2f(0.0f, 0.0f));

	sf::plot::Curve &cAverage = plot.createCurve("Avg. Reward", sf::Color::Red);

	cAverage.setFill(false);

	plot.prepare();

	float plotMin = 99999.0f;
	float plotMax = -99999.0f;

	rt.draw(plot);
	rt.display();

	float avgReward = 0.0f;
	float avgRewardDecay = 0.003f;

	float totalTime = 0.0f;

	float plotUpdateTimer = 0.0f;

	chtm::CHTMRL agent;

	agent.createRandom(2, 3, 32, 32, 3, 2, 3, -0.75f, 0.75f, 0.02f, 0.1f, -0.1f, 0.1f, -0.1f, 0.1f, -0.1f, 0.1f, -0.1f, 0.1f, generator);

	std::vector<float> prevInput(6, 0.0f);

	do {
		clock.restart();

		// ----------------------------- Input -----------------------------

		sf::Event windowEvent;

		while (window.pollEvent(windowEvent))
		{
			switch (windowEvent.type)
			{
			case sf::Event::Closed:
				quit = true;
				break;
			}
		}

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
			quit = true;

		// Update fitness
		if (poleAngle < static_cast<float>(PI))
			fitness = -(static_cast<float>(PI)* 0.5f - poleAngle);
		else
			fitness = -(static_cast<float>(PI)* 0.5f - (static_cast<float>(PI)* 2.0f - poleAngle));

		fitness += static_cast<float>(PI)* 0.5f;

		//fitness = fitness - std::abs(poleAngleVel * 1.0f);

		//fitness = -std::abs(cartX);

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::A))
			fitness = -cartX;
		else if (sf::Keyboard::isKeyPressed(sf::Keyboard::D))
			fitness = cartX;

		// ------------------------------ AI -------------------------------

		float dFitness = fitness - prevFitness;

		//reward = dFitness * 5.0f;

		reward = fitness * 0.2f;

		if (totalTime == 0.0f)
			avgReward = reward;
		else
			avgReward = (1.0f - avgRewardDecay) * avgReward + avgRewardDecay * reward;

		sf::Image img = inputRT.getTexture().copyToImage();

		std::vector<float> state(6);

		//lstmAC.setInput(0, cartX * 0.25f);
		//lstmAC.setInput(1, cartVelX);
		//lstmAC.setInput(2, std::fmod(poleAngle + static_cast<float>(PI), 2.0f * static_cast<float>(PI)));
		//lstmAC.setInput(3, poleAngleVel);

		state[0] = cartX * 0.25f;
		state[1] = cartVelX * 0.4f;
		state[2] = std::fmod(poleAngle + static_cast<float>(PI), 2.0f * static_cast<float>(PI)) / (2.0f * static_cast<float>(PI)) * 2.0f - 1.0f;
		state[3] = poleAngleVel * 0.2f;
		state[4] = prevInput[4];
		state[5] = prevInput[5];

		std::vector<float> action(6);
		std::vector<bool> actionMask(6, false);
		actionMask[4] = true;
		actionMask[5] = true;

		agent.step(reward, state, actionMask, action, 0.2f, 1, 0.05f, 0.8f, 0.0f, 0.1f, 32.0f, 1.0f, 4, 2.0f, 0.5f, 2.0f, 2.0f, 0.0001f, 0.01f, 0.01f, 0.01f, 0.005f, 0.0001f, 0.01f, 0.05f, 0.5f, 0.99f, 0.97f, 1.0f, 0.05f, 0.05f, generator);

		std::normal_distribution<float> pertDist(0.0f, 0.05f);
		std::uniform_real_distribution<float> uniformDist(0.0f, 1.0f);

		if (uniformDist(generator) < 0.05f) {
			prevInput[4] = uniformDist(generator) * 2.0f - 1.0f;
			prevInput[5] = uniformDist(generator) * 2.0f - 1.0f;
		}
		else {
			prevInput[4] = std::min(1.0f, std::max(-1.0f, std::min(1.0f, std::max(-1.0f, action[4])) + pertDist(generator)));
			prevInput[5] = std::min(1.0f, std::max(-1.0f, std::min(1.0f, std::max(-1.0f, action[5])) + pertDist(generator)));
		}

		float dir = prevInput[4];

		//dir = 1.4f * (dir * 2.0f - 1.0f);

		float agentForce = 4000.0f * dir;
		//}
		//float agentForce = 4000.0f * agent.getOutput(0);

		prevFitness = fitness;

		// ---------------------------- Physics ----------------------------

		float pendulumCartAccelX = cartAccelX;

		if (cartX < -cartMoveRadius)
			pendulumCartAccelX = 0.0f;
		else if (cartX > cartMoveRadius)
			pendulumCartAccelX = 0.0f;

		poleAngleAccel = pendulumCartAccelX * std::cos(poleAngle) + g * std::sin(poleAngle);
		poleAngleVel += -poleRotationalFriction * poleAngleVel + poleAngleAccel * dt;
		poleAngle += poleAngleVel * dt;

		massPos = sf::Vector2f(cartX + std::cos(poleAngle + static_cast<float>(PI)* 0.5f) * poleLength, std::sin(poleAngle + static_cast<float>(PI)* 0.5f) * poleLength);

		float force = 0.0f;

		if (std::abs(cartVelX) < maxSpeed) {
			force = std::max(-4000.0f, std::min(4000.0f, agentForce));

			if (sf::Keyboard::isKeyPressed(sf::Keyboard::Left))
				force = -4000.0f;

			if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right))
				force = 4000.0f;
		}

		//if (trainMode) {
		//	en.calculateGradient(std::vector<float>(1, force / 4000.0f));

		//	en.moveAlongGradient(0.01f);
		//}

		//en.updateContext();

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
			poleAngle += static_cast<float>(PI)* 2.0f;

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::T)) {
			if (!tDownLastFrame) {
				trainMode = !trainMode;
			}

			tDownLastFrame = true;
		}
		else
			tDownLastFrame = false;

		// ---------------------------- Rendering ----------------------------

		// Render to input buffer
		inputRT.clear();

		inputCartSprite.setPosition(sf::Vector2f(inputRT.getSize().x * 0.5f + inputPixelsPerMeter * cartX, inputRT.getSize().y * 0.5f + 4.0f));

		inputRT.draw(inputCartSprite);

		inputPoleSprite.setPosition(inputCartSprite.getPosition() + sf::Vector2f(0.0f, -4.0f));
		inputPoleSprite.setRotation(poleAngle * 180.0f / static_cast<float>(PI)+180.0f);

		inputRT.draw(inputPoleSprite);

		inputRT.display();

		window.clear();

		window.draw(backgroundSprite);

		cartSprite.setPosition(sf::Vector2f(800.0f * 0.5f + pixelsPerMeter * cartX, 600.0f * 0.5f + 3.0f));

		window.draw(cartSprite);

		poleSprite.setPosition(cartSprite.getPosition() + sf::Vector2f(0.0f, -45.0f));
		poleSprite.setRotation(poleAngle * 180.0f / static_cast<float>(PI)+180.0f);

		window.draw(poleSprite);

		if (plotUpdateTimer >= 1.0f) {
			plotUpdateTimer -= 1.0f;

			plotMin = std::min<float>(avgReward, plotMin);
			plotMax = std::max<float>(avgReward, plotMax);

			cAverage.addValue(avgReward);

			plot.prepare();

			cAverage.prepare(sf::Vector2f(0.0f, totalTime), sf::Vector2f(plotMin, plotMax));

			rt.draw(plot);

			rt.display();
		}

		sf::Sprite plotSprite;

		plotSprite.setPosition(800.0f, 0.0f);

		plotSprite.setTexture(rt.getTexture());

		window.draw(plotSprite);

		if (trainMode) {
			sf::Text text;

			text.setFont(font);

			text.setString("Train");

			text.setPosition(10.0f, 10.0f);

			text.setColor(sf::Color::Blue);

			window.draw(text);
		}

		sf::Sprite inputSprite;

		inputSprite.setTexture(inputRT.getTexture());

		inputSprite.setPosition(0, 0);
		inputSprite.setScale(4.0f, 4.0f);

		window.draw(inputSprite);

		sf::Image regionImage;

		regionImage.create(32, 32);

		for (int x = 0; x < 32; x++)
		for (int y = 0; y < 32; y++) {
			float s = agent.getRegion().getColumn(x, y)._prediction;

			sf::Color c = sf::Color::Black;

			c.r = 255.0f * s;

			regionImage.setPixel(x, y, c);
		}

		sf::Texture regionTex;
		regionTex.loadFromImage(regionImage);

		float pw = 4.0f;

		sf::Sprite regionSprite;
		regionSprite.setPosition(800.0f - 32 * pw, 0.0f);
		regionSprite.setScale(pw, pw);

		regionSprite.setTexture(regionTex);

		window.draw(regionSprite);

		// -------------------------------------------------------------------

		window.display();

		//dt = clock.getElapsedTime().asSeconds();

		totalTime += dt;
		plotUpdateTimer += dt;
	} while (!quit);

	return 0;
}

	/*deep::FERL ferl;

	ferl.createRandom(2, 2, 12, 0.1f, generator);

	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

	int tunnelLength = 4;

	float reward = 0.0f;

	for (int e = 0; e < 10000; e++) {
		int x = 1;

		bool sideIsUp = dist01(generator) < 0.5f;

		for (int t = 0; t < 8; t++) {
			std::vector<float> input(2);

			if (t == 0)
				input[0] = sideIsUp ? 1.0f : -1.0f;
			else
				input[0] = 0.0f;

			if (x == tunnelLength)
				input[1] = 1.0f;
			else
				input[1] = 0.0f;

			std::vector<float> action(2);

			ferl.step(input, action, reward, 0.5f, 0.9f, 0.9f, 1.0f, 32, 4, 0.05f, 0.15f, 0.05f, 800, 400, 0.05f, 0.0f, generator);

			reward *= 0.5f;

			if (x == tunnelLength) {
				if (action[0] < -0.333f)
					x--;

				if (action[1] > 0.333f) {
					if (sideIsUp)
						reward = 1.0f;
					else
						reward = -0.5f;

					break;
				}
				else if (action[1] < -0.333f) {
					if (!sideIsUp)
						reward = 1.0f;
					else
						reward = -0.5f;

					break;
				}
			}
			else if (x == 1) {
				if (action[0] > 0.333f)
					x++;
			}
			else {
				if (action[0] > 0.333f)
					x++;
				else if (action[0] < -0.333f)
					x--;
			}
		}
	}

	// Tests
	int successes = 0;
	int failures = 0;

	for (int e = 0; e < 1000; e++) {
		int x = 1;

		bool sideIsUp = dist01(generator) < 0.5f;

		for (int t = 0; t < 8; t++) {
			std::vector<float> input(2);

			if (t == 0)
				input[0] = sideIsUp ? 1.0f : -1.0f;
			else
				input[0] = 0.0f;

			if (x == tunnelLength)
				input[1] = 1.0f;
			else
				input[1] = 0.0f;

			std::vector<float> action(2);

			ferl.step(input, action, reward, 0.5f, 0.9f, 0.9f, 1.0f, 32, 4, 0.05f, 0.15f, 0.05f, 800, 400, 0.05f, 0.0f, generator);

			reward *= 0.5f;

			if (x == tunnelLength) {
				if (action[0] < -0.333f)
					x--;

				if (action[1] > 0.333f) {
					if (sideIsUp) {
						reward = 1.0f;
						successes++;
					}
					else {
						reward = -0.5f;
						failures++;
					}

					break;
				}
				else if (action[1] < -0.333f) {
					if (!sideIsUp) {
						reward = 1.0f;
						successes++;
					}
					else {
						reward = -0.5f;
						failures++;
					}

					break;
				}
			}
			else if (x == 1) {
				if (action[0] > 0.333f)
					x++;
			}
			else {
				if (action[0] > 0.333f)
					x++;
				else if (action[0] < -0.333f)
					x--;
			}
		}
	}

	std::cout << successes << " " << failures << std::endl;
}*/

/*struct Image {
	std::vector<unsigned char> _image;
};

int reverseInt(int i) {
	unsigned char c1, c2, c3, c4;

	c1 = i & 255;
	c2 = (i >> 8) & 255;
	c3 = (i >> 16) & 255;
	c4 = (i >> 24) & 255;

	return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

void getMNISTImages(std::vector<Image> &images, const std::string &filename, int numUse) {
	std::ifstream file(filename, std::ios::binary);

	if (file.is_open()) {
		int magicNumber = 0;
		int nImages = 0;
		int nRows = 0;
		int nCols = 0;

		file.read((char*)&magicNumber, sizeof(int));
		magicNumber = reverseInt(magicNumber);
		file.read((char*)&nImages, sizeof(int));
		nImages = reverseInt(nImages);
		file.read((char*)&nRows, sizeof(int));
		nRows = reverseInt(nRows);
		file.read((char*)&nCols, sizeof(int));
		nCols = reverseInt(nCols);

		assert(numUse <= nImages);

		images.resize(numUse);

		for (int i = 0; i < numUse; i++) {
			assert(file.good());

			images[i]._image.resize(nRows * nCols);

			file.read((char*)&(images[i]._image[0]), sizeof(unsigned char) * images[i]._image.size());
		}
	}

	file.close();
}

void getMNISTLabels(std::vector<unsigned char> &labels, const std::string &filename, int numUse) {
	std::ifstream file(filename, std::ios::binary);

	if (file.is_open()) {
		int magicNumber = 0;
		int nLabels = 0;
		int nRows = 0;
		int nCols = 0;

		file.read((char*)&magicNumber, sizeof(magicNumber));
		magicNumber = reverseInt(magicNumber);
		file.read((char*)&nLabels, sizeof(nLabels));
		nLabels = reverseInt(nLabels);

		assert(numUse <= nLabels);

		labels.resize(numUse);

		for (int i = 0; i < numUse; i++) {
			unsigned char temp = 0;
			file.read((char*)&temp, sizeof(unsigned char));

			labels[i] = temp;
		}
	}

	file.close();
}

int main() {
	std::vector<Image> trainingImages;
	std::vector<unsigned char> trainingLabels;

	getMNISTImages(trainingImages, "MNIST/train-images.idx3-ubyte", 60000);
	getMNISTLabels(trainingLabels, "MNIST/train-labels.idx1-ubyte", 60000);

	std::vector<Image> testImages;
	std::vector<unsigned char> testLabels;

	getMNISTImages(testImages, "MNIST/t10k-images.idx3-ubyte", 10000);
	getMNISTLabels(testLabels, "MNIST/t10k-labels.idx1-ubyte", 10000);

	// Get list of odd and even examples
	std::vector<int> oddIndices;
	std::vector<int> evenIndices;

	for (int i = 0; i < trainingLabels.size(); i++) {
		if (trainingLabels[i] % 2 == 0)
			evenIndices.push_back(i);
		else
			oddIndices.push_back(i);
	}

	rbf::SDRRBFNetwork sdrrbfnet;

	std::mt19937 generator(time(nullptr));

	std::vector<rbf::SDRRBFNetwork::LayerDesc> layerDescs(4);

	layerDescs[0]._rbfWidth = 60;
	layerDescs[0]._rbfHeight = 60;
	layerDescs[0]._receptiveRadius = 2;
	layerDescs[0]._inhibitionRadius = 3;
	layerDescs[0]._outputMultiplier = 0.4f;

	layerDescs[1]._rbfWidth = 50;
	layerDescs[1]._rbfHeight = 50;
	layerDescs[1]._receptiveRadius = 2;
	layerDescs[1]._inhibitionRadius = 3;
	layerDescs[1]._outputMultiplier = 0.6f;

	layerDescs[2]._rbfWidth = 40;
	layerDescs[2]._rbfHeight = 40;
	layerDescs[2]._receptiveRadius = 2;
	layerDescs[2]._inhibitionRadius = 3;
	layerDescs[2]._outputMultiplier = 0.8f;

	layerDescs[3]._rbfWidth = 30;
	layerDescs[3]._rbfHeight = 30;
	layerDescs[3]._receptiveRadius = 2;
	layerDescs[3]._inhibitionRadius = 3;
	layerDescs[3]._outputMultiplier = 1.0f;

	sdrrbfnet.createRandom(28, 28, layerDescs, 10, 0.05f, 0.4f, 0.1f, 0.3f, -0.02f, 0.02f, generator);

	std::vector<float> inputf(28 * 28);
	std::vector<float> outputf(10);

	std::uniform_int_distribution<int> selectionDist(0, trainingImages.size() - 1);
	std::uniform_int_distribution<int> selectionDistEven(0, evenIndices.size() - 1);
	std::uniform_int_distribution<int> selectionDistOdd(0, oddIndices.size() - 1);

	int totalIter = 60000;

	// Train on odds
	for (int i = 0; i < totalIter; i++) {
		int trainIndex = oddIndices[selectionDistOdd(generator)];

		for (int j = 0; j < trainingImages[trainIndex]._image.size(); j++)
			inputf[j] = trainingImages[trainIndex]._image[j] / 255.0f;

		sdrrbfnet.getOutput(inputf, outputf, 2.0f, 8.0f, 8.0f, 0.001f, 0.005f, 0.0f, 0.0f, generator);

		std::vector<float> target(10, 0.0f);

		target[trainingLabels[trainIndex]] = 1.0f;

		sdrrbfnet.update(inputf, outputf, target, 0.003f, 0.1f, 0.1f, 0.5f, 0.0001f, 0.05f);

		if (i % 100 == 0)
			std::cout << "Iter: " << i << " / " << totalIter << std::endl;
	}

	// Train on evens
	for (int i = 0; i < totalIter; i++) {
		int trainIndex = evenIndices[selectionDistOdd(generator)];

		for (int j = 0; j < trainingImages[trainIndex]._image.size(); j++)
			inputf[j] = trainingImages[trainIndex]._image[j] / 255.0f;

		sdrrbfnet.getOutput(inputf, outputf, 2.0f, 8.0f, 8.0f, 0.001f, 0.005f, 0.01f, 0.25f, generator);

		std::vector<float> target(10, 0.0f);

		target[trainingLabels[trainIndex]] = 1.0f;

		sdrrbfnet.update(inputf, outputf, target, 0.003f, 0.1f, 0.1f, 0.5f, 0.0001f, 0.05f);

		if (i % 100 == 0)
			std::cout << "Iter: " << i << " / " << totalIter << std::endl;
	}

	int wrongs = 0;
	int oddWrongs = 0;
	int totalOdds = 0;

	for (int i = 0; i < testImages.size(); i++) {
		int trainIndex = i;

		for (int j = 0; j < testImages[trainIndex]._image.size(); j++)
			inputf[j] = testImages[trainIndex]._image[j] / 255.0f;

		sdrrbfnet.getOutput(inputf, outputf, 2.0f, 8.0f, 8.0f, 0.001f, 0.005f, 0.005f, 0.1f, generator);

		int maxIndex = 0;

		for (int j = 0; j < outputf.size(); j++)
		if (outputf[j] > outputf[maxIndex])
			maxIndex = j;

		if (maxIndex != testLabels[trainIndex])
			wrongs++;

		if (testLabels[trainIndex] % 2 == 1) {
			totalOdds++;

			if (maxIndex != testLabels[trainIndex])
				oddWrongs++;
		}

		std::cout << "Result: " << maxIndex << " Actual: " << static_cast<int>(testLabels[trainIndex]) << std::endl;
	}

	std::cout << "Total Error: " << (static_cast<float>(wrongs) / static_cast<float>(testImages.size())) * 100.0f << std::endl;
	std::cout << "Odd Error: " << (static_cast<float>(oddWrongs) / static_cast<float>(totalOdds)) * 100.0f << std::endl;

	return 0;
}*/