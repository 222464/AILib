#include <deep/FERL.h>

#include <lstm/LSTMActorCritic.h>

#include <iostream>

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <time.h>

#include <assert.h>

const int mazeWidth = 32;
const int mazeHeight = 32;
const int mazeSize = mazeWidth * mazeHeight;
const float mazeSpan = std::sqrtf(mazeWidth * mazeWidth + mazeHeight * mazeHeight);

int maze[mazeSize] = {
	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
	1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
	1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
	1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
	1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
	1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
	1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
	1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
	1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
	1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
	1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
	1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
	1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
	1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
	1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
	1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
	1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
	1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
	1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
	1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
	1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
	1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
	1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
	1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
	1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
	1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
	1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
	1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
	1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
	1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
	1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
};

int mazeAt(int x, int y) {
	if (x < 0 || x >= mazeWidth || y < 0 || y >= mazeHeight)
		return 1;

	return maze[x + y * mazeWidth];
}

void observation(int botX, int botY, int endX, int endY, std::vector<float> &obs) {
	if (obs.size() != 8)
		obs.resize(8);

	obs[0] = static_cast<float>(botX) / mazeWidth;
	obs[1] = static_cast<float>(botY) / mazeHeight;
	obs[2] = mazeAt(botX - 1, botY);
	obs[3] = mazeAt(botX + 1, botY);
	obs[4] = mazeAt(botX, botY - 1);
	obs[5] = mazeAt(botX, botY + 1);
	obs[6] = static_cast<float>(endX - botX);
	obs[7] = static_cast<float>(endY - botY);

	// Normalize
	if (obs[6] != 0.0f || obs[7] != 0.0f) {
		float deltaDist = std::sqrtf(obs[6] * obs[6] + obs[7] * obs[7]);
		obs[6] /= deltaDist;
		obs[7] /= deltaDist;
	}
}

void executeAction(int action, int &botX, int &botY) {
	switch (action) {
	case 0:
		if (mazeAt(botX - 1, botY) != 1)
			botX = botX - 1;

		break;

	case 1:
		if (mazeAt(botX + 1, botY) != 1)
			botX = botX + 1;

		break;

	case 2:
		if (mazeAt(botX, botY - 1) != 1)
			botY = botY - 1;

		break;

	case 3:
		if (mazeAt(botX, botY + 1) != 1)
			botY = botY + 1;

		break;
	}
}

float getReward(int botX, int botY, int endX, int endY) {
	float dx = endX - botX;
	float dy = endY - botY;

	float dist = std::sqrt(dx * dx + dy * dy);

	return -dist * dist / mazeSpan + 1.0f;
}

float &getQ(int x, int y, int a, std::vector<float> &q) {
	return q[a * mazeSize + x + y * mazeWidth];
}

int main() {
	std::mt19937 generator(time(nullptr));

	std::uniform_real_distribution<float> uniformDist(0.0f, 1.0f);

	std::vector<float> q(mazeSize * 4, 10.0f);

	// Find start and end
	int startX, startY;
	int endX, endY;

	bool foundStart = false;
	bool foundEnd = false;

	for (int x = 0; x < mazeWidth; x++)
	for (int y = 0; y < mazeHeight; y++) {
		int id = mazeAt(x, y);

		if (id == 2) {
			startX = x;
			startY = y;

			foundStart = true;
		}
		else if (id == 3) {
			endX = x;
			endY = y;

			foundEnd = true;
		}
	}

	assert(foundStart && foundEnd);

	int botX = startX;
	int botY = startY;

	deep::FERL agent;

	agent.createRandom(8, 4, 32, 0.1f, generator);

	//lstm::LSTMActorCritic agent;

	//agent.createRandom(8, 4, 1, 24, 2, 1, 1, 24, 2, 1, -0.05f, 0.05f, generator);

	bool prevWasGoal = false;

	// -------------------------- Visualization ---------------------------

	sf::RenderWindow window;

	window.create(sf::VideoMode(900, 900), "Maze", sf::Style::Default);

	window.setFramerateLimit(60);

	sf::Vector2f tileSize(static_cast<float>(window.getSize().x) / mazeWidth, static_cast<float>(window.getSize().y) / mazeHeight);

	sf::Font font;

	font.loadFromFile("Resources/pixelated.ttf");

	sf::Texture wallTexture;

	wallTexture.loadFromFile("Resources/wall.png");
	
	sf::Texture floorTexture;

	floorTexture.loadFromFile("Resources/floor.png");

	sf::Texture startTexture;

	startTexture.loadFromFile("Resources/start.png");

	sf::Texture endTexture;

	endTexture.loadFromFile("Resources/end.png");

	sf::Texture botTexture;

	botTexture.loadFromFile("Resources/bot.png");

	// ---------------------------- Game Loop -----------------------------

	bool quit = false;

	sf::Clock clock;

	float dt = 0.017f;

	float prevReward = getReward(botX, botY, endX, endY);

	float prevValue = 0.0f;
	float prevMax = 0.0f;

	int prevBotX = botX;
	int prevBotY = botY;
	int prevAction = 0;

	float epsilon = 0.0f;

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

		// ---------------------------- Logic -----------------------------

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::T))
			window.setFramerateLimit(300);
		else
			window.setFramerateLimit(10);

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::E))
			epsilon = 0.0f;
		else
			epsilon = 0.2f;

		std::vector<float> obs;

		observation(botX, botY, endX, endY, obs);

		float reward;

		if (prevWasGoal) {
			prevWasGoal = false;
			reward = 16.0f;
		}
		else
			reward = getReward(botX, botY, endX, endY);

		int maxAction = 0;

		for (int a = 1; a < 4; a++) {
			float qa = getQ(botX, botY, a, q);

			if (qa > getQ(botX, botY, maxAction, q))
				maxAction = a;
		}

		int action;

		if (uniformDist(generator) < epsilon) {
			std::uniform_int_distribution<int> distA(0, 3);

			action = distA(generator);
		}
		else
			action = maxAction;

		// Update Q
		float newAdv = prevMax + (reward + 0.95f * getQ(botX, botY, maxAction, q) - prevMax) * 4.0f;

		float error = 0.01f * (newAdv - prevValue);

		prevMax = getQ(botX, botY, maxAction, q);
		prevValue = getQ(botX, botY, action, q);

		getQ(prevBotX, prevBotY, prevAction, q) += error;

		prevBotX = botX;
		prevBotY = botY;
		prevAction = action;

		std::vector<float> output;

		agent.step(obs, output, reward, 0.001f, 0.95f, 0.8f, 5.0f, 8, 8, 0.05f, 0.15f, 0.05f, generator);

		action = 0;

		for (int a = 1; a < 4; a++)
		if (output[a] > output[action])
			action = a;

		executeAction(action, botX, botY);

		if (botX == endX && botY == endY) {
			botX = startX;
			botY = startY;

			prevWasGoal = true;
		}

		prevReward = reward;

		// ---------------------------- Render ----------------------------

		sf::Sprite sprite;

		sprite.setScale(tileSize.x / wallTexture.getSize().x, tileSize.y / wallTexture.getSize().y);

		sf::Text text;

		text.setFont(font);
		text.setCharacterSize(12);
		text.setColor(sf::Color::Blue);

		// Show world
		for (int x = 0; x < mazeWidth; x++)
		for (int y = 0; y < mazeHeight; y++) {
			sprite.setPosition(x * tileSize.x, y * tileSize.y);

			switch (mazeAt(x, y)) {
			case 0:
				sprite.setTexture(floorTexture);
				break;

			case 1:
				sprite.setTexture(wallTexture);
				break;

			case 2:
				sprite.setTexture(startTexture);
				break;

			case 3:
				sprite.setTexture(endTexture);
				break;
			}

			window.draw(sprite);
			
			if (sf::Keyboard::isKeyPressed(sf::Keyboard::S))
			for (int a = 0; a < 4; a++) {
				text.setPosition(sprite.getPosition() + sf::Vector2f(0.0f, a * tileSize.y * 0.25f));
				text.setString(std::to_string(getQ(x, y, a, q)));

				window.draw(text);
			}
		}

		// Show bot
		sprite.setPosition(botX * tileSize.x, botY * tileSize.y);

		sprite.setTexture(botTexture);

		window.draw(sprite);

		text.setPosition(0.0f, 0.0f);

		std::string str = "";

		for (int a = 0; a < output.size(); a++)
			str += std::to_string(output[a]) + " ";

		text.setString(str);

		window.draw(text);

		// -------------------------------------------------------------------

		window.display();

		//dt = clock.getElapsedTime().asSeconds();
	} while (!quit);

	return 0;
}