#include "main.h"
#include "vexlog/logger.hpp"
#include "vexlog/pf_logger.hpp"
#include <cmath>
#include <random>
#include <tuple>

const size_t N = 1024;

vexmaps::logger::PFLogger<N> logger;

void initialize() { pros::c::serctl(SERCTL_DISABLE_COBS, NULL); }

void disabled() {}
void competition_initialize() {}
void autonomous() {}

void opcontrol() {
  float x[N];
  float y[N];
  float weights[N];

  std::ranlux24_base rng;

  std::uniform_real_distribution<float> x_dist(-1.28, -1.2);
  std::uniform_real_distribution<float> y_dist(0.762, 1.016);
  std::normal_distribution<float> weight_dist(0.2, 0.75);

  std::vector<std::tuple<float, float, float>> particles;

  for (int i = 0; i < N; i++) {
    particles.emplace_back(x_dist(rng), std::abs(weight_dist(rng)), y_dist(rng));
  }
  sort(particles.begin(), particles.end());

  for (int i = 0; i < N; i++) {
    x[i] = std::get<0>(particles[i]);
    y[i] = std::get<2>(particles[i]);

    weights[i] = std::get<1>(particles[i]);
  }

  auto start_time = pros::micros();
  // simulate the particle filter

  logger.particles.addParticles(x, y, weights, N);

  logger.generation_info.distance1.setData(0, 10.5, 10, 60, false);
  logger.generation_info.distance2.setData(1, 393.33, 10, 10, true);
  logger.generation_info.distance3.setData(2, 20.0, 33, 60, false);
  logger.generation_info.distance4.setData(3, 50.1, 58, 60, false);
  logger.generation_info.setData(10, 500, 0, 10, 20);

  auto end_time = pros::micros();

  // TODO: in practice the user should not have to specify a length since they
  // likely dont know
  vexmaps::logger::sendData(&logger, 10000);
  std::cout << "input data time: " << end_time - start_time << std::endl;

  while (true) {
    pros::delay(20);
  }
}
