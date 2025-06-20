#include "main.h"
#include "vexlog/logger.hpp"

vexmaps::logger::PFLogger<500> logger;

void initialize() {
}

void disabled() {}
void competition_initialize() {}
void autonomous() {}


void opcontrol() {
	while (true) {
        // simulate the particle filter
		pros::delay(20);
	}
}
