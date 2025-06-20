#include "main.h"
#include "vexlog/logger.hpp"
#include <cmath>

vexmaps::logger::PFLogger<500> logger;

void initialize() {
    pros::c::serctl(SERCTL_DISABLE_COBS,NULL);
}

void disabled() {}
void competition_initialize() {}
void autonomous() {}


void opcontrol() {
    float x[500];
    float y[500];
    float weights[500];
    x[0] = 0.0001;
    y[0] = 0.0001;
    weights[0] = 0.0001;

    // fake some data
    for(int i = 1;i < 500;i++){
        x[i] = x[i-1]*1.25;
        y[i] = y[i-1]*1.55;
        weights[i] = weights[i-1] * 2;

        if(fabsf(x[i]) > 1.87){
            x[i] = x[i] >= 0 ? -1 : 1;
            x[i] *= 0.0001;
        }
        if(fabsf(y[i]) > 1.87){
            y[i] = y[i] >= 0 ? -1 : 1;
            y[i] *= 0.0001;
        }
        if(weights[i] > 100) weights[i] = 0.00001;
    }
    
    // simulate the particle filter
    logger.particles.addParticles(x, y, weights, 500);

    logger.generation_info.distance1.setData(0,10.5,10,60,false);
    logger.generation_info.distance2.setData(1,393.33,10,10,true);
    logger.generation_info.distance3.setData(2,20.0,33,60,false);
    logger.generation_info.distance4.setData(3,50.1,58,60,false);
    logger.generation_info.setData(10, 500, 0, 10, 20);

    vexmaps::logger::sendData(&logger);

    while(true){
        pros::delay(20);
    }
}
