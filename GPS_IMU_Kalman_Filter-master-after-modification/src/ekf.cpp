//
//  ekf.cpp
//  EKF
//
//  Created by Karan on 4/7/18.
//  Copyright 漏 2018 Karan. All rights reserved.
//
/*这里存在的一个问题是，将fusion与EKF分离开，就可以完成函数复用的问题了吗？*/
/*貌似并没有解决？*/
#include "ekf.hpp"
#include <iostream>

void EKF::start(const int nin, const Eigen::VectorXd &xin, const Eigen::MatrixXd &Pin, const Eigen::MatrixXd &Fin, const Eigen::MatrixXd &Qin)
{
    _num_states = nin;
    _I = Eigen::MatrixXd::Identity(_num_states, _num_states);
    if(this->verbose) std::cout << "    EKF: Number of states ->" << nin << "\n";
    this->_state.resize(nin);
    this->_state = xin;
    if(this->verbose) std::cout << "    EKF: Size of Input states ->" << xin.size() << "\n";
    _P = Pin;
    _JA = Fin;
    _Q = Qin;

    return;
}
void EKF::setQ(const Eigen::MatrixXd &Q_in)
{
    _Q = Q_in;
}
void EKF::updateJA(const double dt)
{
    /*******************************************
     * State Equation Update Rule
        x + v/??(??sin(?) + sin(dt??+?))
        y + v/??(cos(?) ?? cos(dt??+?))
        dt??+ ?
        dta + v??
        ??
        a
     *******************************************/
    if(this->verbose) std::cout << "Updating JA: About to update state equations" << "\n";
    if(this->verbose) std::cout << "Updating JA: size of states" << this->_state.rows() << "x" <<this->_state.cols() << "\n";

    // Updating state equations
    if(fabs(_state(4)) < 0.01){
        _state(0) = _state(0) + (_state(3) * dt) * cos(_state(2));
        if(this->verbose) std::cout << "Updating JA: state 0" << "\n";
        _state(1) = _state(1) + (_state(3) * dt) * sin(_state(2));
        if(this->verbose) std::cout << "Updating JA: state 1" << "\n";
        _state(2) = _state(2);
        if(this->verbose) std::cout << "Updating JA: state 2" << "\n";
        _state(3) = _state(3) + _state(5) * dt;
        if(this->verbose) std::cout << "Updating JA: state 3" << "\n";
        _state(4) = 0.0000001;
        if(this->verbose) std::cout << "Updating JA: state 4" << "\n";
        _state(5) = _state(5);
        if(this->verbose) std::cout << "Updating JA: state 5" << "\n";
    }else{
        _state(0) = _state(0) + (_state(3)/_state(4)) * (sin(_state(4) * dt + _state(2)) - sin(_state(2)));
        if(this->verbose) std::cout << "Updating JA: state 0" << "\n";
        _state(1) = _state(1) + (_state(3)/_state(4)) * (-cos(_state(4) * dt + _state(2)) + cos(_state(2)));
        if(this->verbose) std::cout << "Updating JA: state 1" << "\n";
        _state(2) = std::fmod((_state(2) + _state(4) * dt + M_PI), (2.0 * M_PI)) - M_PI;
        if(this->verbose) std::cout << "Updating JA: state 2" << "\n";
        _state(3) = _state(3) + _state(5) * dt;
        if(this->verbose) std::cout << "Updating JA: state 3" << "\n";
        _state(4) = _state(4);
        if(this->verbose) std::cout << "Updating JA: state 4" << "\n";
        _state(5) = _state(5);
        if(this->verbose) std::cout << "Updating JA: state 5" << "\n";
    }

    if(this->verbose) std::cout << "Updating JA: About to calculate jacobian" << "\n";
    // Calculate jacobian
    _JA =  calculate_joacobian(_state, dt);
}

void EKF::predict()
{
    // Prediction step
    _P = _JA * _P * _JA.transpose() + _Q;
}

void EKF::update(const Eigen::VectorXd& Z, const Eigen::VectorXd& Hx, const Eigen::MatrixXd &JH, const Eigen::MatrixXd &R)
{
    Eigen::MatrixXd JHT = _P * JH.transpose();
    // Temporary variable for storing this intermediate value
    Eigen::MatrixXd _S = JH * JHT  + R;
    // Compute the Kalman gain
    _K = JHT * _S.inverse();
    // Update the estimate
    Eigen::VectorXd y = Z - Hx;
	
    //xintianjia
    FILE * fd3;
    fd3 = fopen("teststate.txt","a+");
    fprintf(fd3,"%f\t%f\t%f\t%f\n",Z(0),Z(1),Hx(0),Hx(1));
    fclose(fd3);
    _state = _state + _K * y;
    // Update the error covariance
    _P = (_I - _K * JH) * _P;
}

Eigen::VectorXd EKF::get_resulting_state() const
{

    return _state;
}
