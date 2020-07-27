//
//  main.cpp
//  ExtendedKalmanFilter
//
//  Created by Karan on 4/6/18.
//  Copyright 婕? 2018 Karan. All rights reserved.
//

#include <iostream>
#include <stdio.h>
#include <fstream>
#include <sstream>
//#include <windows.h>
# include <unistd.h>
#include "fusion.hpp"
//#include 
#include <sstream>
//#include "parameterReader.hpp"
#include "geo_ned.hpp"
#include "run_fusion.hpp"
#include<cstdio>
using namespace std;
using namespace geodectic_converter;


double StringToDouble(string Input)
{
    double Result;
    stringstream Oss;
    Oss<<Input;
    Oss>>Result;
    return Result;
}

int main(int argc, const char * argv[])
{
    char *savePath = "teststate.txt";
	char *savePath2 = "state6.txt";
    if(remove(savePath)==0&&remove(savePath2) == 0)
    {
        cout<<"remove sucesses"<<endl;
    }
    else
    {
        cout<<"remove fail"<<endl;
    }
//    std::ifstream ip("./data.csv");
    std::ifstream ip("./2014-03-26-000-Data.csv");
//    std::ifstream ip("./2014-02-14-002-Data.csv");
    if(!ip.is_open())
    {
        std::cerr << "Failed to open the data file";
        std::exit(EXIT_FAILURE);
    }

    std::string timestamp;
    double ax;
    double yaw_rate;
    double yaw;
    std::string str;
    double tt,latitude,longitude,altitude;
    double x=0,y=0,z=0;
    double course,speed;
    std::getline(ip, str); // Skip the first line

    GeodecticConverter tp1;
    GpsIns tp2(true);

    int cn = 1;
    FILE *fd;
    fd = fopen("state.save.txt","wt");
    while(std::getline(ip, str))
    {
        std::istringstream iss(str);
        std::string token;

        int cunt = 1;

        while (std::getline(iss, token, ','))
        {
            double tmp = StringToDouble(token);
            if(cunt == 2) tt = tmp;
            if(cunt==15) latitude = tmp;
            if(cunt==16) longitude = tmp;
            if(cunt == 17) altitude = tmp;
            if(cunt == 4) ax = tmp;
            if(cunt == 9) yaw_rate = tmp;
            if(cunt == 12) yaw = tmp;
            if(cunt  ==13) speed = tmp;
            if(cunt  ==14) course = tmp;
            cunt++;
        }
        //
        if(cn==1){
            tp1.intializeReference(latitude,longitude,altitude );
            //tp2.read_time(int(tt));
        }
//        tp2.read_encoders(course);
        tp2.read_gps_data(latitude,longitude,altitude);
        tp2.read_imu_data(yaw_rate,ax);
        tp2.loop();

        Eigen::VectorXd state(6,1);
        state = tp2.get_estimated_state();//状态量

        fprintf(fd,"%f\t%f\t%f\t%f\t%f\t%f\t",state[0],state[1],state[2],state[3],state[4],state[5]);
		/*这里存在问题，就是状态量与测量量一致，这是为什么呢？*/
        tp1.geodetic2Ned(latitude,longitude,altitude , &x,&y,&z);
        fprintf(fd,"%f\t%f\t%f\t\n",x,y,z);
        cn++;
        cout<<latitude<<longitude<<altitude<<endl;
    }
    fclose(fd);
    clock_t t1 = clock();
//    sleep(5);
    clock_t t2 = clock();
    cout<<t2-t1<<endl;

    std::cout << "endl" <<std::endl;
    cout<<1.e6<<endl;
    return 0;
}
