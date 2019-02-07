//
//  Neuron.hpp
//  NeuralNetwork
//
//  Created by  IvanGorbunov on 02/02/2019.
//  Copyright © 2019 Ivan Gorbunov. All rights reserved.
//

#ifndef Neuron_hpp
#define Neuron_hpp
#include <random>
#include <stdio.h>
#include <vector>
#include <cmath>
#include <string>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <fstream>


using namespace std;


template <typename T,class F>
class Net;

template <typename T>
class tg {
    
public:
    T operator() (T i){
        return tan(i);
    }
    T operator[](T i){
        return 1/(cos(i)*cos(i));
    }
};

template <typename T>
class linear {
    
public:
    T operator() (T i){
        return i;
    }
    T operator[](T i){
        return 1;
    }
};

template <typename T>
class sigm {
    
public:
    T operator() (T i){
        return 10/(1+exp(i*(-1)));
    }
    T operator[](T i){
        return (10/(1+exp(i*(-1))))*(1-1/(1+exp(i*(-1))));
    }
};

template <typename T>
class porog {
    
public:
    T operator() (T i){
        if (i>0) return 1;
        else return 0;
    }
};

template <typename T>
class bipsigm {
    
public:
    T operator() (T i){
        return 1-2/(1+exp(i*(-1)));
    }
    T operator[](T i){
        return (1/(1+exp(i*(-1))))*(1-1/(1+exp(i*(-1))));
    }
};

class Connection
{
public:
    double weight;
    double deltaWeight;
    Connection(unsigned int num_of_out){
        deltaWeight=0;
        srand(static_cast<unsigned int>(time(0)));
        weight=1/sqrt(num_of_out)*(2*0.00001*(rand()%100000)-1);
    }
};



template <typename T=double,class F=sigm<T> >
class Neuron
{
public:
    F actf;
    T sum;
    unsigned int typeN;
    double m_outputVal;
    unsigned int m_myIndex;
    unsigned int num_of_outputs;
    vector<Connection> m_outputWeights;
    Neuron(unsigned int index,unsigned int outnum,double b=0);
    void countOut(vector<Neuron<T,F>> &prevLayer);
    void updateW(vector<double> w);
    double bias;
private:
};



#endif /* Neuron_hpp */
