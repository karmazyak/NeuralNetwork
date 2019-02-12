//
//  Neuron.hpp
//  NeuralNetwork
//
//  Created by  IvanGorbunov on 02/02/2019.
//  Copyright © 2019 Ivan Gorbunov. All rights reserved.
//

#ifndef Neuron_hpp
#define Neuron_hpp
#include "ActFunc.cpp"

using namespace std;


template <typename T>
class Net;



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



template <typename T=double>
class Neuron
{
public:
    ActF<T>* actf;
    T sum;
    unsigned int typeN;
    double m_outputVal;
    unsigned int m_myIndex;
    unsigned int num_of_outputs;
    vector<Connection> m_outputWeights;
    Neuron(ActF<T> *,unsigned int index,unsigned int outnum,double b=0);
    void countOut(vector<Neuron<T>> &prevLayer);
    void updateW(vector<double> w);
    double bias;
private:
};



#endif /* Neuron_hpp */
