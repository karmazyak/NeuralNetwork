//
//  Neuron.cpp
//  NeuralNetwork
//
//  Created by  IvanGorbunov on 02/02/2019.
//  Copyright © 2019 Ivan Gorbunov. All rights reserved.
//

#include "Neuron.hpp"



template <typename T,class F> Neuron<T,F>::Neuron(unsigned int index,unsigned int outnum,double b):m_myIndex(index),num_of_outputs(outnum),m_outputVal(0),bias(b){
    for(int k=0;k<outnum;k++){
        m_outputWeights.push_back(Connection(outnum));
    }
    if(outnum==0)typeN=1;
    else typeN=0;
    
};

template <typename T,class F>
void Neuron<T,F>::countOut(vector<Neuron<T,F>> &prevLayer){
    T sum=this->bias;
    for(auto it=prevLayer.begin();it<prevLayer.end();it++){
        sum+=it->m_outputVal*it->m_outputWeights[this->m_myIndex].weight;
    }
    this->sum=sum;
    this->m_outputVal=this->actf(sum);
}


template <typename T,class F>
void Neuron<T,F>::updateW(vector<double> w){
    for(unsigned int i=0;i<min(w.size(),m_outputWeights.size());i++){
        m_outputWeights[i].weight=w[i];
    }
}

template <typename T>
struct Sum {
    Sum() { sum = 0; }
    void operator()(Connection k) { sum +=k.deltaWeight*(k.weight-k.deltaWeight); }
    void zero (){sum=0;}
    T sum;
};
