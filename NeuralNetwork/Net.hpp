//
//  Net.hpp
//  NeuralNetwork
//
//  Created by  IvanGorbunov on 30/01/2019.
//  Copyright © 2019 Ivan Gorbunov. All rights reserved.
//

#ifndef Net_hpp
#define Net_hpp
#include "Neuron.cpp"

template <typename T>

class Net {
private:
    void UpdateLayerW(vector<vector<double>> w,unsigned int LayerInd);
    vector<T> out;
    void RememberOut();
    vector<vector<Neuron<T>>> m_layer;
    unsigned int input_neurons_num;
    unsigned int hiden_neur_num;
    double factor_mashtab;
    Net(){}
    Net& operator= (Net&)=delete;
    Net(Net &)=delete;
public:
    double learning_rate;
    void addLayer(ActF<T>* ActFunc,unsigned int num_of_neurons,unsigned int out_num,double b=0);
    void CountErrorLastLayer(vector<T> target);
    void addInputToLayer(vector<T> input);
    void FeedForvard();
    void UpdateAllNet(vector<vector<vector<double>>> w);
    void PrintOut() const;
    void MakeNet(vector<ActF<T> *>,vector<unsigned int> NetConf,double rate=0.1,double b=-999999);
    void BackProp(vector<T> target);
    void CountErrorLayer(unsigned int currLayerInd);
    static Net& CreateNet(){
        static Net k;
        return k;
    }
    
};

#endif /* Net_hpp */
