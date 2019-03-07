//
//  Builder.hpp
//  NeuralNetwork
//
//  Created by  IvanGorbunov on 16/02/2019.
//  Copyright © 2019 Ivan Gorbunov. All rights reserved.
//

#ifndef Builder_hpp
#define Builder_hpp
#include <stdlib.h>
#include <stdio.h>
#include "Net.cpp"




template<typename T>
class NetBuilder
{

public:
    Net<T>* net;
    ActFContainer<T>* functions;
    NetBuilder()  {
        net=&Net<T>::CreateNet();
        functions=&ActFContainer<T>::Instance();
    }
    virtual ~NetBuilder() {}
    virtual void createNet(vector<ActF<T> *> functs,vector<unsigned int> NetConf,double rate)=0;
    virtual void UpdateWeights(vector<vector<vector<double>>> w)=0;
    virtual void LoadWeightsFromIni(string name)=0;

};



template<typename T>
class GdBuilder: public NetBuilder<T>
{
public:
    void createNet(vector<ActF<T> *> functs,vector<unsigned int> NetConf,double rate);
    void UpdateWeights(vector<vector<vector<double>>> w);
    void LoadWeightsFromIni(string name);
    void LoadWeightsFromBinaryOneLayer(const string name,unsigned int l_num);
    
};




template<typename T>
class Director
{
public:
    Net<T>& createWithRandomWeightsFromFunc(NetBuilder<T> & builder,vector<ActF<T> *> functs,vector<unsigned int> NetConf,double rate);
    
    Net<T>& createWithWeightsFromFunc(NetBuilder<T> & builder,vector<ActF<T> *> functs,vector<unsigned int> NetConf,double rate,vector<vector<vector<double>>> w);
    Net<T>& createWithRandomWeightsFromINI(NetBuilder<T> & builder,string name);
    Net<T>& createWithWeightsFromINI(NetBuilder<T> & builder,string nameConf,string nameW);
    
};

#endif /* Builder_hpp */

