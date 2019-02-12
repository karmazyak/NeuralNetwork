//
//  ActFunc.hpp
//  NeuralNetwork
//
//  Created by  IvanGorbunov on 12/02/2019.
//  Copyright © 2019 Ivan Gorbunov. All rights reserved.
//

#ifndef ActFunc_hpp
#define ActFunc_hpp

#include <stdio.h>
template <typename T>
class ActF
{
public:
    
    virtual T operator() (T i)=0;
    virtual T operator[] (T i)=0;
private:
    
};



template <typename T>
class tg: public ActF<T> {
    
public:
    T operator() (T i) override{
        return tan(i);
    }
    T operator[](T i) override{
        return 1/(cos(i)*cos(i));
    }
};

template <typename T>
class linear: public ActF<T>  {
    
public:
    T operator() (T i) override{
        return i;
    }
    T operator[](T i) override{
        return 1;
    }
};

template <typename T>
class sigm: public ActF<T>  {
    
public:
    T operator() (T i) override{
        return 1/(1+exp(i*(-1)));
    }
    T operator[](T i) override{
        return (1/(1+exp(i*(-1))))*(1-1/(1+exp(i*(-1))));
    }
};

template <typename T>
class tansig : public ActF<T> {
    
public:
    T operator() (T i) override{
        return 2/(1+exp(i*(-2)))-1;
    }
    T operator[] (T i) override{
        return (4*exp(-2*i))/((1+exp(i*(-2)))*(1+exp(i*(-2))));
    }
};

template <typename T>
class purelin : public ActF<T> {
    
public:
    T operator() (T i) override{
        return i;
    }
    T operator[] (T i) override{
        return 1;
    }
};

template <typename T>
class bipsigm : public ActF<T> {
    
public:
    T operator() (T i) override{
        return 1-2/(1+exp(i*(-1)));
    }
    T operator[](T i) override{
        return (1/(1+exp(i*(-1))))*(1-1/(1+exp(i*(-1))));
    }
};

template <typename T>
class ActFContainer
{
public:
    bipsigm<T> bigsim;
    purelin<T> purelin;
    tansig<T> tansig;
    sigm<T> sigm;
    linear<T> linear;
    tg<T> tg;
    static ActFContainer& Instance()
    {
        static ActFContainer theSingleInstance;
        return theSingleInstance;
    }
private:
    ActFContainer(){
        
    };
    ActFContainer(const ActFContainer& root) = delete;
    ActFContainer& operator=(const ActFContainer&) = delete;
};


#endif /* ActFunc_hpp */
