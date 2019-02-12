//
//  main.cpp
//  NeuralNetwork
//
//  Created by  IvanGorbunov on 30/01/2019.
//  Copyright © 2019 Ivan Gorbunov. All rights reserved.
//
#include "Net.cpp"
#include <iostream>
#include <chrono>
double Rand(double a){
    srand(static_cast<unsigned int>(time(0)));
    return 2*a*0.00001*(rand()%100000)-a;
}

int main(int argc, const char * argv[]) {
    Net<double> & k=Net<double>::CreateNet();
    ActFContainer<double> &j=ActFContainer<double>::Instance();
    k.MakeNet({&j.linear,&j.tansig,&j.purelin}, {784,200,10});
    k.addInputToLayer({3,4,5});
    k.FeedForvard();
    k.BackProp({7,8,9});
    return 0;
}
