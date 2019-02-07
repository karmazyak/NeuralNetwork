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
    Net<double,sigm<double>> & k=Net<double,sigm<double>>::CreateNet();
    vector<unsigned int> Conf={30,3,30};
    vector<vector<vector<double>>> W={{{0.45,0.78},{-0.12,0.13}},{{1.5},{-2.3}}};
    vector<double> inp={1};
    k.MakeNet(Conf,0.02,0);
    k.UpdateAllNet(W);
    k.addInputToLayer(inp);
    k.FeedForvard();
    k.PrintOut() ;
    double z;
    vector<double> out;
    cout<<"qqqqqq";
    for(int i=0;i<500;i++){
        k.addInputToLayer(inp);
        k.FeedForvard();
        k.BackProp(out);
    }
    
    k.PrintOut();
    cout<<"out"<<endl;
    for(auto c:out)
        cout<<c;
    return 0;
}
