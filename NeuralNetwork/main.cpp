//
//  main.cpp
//  NeuralNetwork
//
//  Created by  IvanGorbunov on 30/01/2019.
//  Copyright © 2019 Ivan Gorbunov. All rights reserved.
//
#include "Builder.cpp"
#include <iostream>
#include <chrono>
double Rand(double a){
    return 2*a*0.00001*(rand()%100000)-a;
}

double f(double a){
    return sin(a)*2+5;
}



int main(int argc, const char * argv[]) {
    NetBuilder<double> * builder=new GdBuilder<double>;

    Director<double> dir;
    Net<double> & net=dir.createWithRandomWeightsFromINI(*builder, "myfile.ini");

    /*
    Net<double> & k=Net<double>::CreateNet();
    ActFContainer<double> &j=ActFContainer<double>::Instance();
    k.MakeNet({&j.linear,&j.sigm,&j.linear}, {1,5,1},0.02);
    k.addInputToLayer({3,4,5});
    k.FeedForvard();
    k.BackProp({7,8,9});
    double x;
    for(int d=0;d<10000;d++){
        x=Rand(1);
        k.addInputToLayer({x});
        k.FeedForvard();
        k.BackProp({f(x)});
    }
    vector<double>x1;
    vector<double>t1;
    vector<double>out;
    vector<double>del;
    for(int num=0;num<10;num++){
    x=Rand(1);
    t1.push_back(f(x));
    x1.push_back(x);
    k.addInputToLayer({x});
    k.FeedForvard();
    out.push_back(k.out[0]);
        del.push_back(k.out[0]-f(x));
    }
    mINI::INIFile file("myfile.ini");

    mINI::INIStructure ini;
    ini["layer"];
    file.write(ini);
    mINI::INIStructure ini;
    mINI::INIFile file("myfile.ini");
    ini["meta"]["rate"]="0.2";
    ini["meta"]["l_num"]="3";
    ini["layer 1"]["neuron_num"]="1";
    ini["layer 1"]["act_func"]="linear";
    ini["layer 2"]["neuron_num"]="6";
    ini["layer 2"]["act_func"]="sigm";
    ini["layer 3"]["neuron_num"]="1";
    ini["layer 3"]["act_func"]="linear";
    file.write(ini);*/
    return 0;
}
