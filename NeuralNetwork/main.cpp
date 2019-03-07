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
#include "MNISTParser.hpp"
#include "mnist_utils.hpp"
double Rand(double a){
    return 2*a*0.00001*(rand()%100000)-a;
}

double f(double a){
    return sin(a)*2+5;
}



int main(int argc, const char * argv[]) {
   
    
    
    NetBuilder<double> * builder=new GdBuilder<double>;
    Director<double> dir;
   // Net<double> & net=dir.createWithWeightsFromINI(*builder, "myfile.ini","Weights1");
    
  Net<double> & net=dir.createWithRandomWeightsFromINI(*builder, "myfile.ini");
   /*
    auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
    normalize_dataset(dataset);
    dataset.resize_training(3000);
    vector<vector<double>>training_labels;
   unsigned int k;
    for(int i=0;i<dataset.training_labels.size();i++){
        k=dataset.training_labels[i];
        training_labels.push_back(vector<double>(10));
        training_labels[i][k]=1;
        
    }
    vector<double> a;
    vector<vector<double>> training_images;
    for(int i=0;i<dataset.training_images.size();i++){
        for(int j=0;j<dataset.training_images[5].size();j++){
            a.push_back((double)dataset.training_images[i][j]);
        }
        training_images.push_back(a);
        a.clear(); 
    }
    
  // cout<< net.count_accuracy(training_images, training_labels);
   net.Fit(training_images, training_labels,4,false,128);
    //cout<< net.count_accuracy(training_images, training_labels);  */
    
/*#if 0
    double biasarr[200];
    FILE* fff = fopen("/Users/valyagorbunova/Library/Developer/Xcode/DerivedData/NeuralNetwork-gtsewaeridktwjbexvbrpkbvjuix/Build/Products/Debug/net_train_weight2.bin","rb");
    assert(fff!=NULL);
    size_t ret = fread(biasarr,sizeof(double),200,fff);
    assert(ret==200);
#endif
    
    ifstream in;
    in.open("/Users/valyagorbunova/Library/Developer/Xcode/DerivedData/NeuralNetwork-gtsewaeridktwjbexvbrpkbvjuix/Build/Products/Debug/net_train_weight2.bin", ios::in|ios::binary);
    
    if(!in.is_open()) abort();
    double w[200];
    in.read((char*)w,sizeof(double)*200);*/
    return 0;
}
